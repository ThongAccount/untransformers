#!/usr/bin/env python3
"""Gate check for the Milestone-3 systems comparison (G6 / G7).

Verifies that a systems-results JSON measured everything the milestone
requires, on every arm (including the optimized SDPA baseline), at every
context length, with the conditions recorded. It does not judge the numbers.

Since protocol revision 2 (`docs/milestone-3/PROTOCOL.md` §4.3, S1–S10) the
check also requires the *evidence* the write-up depends on:

  * the optimized baseline `attention-32h-sdpa` is timed in every block;
  * every row carries its units and its quality level (S8, S10);
  * the run order is palindromic and every arm has an A/A null (S2);
  * the numerics flags are recorded (S5);
  * the traffic rows are labelled analytic and carry their formulas (S7);
  * the decode rows separate the first token from the steady state (S4).

Exit 0 prints SYSTEMS_OK (and PAIRED_QUALITY_OK when --require-paired-quality
is given).

Usage:
    python tools/verify/check_systems_results.py --file results/systems.json
    python tools/verify/check_systems_results.py --file results/systems.json \
        --require-paired-quality
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

SCHEMA = "untransformers-systems-run/2"
ARMS = ("HE", "attention-32h", "attention-32h-sdpa")
QUALITY_ARMS = ("HE", "attention-32h")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _has_stats(row: Dict, key: str) -> bool:
    st = row.get(key) or {}
    return (_finite(st.get("median")) and st.get("median") > 0
            and _finite(st.get("n")) and st.get("n", 0) > 0
            and _finite(st.get("p10")) and _finite(st.get("p90"))
            and _finite(st.get("min")))


def _slot_median(row: Dict):
    """The median of one measurement slot, whatever block it came from."""
    for key in ("latency_stats", "latency", "step_seconds",
                "steady_state_per_token_seconds"):
        block = row.get(key)
        if isinstance(block, dict) and block.get("median") is not None:
            return float(block["median"])
    return None


def _recompute_same_context_null(slots: List[Dict]):
    """Recompute the A/A null from the slot rows.

    The null is the same arm measured at the start and at the end of the
    palindromic order **at the same context**. Returns the per-(arm, context)
    spreads and the per-context floor (the largest spread over the arms).
    """
    groups: Dict = {}
    for r in slots:
        groups.setdefault((r.get("arm"), r.get("context_tokens")), []).append(r)
    spread: Dict = {}
    for (arm, ctx), rows in groups.items():
        rows = sorted(rows, key=lambda r: r.get("order_index", 0))
        if len(rows) < 2:
            continue
        first = _slot_median(rows[0])
        last = _slot_median(rows[-1])
        if first is None or last is None:
            continue
        spread[(arm, ctx)] = abs(last - first)
    floors: Dict = {}
    for (_arm, ctx), value in spread.items():
        key = str(ctx)
        floors[key] = max(floors.get(key, 0.0), value)
    return spread, floors


def _quality_block_ok(row: Dict) -> bool:
    """A row must say which quality level it was measured at (S8)."""
    q = row.get("quality")
    if not isinstance(q, dict):
        return False
    if q.get("loaded"):
        return (_finite(q.get("final_held_out_loss"))
                and isinstance(q.get("steps"), int) and q.get("steps", 0) > 0
                and bool(q.get("quality_level")))
    return bool(q.get("quality_level")) and "initialisation" in q["quality_level"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--require-paired-quality", action="store_true")
    args = ap.parse_args()

    failures: list[str] = []
    notes: list[str] = []

    path = Path(args.file)
    if not path.exists():
        print(f"FAIL: file not found: {path}")
        return 1
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {path} is not valid JSON: {exc}")
        return 1

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    check(data.get("schema") == SCHEMA,
          f"schema is {data.get('schema')!r}, expected {SCHEMA!r}")
    check("created_utc" in data, "created_utc missing")

    env = data.get("environment", {}) or {}
    for key in ("python", "platform", "torch", "device", "torch_threads"):
        check(key in env, f"environment.{key} missing")
    notes.append(f"device={env.get('device')} gpu={env.get('gpu', '-')} "
                 f"torch={env.get('torch')} threads={env.get('torch_threads')}")

    # --- S5: numerics settings must be recorded ----------------------------
    num = env.get("numerics") or {}
    for key in ("allow_tf32_matmul", "allow_tf32_cudnn", "float32_matmul_precision",
                "cudnn_benchmark", "cudnn_deterministic"):
        check(key in num, f"environment.numerics.{key} missing (S5)")
    check(env.get("git_commit"), "environment.git_commit missing (the run must name its code)")
    check(bool(env.get("arms")), "environment.arms missing")

    cfg = data.get("config", {}) or {}
    for key in ("d", "l", "n_layers", "ffn_width", "vocab", "batch_size",
                "n_heads", "trials", "warmup", "gen_tokens", "decode_trials",
                "order_schedule"):
        check(key in cfg, f"config.{key} missing")
    l = cfg.get("l")
    batch = cfg.get("batch_size")
    d = cfg.get("d")
    check(isinstance(cfg.get("trials"), int) and cfg.get("trials", 0) >= 10,
          f"config.trials={cfg.get('trials')} is below the S3 minimum of 10")
    check(isinstance(cfg.get("decode_trials"), int) and cfg.get("decode_trials", 0) >= 5,
          f"config.decode_trials={cfg.get('decode_trials')} is below the S3 minimum of 5")
    sched = cfg.get("order_schedule") or []
    check(sched == list(ARMS) + list(reversed(ARMS)),
          f"config.order_schedule {sched} is not the palindromic ABBA order {list(ARMS) + list(reversed(ARMS))}")

    # --- A. training step --------------------------------------------------
    ts_rows = data.get("training_step", []) or []
    ts_by_arm = {r.get("arm"): r for r in ts_rows}
    for arm in ARMS:
        row = ts_by_arm.get(arm)
        if row is None:
            failures.append(f"training_step: no row for arm {arm} (S1: the optimized "
                            f"baseline must be timed)")
            continue
        check(_has_stats(row, "step_seconds"),
              f"training_step[{arm}]: median/p10/p90/min/n missing or invalid (S3)")
        check(row.get("tokens_per_step") == (batch * l if batch and l else None),
              f"training_step[{arm}]: tokens_per_step {row.get('tokens_per_step')} "
              f"!= batch*l")
        check(_finite(row.get("tokens_per_second")) and row.get("tokens_per_second") > 0,
              f"training_step[{arm}]: tokens_per_second invalid")
        check("memory" in row and row["memory"], f"training_step[{arm}]: no memory row")
        check(bool((row.get("memory") or {}).get("scope")),
              f"training_step[{arm}]: memory row has no scope label (S6)")
        check(bool(row.get("units")), f"training_step[{arm}]: no units (S10)")
        check(_quality_block_ok(row),
              f"training_step[{arm}]: no quality level attached to the timed model (S8)")
        check((row.get("slots") or 0) >= 2,
              f"training_step[{arm}]: only {row.get('slots')} measurement slot(s) — "
              f"the A/A null needs the arm measured twice (S2)")
        med = (row.get("step_seconds") or {}).get("median")
        notes.append(f"training_step[{arm}]: {med:.4f}s/step, "
                     f"{row.get('tokens_per_second')} tok/s, params={row.get('n_params')}, "
                     f"quality={'checkpoint' if (row.get('quality') or {}).get('loaded') else 'init'}")
    if all(a in ts_by_arm for a in ARMS):
        counts = {a: ts_by_arm[a].get("n_params") for a in ARMS}
        check(len(set(counts.values())) == 1,
              f"training_step: the arms do not have the same parameter count: {counts}")

    # --- S2: order effect / A/A null ---------------------------------------
    order = data.get("order_effect", {}) or {}
    slot_keys = {"training_step": "training_step_slots",
                 "prefill": "prefill_slots",
                 "incremental_decode": "incremental_decode_slots"}
    for block in ("training_step", "prefill", "incremental_decode"):
        oe = order.get(block) or {}
        check(bool(oe), f"order_effect.{block} missing (S2)")
        if not oe:
            continue
        by_ctx = oe.get("aa_null_by_context") or {}
        check(bool(by_ctx),
              f"order_effect.{block}: no same-context A/A null (S2) — the null must "
              f"compare the arm's first and last slot AT THE SAME CONTEXT, otherwise "
              f"it measures the growth of the latency with the context")
        floors = oe.get("floor_by_context_seconds") or {}
        check(bool(floors), f"order_effect.{block}: no per-context drift floor (S2)")
        slots = data.get(slot_keys[block]) or []
        spread, recomputed = _recompute_same_context_null(slots)
        check(bool(spread),
              f"order_effect.{block}: no slot rows to recompute the null from (S2)")
        for (arm, ctx), value in sorted(spread.items(), key=lambda kv: str(kv[0])):
            entry = (by_ctx.get(str(ctx)) or {}).get(arm)
            if not entry:
                failures.append(f"order_effect.{block}: no A/A null for arm {arm} "
                                f"at context {ctx} (S2)")
                continue
            check(_finite(entry.get("difference_seconds")),
                  f"order_effect.{block}[{arm}@{ctx}]: A/A difference missing")
            check(abs(abs(entry.get("difference_seconds", 0.0)) - value) <= 1e-12,
                  f"order_effect.{block}[{arm}@{ctx}]: the A/A difference "
                  f"{entry.get('difference_seconds')!r} is not the same-context null "
                  f"recomputed from the slot rows ({value!r})")
        for ctx, floor in sorted(recomputed.items()):
            check(abs((floors.get(ctx) or 0.0) - floor) <= 1e-12,
                  f"order_effect.{block}: the drift floor at context {ctx} is "
                  f"{floors.get(ctx)!r}, but the same-context null recomputed from "
                  f"the slot rows gives {floor!r}")
        check(_finite(oe.get("drift_floor_seconds")),
              f"order_effect.{block}: drift floor missing (S2)")
        if recomputed:
            check(abs((oe.get("drift_floor_seconds") or 0.0)
                      - max(recomputed.values())) <= 1e-12,
                  f"order_effect.{block}: the block drift floor is not the largest "
                  f"per-context floor")
    notes.append("order_effect drift floors: " + ", ".join(
        f"{b}={(order.get(b) or {}).get('drift_floor_seconds')}"
        for b in ("training_step", "prefill", "incremental_decode")))

    # --- B. prefill --------------------------------------------------------
    pre = data.get("prefill", []) or []
    pre_by_arm: Dict[str, Dict] = {}
    for r in pre:
        pre_by_arm.setdefault(r.get("arm"), {})[r.get("context_tokens")] = r
    for arm in ARMS:
        rows = pre_by_arm.get(arm)
        if not rows:
            failures.append(f"prefill: no rows for arm {arm} (S1)")
            continue
        for ctx, row in rows.items():
            check(_has_stats(row, "latency"),
                  f"prefill[{arm}@{ctx}]: median/p10/p90/min/n missing or invalid")
            check("memory" in row and row["memory"], f"prefill[{arm}@{ctx}]: no memory row")
            check(bool((row.get("memory") or {}).get("scope")),
                  f"prefill[{arm}@{ctx}]: memory row has no scope label (S6)")
            check(bool(row.get("units")), f"prefill[{arm}@{ctx}]: no units (S10)")
            check(_quality_block_ok(row),
                  f"prefill[{arm}@{ctx}]: no quality level attached (S8)")
            if l:
                check(ctx + cfg.get("gen_tokens", 0) <= l,
                      f"prefill[{arm}@{ctx}]: context+gen exceeds the positional table l={l}")
        notes.append(f"prefill[{arm}]: contexts {sorted(rows)}")
    if all(a in pre_by_arm for a in ARMS):
        check(set(pre_by_arm["HE"]) == set(pre_by_arm["attention-32h"]),
              "prefill: HE and attention-32h were not measured at the same context lengths")
        check(set(pre_by_arm["HE"]) == set(pre_by_arm["attention-32h-sdpa"]),
              "prefill: HE and attention-32h-sdpa were not measured at the same context lengths")

    # --- C. incremental decoding -------------------------------------------
    inc = data.get("incremental_decode", []) or []
    inc_by_arm: Dict[str, Dict] = {}
    for r in inc:
        inc_by_arm.setdefault(r.get("arm"), {})[r.get("context_tokens")] = r
    for arm in ARMS:
        rows = inc_by_arm.get(arm)
        if not rows:
            failures.append(f"incremental_decode: no rows for arm {arm} (S1)")
            continue
        for ctx, row in rows.items():
            check(_has_stats(row, "steady_state_per_token_seconds"),
                  f"incremental_decode[{arm}@{ctx}]: steady-state per-token stats missing (S4)")
            check(_has_stats(row, "first_token_seconds"),
                  f"incremental_decode[{arm}@{ctx}]: first-token stats missing (S4)")
            check(_finite(row.get("tokens_per_second")) and row.get("tokens_per_second") > 0,
                  f"incremental_decode[{arm}@{ctx}]: tokens_per_second invalid")
            check("cache_bytes_at_prompt" in row,
                  f"incremental_decode[{arm}@{ctx}]: cache bytes missing")
            check(bool(row.get("units")), f"incremental_decode[{arm}@{ctx}]: no units (S10)")
            check(_quality_block_ok(row),
                  f"incremental_decode[{arm}@{ctx}]: no quality level attached (S8)")
        notes.append(f"incremental_decode[{arm}]: contexts {sorted(rows)}")
    if all(a in inc_by_arm for a in ARMS):
        check(set(inc_by_arm["HE"]) == set(inc_by_arm["attention-32h"]),
              "incremental_decode: HE and attention-32h were not measured at the same contexts")
        check(set(inc_by_arm["HE"]) == set(inc_by_arm["attention-32h-sdpa"]),
              "incremental_decode: HE and attention-32h-sdpa were not measured at the same contexts")

    # --- D. sublayer-1 scaling ---------------------------------------------
    sub = data.get("sublayer_scaling", []) or []
    sub_by_arm: Dict[str, Dict] = {}
    for r in sub:
        sub_by_arm.setdefault(r.get("arm"), {})[r.get("context_tokens")] = r
    for arm in ARMS:
        rows = sub_by_arm.get(arm)
        if not rows:
            failures.append(f"sublayer_scaling: no rows for arm {arm} (S1)")
            continue
        check(len(rows) >= 4,
              f"sublayer_scaling[{arm}]: only {len(rows)} context lengths")
        for ctx, row in rows.items():
            check(_has_stats(row, "per_token_seconds"),
                  f"sublayer_scaling[{arm}@{ctx}]: median/p10/p90/min/n missing (S3)")
            check(_finite(row.get("state_bytes_per_layer")),
                  f"sublayer_scaling[{arm}@{ctx}]: state bytes missing")
            check("analytic" in (row.get("state_bytes_label") or ""),
                  f"sublayer_scaling[{arm}@{ctx}]: state bytes are not labelled analytic (S7)")
            check(bool(row.get("state_bytes_formula")),
                  f"sublayer_scaling[{arm}@{ctx}]: state bytes carry no formula (S7)")
            check(row.get("scope") == "sublayer-1",
                  f"sublayer_scaling[{arm}@{ctx}]: scope label missing (S9)")
        notes.append(f"sublayer_scaling[{arm}]: contexts {sorted(rows)}")
    if "HE" in sub_by_arm and "attention-32h" in sub_by_arm:
        he_states = {c: r.get("state_bytes_per_layer") for c, r in sub_by_arm["HE"].items()}
        at_states = {c: r.get("state_bytes_per_layer") for c, r in sub_by_arm["attention-32h"].items()}
        check(len(set(he_states.values())) == 1,
              f"sublayer_scaling[HE]: state bytes are not constant across contexts: {he_states}")
        shared = sorted(set(he_states) & set(at_states))
        if shared:
            first, last = shared[0], shared[-1]
            check((at_states[last] or 0) > (at_states[first] or 0),
                  "sublayer_scaling[attention-32h]: KV cache does not grow with context")
            check(he_states[first] == he_states[last],
                  "sublayer_scaling[HE]: state size changed with context")

    # --- E. traffic ---------------------------------------------------------
    wt = data.get("weight_traffic", {}) or {}
    for arm in ARMS:
        row = wt.get(arm)
        if row is None:
            failures.append(f"weight_traffic: no row for arm {arm} (S1)")
            continue
        check(_finite(row.get("parameter_bytes_fp32")) and row.get("parameter_bytes_fp32", 0) > 0,
              f"weight_traffic[{arm}]: parameter bytes invalid")
        check(_finite(row.get("training_step_parameter_bytes_read")),
              f"weight_traffic[{arm}]: training-step traffic missing")
        check(_finite(row.get("decode_step_parameter_bytes_read")),
              f"weight_traffic[{arm}]: decode-step traffic missing")
        check("analytic" in (row.get("label") or ""),
              f"weight_traffic[{arm}]: row is not labelled analytic (S7)")
        check(bool(row.get("parameter_bytes_formula"))
              and bool(row.get("training_step_parameter_bytes_formula"))
              and bool(row.get("decode_step_parameter_bytes_formula")),
              f"weight_traffic[{arm}]: traffic rows carry no formulas (S7)")
    if all(a in wt for a in ARMS):
        check(wt["HE"].get("n_params") == wt["attention-32h"].get("n_params"),
              "weight_traffic: HE and attention-32h do not carry the same parameter count")
        notes.append(f"weight_traffic: params={wt['HE'].get('n_params')} per arm")

    stt = data.get("state_traffic", {}) or {}
    for arm in ARMS:
        row = stt.get(arm)
        if row is None:
            failures.append(f"state_traffic: no row for arm {arm} (S7)")
            continue
        check("analytic" in (row.get("label") or ""),
              f"state_traffic[{arm}]: row is not labelled analytic (S7)")
        check(bool(row.get("state_bytes_per_layer_formula")),
              f"state_traffic[{arm}]: no formula (S7)")
        check(bool(row.get("per_context")),
              f"state_traffic[{arm}]: no per-context rows (S7)")
    if "HE" in stt and "attention-32h" in stt:
        check(stt["HE"].get("grows_with_context") is False,
              "state_traffic[HE]: HE's per-token state is flat, not growing")
        check(stt["attention-32h"].get("grows_with_context") is True,
              "state_traffic[attention-32h]: the KV cache grows with the context")

    # --- F. paired quality --------------------------------------------------
    paired = data.get("paired_quality", {}) or {}
    if args.require_paired_quality:
        for arm in QUALITY_ARMS:
            seeds = paired.get(arm)
            if not seeds:
                failures.append(f"paired_quality: no checkpoints for arm {arm}")
                continue
            for seed, row in seeds.items():
                check(_finite(row.get("final_held_out_loss")),
                      f"paired_quality[{arm} seed {seed}]: final held-out loss missing")
                check(isinstance(row.get("steps"), int) and row.get("steps", 0) > 0,
                      f"paired_quality[{arm} seed {seed}]: steps missing")
        if all(a in paired for a in QUALITY_ARMS):
            check(set(paired["HE"]) == set(paired["attention-32h"]),
                  "paired_quality: the two arms do not carry the same seeds")
        notes.append("paired_quality seeds: " +
                     ", ".join(f"{a}={sorted(paired.get(a, {}))}" for a in QUALITY_ARMS))

    for note in notes:
        print(f"[info] {note}")
    if failures:
        print(f"FAIL: {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1

    if args.require_paired_quality:
        print(f"PAIRED_QUALITY_OK: {path} carries quality checkpoints for both arms")
    print(f"SYSTEMS_OK: {path} "
          f"(d={d} l={l} batch={batch} device={env.get('device')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
