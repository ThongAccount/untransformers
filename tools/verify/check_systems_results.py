#!/usr/bin/env python3
"""Gate check for the Milestone-3 systems comparison (G6 / G7).

Verifies that a systems-results JSON measured everything the milestone
requires, on both arms, at every context length, with the conditions recorded.
It does not judge the numbers. Exit 0 prints SYSTEMS_OK (and PAIRED_QUALITY_OK
when --require-paired-quality is given).

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

SCHEMA = "untransformers-systems-run/1"
ARMS = ("HE", "attention-32h")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


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

    cfg = data.get("config", {}) or {}
    for key in ("d", "l", "n_layers", "ffn_width", "vocab", "batch_size",
                "n_heads", "trials", "warmup", "gen_tokens"):
        check(key in cfg, f"config.{key} missing")
    l = cfg.get("l")
    batch = cfg.get("batch_size")
    d = cfg.get("d")

    # --- A. training step ------------------------------------------------
    ts_rows = data.get("training_step", []) or []
    ts_by_arm = {r.get("arm"): r for r in ts_rows}
    for arm in ARMS:
        row = ts_by_arm.get(arm)
        if row is None:
            failures.append(f"training_step: no row for arm {arm}")
            continue
        med = (row.get("step_seconds") or {}).get("median")
        check(_finite(med) and med > 0, f"training_step[{arm}]: median step time invalid ({med})")
        check(row.get("tokens_per_step") == (batch * l if batch and l else None),
              f"training_step[{arm}]: tokens_per_step {row.get('tokens_per_step')} "
              f"!= batch*l")
        check(_finite(row.get("tokens_per_second")) and row.get("tokens_per_second") > 0,
              f"training_step[{arm}]: tokens_per_second invalid")
        check("memory" in row and row["memory"], f"training_step[{arm}]: no memory row")
        notes.append(f"training_step[{arm}]: {med:.4f}s/step, "
                     f"{row.get('tokens_per_second')} tok/s, params={row.get('n_params')}")
    if all(a in ts_by_arm for a in ARMS):
        check(ts_by_arm["HE"].get("n_params") == ts_by_arm["attention-32h"].get("n_params"),
              "training_step: the two arms do not have the same parameter count")

    # --- B. prefill ------------------------------------------------------
    pre = data.get("prefill", []) or []
    pre_by_arm = {}
    for r in pre:
        pre_by_arm.setdefault(r.get("arm"), {})[r.get("context_tokens")] = r
    for arm in ARMS:
        rows = pre_by_arm.get(arm)
        if not rows:
            failures.append(f"prefill: no rows for arm {arm}")
            continue
        for ctx, row in rows.items():
            lat = (row.get("latency") or {}).get("median")
            check(_finite(lat) and lat > 0, f"prefill[{arm}@{ctx}]: median latency invalid")
            check("memory" in row and row["memory"], f"prefill[{arm}@{ctx}]: no memory row")
            if l:
                check(ctx + cfg.get("gen_tokens", 0) <= l,
                      f"prefill[{arm}@{ctx}]: context+gen exceeds the positional table l={l}")
        notes.append(f"prefill[{arm}]: contexts {sorted(rows)}")
    if all(a in pre_by_arm for a in ARMS):
        check(set(pre_by_arm["HE"]) == set(pre_by_arm["attention-32h"]),
              "prefill: the two arms were not measured at the same context lengths")

    # --- C. incremental decoding ----------------------------------------
    inc = data.get("incremental_decode", []) or []
    inc_by_arm = {}
    for r in inc:
        inc_by_arm.setdefault(r.get("arm"), {})[r.get("context_tokens")] = r
    for arm in ARMS:
        rows = inc_by_arm.get(arm)
        if not rows:
            failures.append(f"incremental_decode: no rows for arm {arm}")
            continue
        for ctx, row in rows.items():
            per = row.get("per_token_seconds") or {}
            check(_finite(per.get("median")) and per.get("median") > 0,
                  f"incremental_decode[{arm}@{ctx}]: median per-token time invalid")
            check(_finite(per.get("samples")) and per.get("samples", 0) > 0,
                  f"incremental_decode[{arm}@{ctx}]: no timing samples")
            check(_finite(row.get("tokens_per_second")) and row.get("tokens_per_second") > 0,
                  f"incremental_decode[{arm}@{ctx}]: tokens_per_second invalid")
            check("cache_bytes_at_prompt" in row,
                  f"incremental_decode[{arm}@{ctx}]: cache bytes missing")
        notes.append(f"incremental_decode[{arm}]: contexts {sorted(rows)}")
    if all(a in inc_by_arm for a in ARMS):
        check(set(inc_by_arm["HE"]) == set(inc_by_arm["attention-32h"]),
              "incremental_decode: the two arms were not measured at the same context lengths")

    # --- D. sublayer-1 scaling ------------------------------------------
    sub = data.get("sublayer_scaling", []) or []
    sub_by_arm = {}
    for r in sub:
        sub_by_arm.setdefault(r.get("arm"), {})[r.get("context_tokens")] = r
    for arm in ARMS:
        rows = sub_by_arm.get(arm)
        if not rows:
            failures.append(f"sublayer_scaling: no rows for arm {arm}")
            continue
        check(len(rows) >= 4,
              f"sublayer_scaling[{arm}]: only {len(rows)} context lengths")
        for ctx, row in rows.items():
            med = (row.get("per_token_seconds") or {}).get("median")
            check(_finite(med) and med > 0,
                  f"sublayer_scaling[{arm}@{ctx}]: median per-token time invalid")
            check(_finite(row.get("state_bytes_per_layer")),
                  f"sublayer_scaling[{arm}@{ctx}]: state bytes missing")
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

    # --- E. weight traffic ----------------------------------------------
    wt = data.get("weight_traffic", {}) or {}
    for arm in ARMS:
        row = wt.get(arm)
        if row is None:
            failures.append(f"weight_traffic: no row for arm {arm}")
            continue
        check(_finite(row.get("parameter_bytes_fp32")) and row.get("parameter_bytes_fp32", 0) > 0,
              f"weight_traffic[{arm}]: parameter bytes invalid")
        check(_finite(row.get("training_step_parameter_bytes_read")),
              f"weight_traffic[{arm}]: training-step traffic missing")
        check(_finite(row.get("decode_step_parameter_bytes_read")),
              f"weight_traffic[{arm}]: decode-step traffic missing")
    if all(a in wt for a in ARMS):
        check(wt["HE"].get("n_params") == wt["attention-32h"].get("n_params"),
              "weight_traffic: the two arms do not carry the same parameter count")
        notes.append(f"weight_traffic: params={wt['HE'].get('n_params')} per arm")

    # --- F. paired quality ----------------------------------------------
    paired = data.get("paired_quality", {}) or {}
    if args.require_paired_quality:
        for arm in ARMS:
            seeds = paired.get(arm)
            if not seeds:
                failures.append(f"paired_quality: no checkpoints for arm {arm}")
                continue
            for seed, row in seeds.items():
                check(_finite(row.get("final_held_out_loss")),
                      f"paired_quality[{arm} seed {seed}]: final held-out loss missing")
                check(isinstance(row.get("steps"), int) and row.get("steps", 0) > 0,
                      f"paired_quality[{arm} seed {seed}]: steps missing")
        if all(a in paired for a in ARMS):
            check(set(paired["HE"]) == set(paired["attention-32h"]),
                  "paired_quality: the two arms do not carry the same seeds")
        notes.append("paired_quality seeds: " +
                     ", ".join(f"{a}={sorted(paired.get(a, {}))}" for a in ARMS))

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
