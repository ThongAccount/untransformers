#!/usr/bin/env python3
"""Gate check for a Milestone-2 quality run (G3 pilot / G4 GPU sweep).

Verifies STRUCTURE and INTERNAL CONSISTENCY of a quality-results JSON — it
does not judge the numbers and it does not re-run anything. Exit 0 prints
`<LABEL>_OK` (e.g. PILOT_OK, FULL_OK); any failure prints the failed checks
and exits non-zero.

Usage:
    python tools/verify/check_quality_results.py --file results/quality_pilot.json \
        --label pilot --arms HE attention-32h --seeds 1 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCHEMA = "untransformers-quality-run/1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
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

    # --- identity -------------------------------------------------------
    check(data.get("schema") == SCHEMA,
          f"schema is {data.get('schema')!r}, expected {SCHEMA!r}")
    check(data.get("label") == args.label,
          f"label is {data.get('label')!r}, expected {args.label!r}")
    for key in ("created_utc", "protocol", "config", "corpus", "environment", "results"):
        check(key in data, f"missing top-level key {key!r}")

    protocol = data.get("protocol", {}) or {}
    check(list(protocol.get("arms", [])) == list(args.arms),
          f"protocol.arms {protocol.get('arms')} != requested {args.arms}")
    check(sorted(protocol.get("seeds", [])) == sorted(args.seeds),
          f"protocol.seeds {protocol.get('seeds')} != requested {args.seeds}")
    check(protocol.get("same_batches_same_order") is True,
          "protocol.same_batches_same_order is not True")
    check(protocol.get("paired_init_seeds") is True,
          "protocol.paired_init_seeds is not True")
    check(isinstance(protocol.get("tokens_per_step"), int) and protocol["tokens_per_step"] > 0,
          "protocol.tokens_per_step missing or not positive")
    check(isinstance(protocol.get("steps"), int) and protocol["steps"] > 0,
          "protocol.steps missing or not positive")

    # --- corpus identity ------------------------------------------------
    corpus = data.get("corpus", {}) or {}
    for key in ("corpus_id", "tokenizer_sha256", "n_tokens", "books"):
        check(key in corpus and corpus[key] not in (None, ""),
              f"corpus.{key} missing")
    # The manifest's own hash is recorded when the runner can see it; the
    # corpus_id + tokenizer hash are the anchors every run must carry.
    mh = corpus.get("manifest_sha256")
    if mh:
        check(isinstance(mh, str) and len(mh) == 64 and
              all(c in "0123456789abcdef" for c in mh.lower()),
              f"corpus.manifest_sha256 is not a sha256 hex digest: {mh!r}")
    check(isinstance(corpus.get("n_tokens"), int) and corpus.get("n_tokens", 0) > 0,
          "corpus.n_tokens not positive")
    check(corpus.get("books") == 100,
          f"corpus.books is {corpus.get('books')}, expected the pinned 100")

    # --- environment ----------------------------------------------------
    env = data.get("environment", {}) or {}
    for key in ("python", "platform", "torch", "device"):
        check(key in env, f"environment.{key} missing")
    notes.append(f"device={env.get('device')} torch={env.get('torch')} "
                 f"python={env.get('python')}")

    # --- per-arm runs ---------------------------------------------------
    results = data.get("results", {}) or {}
    steps_seen = set()
    finals = {}
    for arm in args.arms:
        if arm not in results:
            failures.append(f"results.{arm} missing")
            continue
        arm_res = results[arm]
        runs = arm_res.get("runs", [])
        got_seeds = sorted(r.get("seed") for r in runs)
        check(got_seeds == sorted(args.seeds),
              f"results.{arm}.runs seeds {got_seeds} != requested {sorted(args.seeds)}")
        for run in runs:
            seed = run.get("seed")
            steps = run.get("steps")
            curve = run.get("train_curve") or []
            evals = run.get("evals") or []
            final = run.get("final_held_out_loss")
            check(isinstance(steps, int) and steps > 0, f"{arm} seed {seed}: steps invalid")
            check(len(curve) == steps,
                  f"{arm} seed {seed}: train_curve has {len(curve)} points, expected {steps}")
            check(len(evals) > 0, f"{arm} seed {seed}: no eval points")
            check(final is not None and isinstance(final, (int, float)) and math.isfinite(final),
                  f"{arm} seed {seed}: final_held_out_loss missing or not finite")
            check(final is None or final > 0,
                  f"{arm} seed {seed}: final_held_out_loss {final} not positive")
            for e in evals:
                hl = e.get("held_out_loss")
                check(isinstance(hl, (int, float)) and math.isfinite(hl),
                      f"{arm} seed {seed}: eval at step {e.get('step')} has bad loss")
            if steps:
                steps_seen.add(steps)
            if final is not None:
                finals[(arm, seed)] = final
        notes.append(f"{arm}: seeds={got_seeds} steps={arm_res.get('steps')} "
                     f"wall={arm_res.get('wall_seconds')}")

    check(len(steps_seen) <= 1,
          f"runs disagree on step count: {sorted(steps_seen)}")

    # --- paired statistics ----------------------------------------------
    if len(args.arms) == 2:
        a, b = args.arms
        paired = data.get("paired_final")
        if not paired:
            failures.append("paired_final missing (two arms requested)")
        else:
            check(sorted(paired.get("seeds", [])) == sorted(args.seeds),
                  f"paired_final.seeds {paired.get('seeds')} != {sorted(args.seeds)}")
            check(paired.get("n") == len(args.seeds),
                  f"paired_final.n {paired.get('n')} != {len(args.seeds)}")
            diffs = paired.get("differences_arm_a_minus_arm_b", {}) or {}
            check(sorted(diffs.keys()) == sorted(str(s) for s in args.seeds),
                  f"paired_final differences keys {sorted(diffs.keys())} "
                  f"!= {sorted(str(s) for s in args.seeds)}")
            for seed in args.seeds:
                fa = (paired.get("arm_a_final_held_out_loss") or {}).get(str(seed))
                fb = (paired.get("arm_b_final_held_out_loss") or {}).get(str(seed))
                d = diffs.get(str(seed))
                if fa is None or fb is None or d is None:
                    failures.append(f"paired_final missing seed {seed}")
                    continue
                check(abs((fa - fb) - d) <= 1e-9,
                      f"paired_final difference for seed {seed} is not fa-fb")
            mean = paired.get("mean_difference")
            check(mean is not None and math.isfinite(mean),
                  "paired_final.mean_difference missing or not finite")
            sd = paired.get("std_difference")
            check(sd is not None and math.isfinite(sd) and sd >= 0,
                  "paired_final.std_difference missing or not finite")
            if finals:
                notes.append("final held-out loss: " +
                             ", ".join(f"{a}={finals.get((a,s))} {b}={finals.get((b,s))}"
                                       for s in args.seeds))

    for note in notes:
        print(f"[info] {note}")
    if failures:
        print(f"FAIL: {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"{args.label.upper()}_OK: {path} "
          f"({len(args.arms)} arms x {len(args.seeds)} seeds, "
          f"{protocol.get('steps')} steps, {protocol.get('tokens_processed_per_run')} tokens/run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
