#!/usr/bin/env python3
"""Gate check for a Milestone-2 quality run (G3 pilot / G4 GPU sweep).

Verifies STRUCTURE and INTERNAL CONSISTENCY of a quality-results JSON — it
does not judge the numbers and it does not re-run anything. Exit 0 prints
`<LABEL>_OK` (e.g. PILOT_OK, FULL_OK); any failure prints the failed checks
and exits non-zero.

Since protocol revision 2 (`docs/milestone-3/PROTOCOL.md` §5, F1) the gate no
longer trusts the runner's own `same_batches_same_order` / `paired_init_seeds`
literals: the pairing claims must be certified by a `pairing_evidence` block
(produced by `tools/verify/verify_pairing_evidence.py`), either inside the
results file or supplied with `--pairing-evidence`. A literal that
contradicts the evidence fails the gate.

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
PAIRING_SCHEMA = "untransformers-pairing-evidence/1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--pairing-evidence", default="",
                    help="external pairing-evidence JSON (default: the block inside "
                         "the results file)")
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

    # --- pairing evidence (F1) -------------------------------------------
    # The gate must certify the pairing claims from evidence, not from the
    # runner's own literals (Researcher's memo F1).
    evidence = data.get("pairing_evidence")
    if args.pairing_evidence:
        ev_path = Path(args.pairing_evidence)
        if not ev_path.exists():
            failures.append(f"pairing evidence file not found: {ev_path}")
        else:
            try:
                evidence = json.loads(ev_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"pairing evidence {ev_path} is not valid JSON: {exc}")
                evidence = None
    if not isinstance(evidence, dict) or not evidence:
        failures.append("pairing evidence missing — the gate cannot certify "
                        "same_batches_same_order / paired_init_seeds without it "
                        "(F1: derive the claims, do not assert them)")
        claims = {}
    else:
        check(evidence.get("schema") == PAIRING_SCHEMA,
              f"pairing evidence schema is {evidence.get('schema')!r}, expected "
              f"{PAIRING_SCHEMA!r}")
        verifier = evidence.get("verifier") or {}
        check(bool(verifier.get("tool")),
              "pairing evidence carries no verifier stamp (tool)")
        check(bool(verifier.get("git_commit")),
              "pairing evidence carries no verifier git commit")
        ev_corpus = evidence.get("corpus") or {}
        check(ev_corpus.get("corpus_id") == corpus.get("corpus_id"),
              f"pairing evidence corpus_id {ev_corpus.get('corpus_id')} != the run's "
              f"{corpus.get('corpus_id')}")
        check(ev_corpus.get("tokenizer_sha256") == corpus.get("tokenizer_sha256"),
              "pairing evidence tokenizer_sha256 != the run's")
        ev_protocol = evidence.get("protocol") or {}
        if ev_protocol.get("train_seed") is not None and \
                protocol.get("train_seed") is not None:
            check(ev_protocol["train_seed"] == protocol["train_seed"],
                  f"pairing evidence train_seed {ev_protocol['train_seed']} != the "
                  f"run's {protocol['train_seed']}")
        for key, run_key in (("l", "l"), ("batch_size", "batch_size")):
            ev_value = ev_protocol.get(key)
            run_value = (data.get("config") or {}).get(run_key)
            if ev_value is not None and run_value is not None:
                check(ev_value == run_value,
                      f"pairing evidence {key} {ev_value} != the run's {run_value}")
        check(set(args.arms).issubset(set(ev_protocol.get("arms") or [])),
              f"pairing evidence does not cover the arms {args.arms}")
        check(set(args.seeds).issubset({int(s) for s in (ev_protocol.get("seeds") or [])}),
              f"pairing evidence does not cover the seeds {args.seeds}")
        claims = evidence.get("claims") or {}
        batches = claims.get("same_batches_same_order") or {}
        check(batches.get("verified") is True,
              "pairing evidence does not verify same_batches_same_order")
        for digest_key in ("train_stream_sha256", "held_out_stream_sha256"):
            digest = batches.get(digest_key)
            check(isinstance(digest, str) and len(digest) == 64 and
                  all(c in "0123456789abcdef" for c in digest.lower()),
                  f"pairing evidence {digest_key} is not a sha256 hex digest: {digest!r}")
        check(isinstance(batches.get("n_batches_hashed"), int) and
              batches.get("n_batches_hashed", 0) > 0,
              "pairing evidence hashed no batch windows")
        init = claims.get("paired_init_seeds") or {}
        check(init.get("verified") is True,
              "pairing evidence does not verify paired_init_seeds")
        per_seed = init.get("per_seed") or {}
        for seed in args.seeds:
            entry = per_seed.get(str(seed))
            if not entry:
                failures.append(f"pairing evidence has no shared-parameter comparison "
                                f"for seed {seed}")
                continue
            check(entry.get("bitwise_equal") is True,
                  f"pairing evidence: the shared parameters of seed {seed} are not "
                  f"bitwise equal across the arms")
            # The count may sit on the seed (aggregated) or on each per-arm
            # comparison; both shapes are accepted.
            drawn = entry.get("n_generator_drawn")
            if not isinstance(drawn, int) or drawn <= 0:
                per_arm = entry.get("per_arm") or {}
                drawn = max([int((e or {}).get("n_generator_drawn") or 0)
                             for e in per_arm.values()] or [0])
            check(drawn > 0,
                  f"pairing evidence: seed {seed} compared no generator-drawn "
                  f"parameter")
        # the runner's literals (if present) must not contradict the evidence
        for literal, claim_name in (("same_batches_same_order", "same_batches_same_order"),
                                    ("paired_init_seeds", "paired_init_seeds")):
            if literal in protocol:
                claim = (claims.get(claim_name) or {}).get("verified")
                check(protocol[literal] is claim,
                      f"protocol.{literal}={protocol[literal]!r} contradicts the "
                      f"pairing evidence (verified={claim!r})")

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
