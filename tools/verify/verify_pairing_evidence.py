#!/usr/bin/env python3
"""Derive the pairing evidence for a quality run (Researcher's memo F1).

The G3/G4 gates used to assert `same_batches_same_order` and
`paired_init_seeds` because the runner wrote those literals — the gate could
not fail. This tool derives both claims from artefacts instead:

  * `same_batches_same_order` — the batch streams are pure functions of
    (train_seed, split, corpus_id, tokenizer_sha256, l, batch) and take no arm
    and no seed argument; the first N windows of each stream are hashed and the
    corpus/config identity is pinned;
  * `paired_init_seeds` — both arms are rebuilt with the same seed through the
    runner's own `apply_init_policy` and their shared (non-sublayer-1)
    parameters are compared bitwise.

It also cross-checks each checkpoint against the number the results file
reports for it (arm, seed, steps, final held-out loss), so the artefact and
the reported result cannot drift apart.

Usage:
    python tools/verify/verify_pairing_evidence.py \
        --results results/quality_gpu_full.json \
        --checkpoint-dir results/checkpoints \
        --out results/pairing_evidence.json \
        --merge-into results/quality_gpu_full.json

`--merge-into` writes the evidence block into the results JSON (and replaces
the two hardcoded protocol literals with the derived values). The digest of
the results file *before* the merge is recorded inside the evidence block, so
the chain of custody is visible: the merge touches no measured number.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hextractor.corpus import CorpusBundle  # noqa: E402
from hextractor.harness import TrainConfig  # noqa: E402
from hextractor.pairing import build_pairing_evidence, evidence_json, sha256_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="the quality results JSON (protocol/config/corpus source)")
    ap.add_argument("--checkpoint-dir", default="",
                    help="directory with {arm}_seed{seed}.pt checkpoints")
    ap.add_argument("--out", default="results/pairing_evidence.json")
    ap.add_argument("--merge-into", default="",
                    help="write the evidence block into this results JSON")
    ap.add_argument("--batches", type=int, default=100,
                    help="how many batch windows to hash (default 100)")
    ap.add_argument("--root", default=str(REPO_ROOT))
    args = ap.parse_args()

    root = Path(args.root)
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"FAIL: results file not found: {results_path}")
        return 1
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {results_path} is not valid JSON: {exc}")
        return 1

    protocol = results.get("protocol", {}) or {}
    config = results.get("config", {}) or {}
    corpus = results.get("corpus", {}) or {}
    arms = list(protocol.get("arms") or [])
    seeds = [int(s) for s in (protocol.get("seeds") or [])]
    if len(arms) < 2 or not seeds:
        print("FAIL: the results file does not name the arms and seeds to pair")
        return 1

    cfg = TrainConfig(
        d=int(config.get("d", 128)),
        l=int(config.get("l", 128)),
        n_layers=int(config.get("n_layers", 18)),
        ffn_width=int(config.get("ffn_width", 512)),
        vocab=int(config.get("vocab", 5000)),
        n_heads=int(config.get("n_heads", 32)),
        batch_size=int(config.get("batch_size", 64)),
    )

    print(f"[pairing] loading the corpus from {root}", flush=True)
    bundle = CorpusBundle.load(root=root, verify=True, load_tokens=True)

    # --- the corpus identity in the results file must be the one on disk ----
    if corpus.get("corpus_id") and corpus["corpus_id"] != bundle.corpus_id:
        print(f"FAIL: corpus_id in {results_path} ({corpus['corpus_id']}) is not the "
              f"corpus on disk ({bundle.corpus_id})")
        return 1
    if corpus.get("tokenizer_sha256") and corpus["tokenizer_sha256"] != bundle.tokenizer_sha256:
        print(f"FAIL: tokenizer_sha256 in {results_path} does not match the corpus "
              f"on disk")
        return 1

    ckdir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    evidence = build_pairing_evidence(
        bundle=bundle, cfg=cfg, arms=arms, seeds=seeds,
        train_seed=int(protocol.get("train_seed", 0)),
        steps=int(protocol.get("steps", 0)),
        checkpoint_dir=ckdir, n_batches=args.batches, repo_root=root)

    # --- cross-check the checkpoints against the reported numbers -----------
    reported = {}
    for arm, arm_res in (results.get("results") or {}).items():
        for run in (arm_res or {}).get("runs", []) or []:
            reported[(arm, run.get("seed"))] = {
                "steps": run.get("steps"),
                "final_held_out_loss": run.get("final_held_out_loss"),
            }
    cross = {}
    mismatches = []
    for key, entry in (evidence.get("checkpoints") or {}).items():
        if not entry.get("present"):
            cross[key] = {"present": False}
            continue
        arm, _, seed = key.rpartition("_seed")
        seed_i = int(seed)
        rep = reported.get((arm, seed_i))
        row = {"present": True, "reported_in_results": rep}
        if rep is not None:
            row["steps_match"] = (entry.get("steps") == rep.get("steps"))
            row["final_held_out_loss_match"] = (
                entry.get("final_held_out_loss") == rep.get("final_held_out_loss"))
            row["arm_match"] = (entry.get("arm") == arm)
            row["seed_match"] = (entry.get("seed") == seed_i)
            if not all((row["steps_match"], row["final_held_out_loss_match"],
                        row["arm_match"], row["seed_match"])):
                mismatches.append(key)
        cross[key] = row
    evidence["results_cross_check"] = {
        "method": ("each checkpoint's own metadata (arm, seed, steps, final held-out "
                   "loss) is compared with the number the results file reports for "
                   "the same (arm, seed)"),
        "per_checkpoint": cross,
        "mismatches": mismatches,
    }
    evidence["results_file"] = {
        "path": str(results_path),
        "sha256_before_merge": sha256_file(results_path),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(evidence_json(evidence), encoding="utf-8")
    print(f"[pairing] evidence written: {out_path}", flush=True)

    claims = evidence["claims"]
    print(f"[pairing] same_batches_same_order: verified="
          f"{claims['same_batches_same_order']['verified']}")
    print(f"[pairing] paired_init_seeds: verified="
          f"{claims['paired_init_seeds']['verified']}")
    if mismatches:
        print(f"FAIL: checkpoint metadata disagrees with the results file for: "
              f"{mismatches}")
        return 1

    if args.merge_into:
        merge_path = Path(args.merge_into)
        target = json.loads(merge_path.read_text(encoding="utf-8"))
        target["pairing_evidence"] = evidence
        target.setdefault("protocol", {})["same_batches_same_order"] = \
            claims["same_batches_same_order"]["verified"]
        target.setdefault("protocol", {})["paired_init_seeds"] = \
            claims["paired_init_seeds"]["verified"]
        target["pairing_evidence_added_utc"] = \
            datetime.now(timezone.utc).isoformat(timespec="seconds")
        merge_path.write_text(json.dumps(target, indent=2) + "\n", encoding="utf-8")
        print(f"[pairing] evidence merged into {merge_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
