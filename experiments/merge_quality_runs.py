#!/usr/bin/env python3
"""Merge the per-run quality artefacts into one quality-results file.

The sweep runs every (arm, seed) as an independent invocation of
`experiments/run_quality.py`, so that a session that dies loses at most the
run in flight. Each invocation writes its own JSON; this script merges them
into the single file the gates (`tools/verify/check_quality_results.py`) read.

The merge is exact, not approximate: the harness hands out a FRESH data
stream per seed (`corpus_stream_factory` returns a factory; `train_one_seed`
calls it again for every seed), so every seed replays the same batches from
the start and every arm sees the same batches in the same order. Running the
six (arm, seed) pairs separately therefore produces exactly the same numbers
as one combined invocation — the split only changes how the work is packaged.

Usage:
    python experiments/merge_quality_runs.py --runs results/runs \
        --out results/quality_gpu_full.json --label full \
        --checkpoint-dir results/checkpoints --commit <sha>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hextractor.corpus import CorpusBundle  # noqa: E402
from hextractor.harness import TrainConfig, dispersion  # noqa: E402
from hextractor.pairing import build_pairing_evidence  # noqa: E402

from run_quality import paired_stats  # noqa: E402  (same directory)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory with the per-run JSONs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="full")
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--commit", default="", help="the commit that was measured")
    ap.add_argument("--pairing-batches", type=int, default=100)
    args = ap.parse_args()

    run_files = sorted(Path(args.runs).glob("*.json"))
    if not run_files:
        print(f"FAIL: no per-run JSONs in {args.runs}")
        return 1

    parts = []
    for path in run_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        arms = list(data.get("protocol", {}).get("arms") or [])
        if len(arms) != 1:
            print(f"FAIL: {path} is not a single-arm artefact ({arms})")
            return 1
        arm = arms[0]
        runs = (data.get("results") or {}).get(arm, {}).get("runs") or []
        if len(runs) != 1:
            print(f"FAIL: {path} does not carry exactly one run ({len(runs)})")
            return 1
        parts.append((path, data, arm, runs[0]))
    print(f"[merge] {len(parts)} per-run artefacts")

    # --- consistency of the parts -------------------------------------------
    first = parts[0][1]
    for path, data, _arm, _run in parts[1:]:
        for key in ("config", "corpus"):
            if data.get(key) != first.get(key):
                print(f"FAIL: {path} disagrees with {parts[0][0]} on {key}")
                return 1
        for key in ("steps", "train_seed", "eval_every", "eval_batches",
                    "tokens_per_step"):
            if (data.get("protocol") or {}).get(key) != (first.get("protocol") or {}).get(key):
                print(f"FAIL: {path} disagrees with {parts[0][0]} on protocol.{key}")
                return 1

    arms: list = []
    for _p, _d, arm, _r in parts:
        if arm not in arms:
            arms.append(arm)
    seeds = sorted({int(run["seed"]) for _p, _d, _a, run in parts})
    print(f"[merge] arms={arms} seeds={seeds}")

    results: dict = {}
    for arm in arms:
        runs = sorted((run for _p, _d, a, run in parts if a == arm),
                      key=lambda r: int(r["seed"]))
        proto = next(d["protocol"] for _p, d, a, _r in parts if a == arm)
        results[arm] = {
            "arm": arm,
            "seeds": [int(r["seed"]) for r in runs],
            "train_seed": proto.get("train_seed"),
            "steps": proto.get("steps"),
            "runs": runs,
            "dispersion": dispersion(runs),
            "source_artefacts": [str(p) for p, _d, a, _r in parts if a == arm],
        }

    protocol = dict(first["protocol"])
    protocol["arms"] = arms
    protocol["seeds"] = seeds

    out = {
        "schema": first.get("schema", "untransformers-quality-run/1"),
        "label": args.label,
        "created_utc": first.get("created_utc"),
        "updated_utc": first.get("updated_utc"),
        "commit": args.commit or first.get("commit"),
        "merged_from": [str(p) for p, _d, _a, _r in parts],
        "merge_note": ("each (arm, seed) was an independent invocation of the same "
                       "runner; the harness hands out a fresh data stream per seed, "
                       "so the split does not change the data order or the numbers"),
        "protocol": protocol,
        "config": first["config"],
        "corpus": first["corpus"],
        "environment": first["environment"],
        "results": results,
    }

    if len(arms) == 2:
        out["paired_final"] = paired_stats(results[arms[0]], results[arms[1]])

    # --- pairing evidence, rebuilt over the full arm/seed set --------------
    cfg_fields = first["config"]
    cfg = TrainConfig(d=cfg_fields["d"], l=cfg_fields["l"],
                      n_layers=cfg_fields["n_layers"],
                      ffn_width=cfg_fields["ffn_width"], vocab=cfg_fields["vocab"],
                      batch_size=cfg_fields["batch_size"])
    bundle = CorpusBundle.load(root=REPO_ROOT, verify=True, load_tokens=True)
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    evidence = build_pairing_evidence(
        bundle, cfg, arms, seeds, train_seed=protocol.get("train_seed", 0),
        steps=protocol.get("steps", 0), checkpoint_dir=ckpt_dir,
        n_batches=args.pairing_batches, repo_root=REPO_ROOT)
    out["pairing_evidence"] = evidence
    claims = evidence["claims"]
    protocol["same_batches_same_order"] = bool(claims["same_batches_same_order"]["verified"])
    protocol["paired_init_seeds"] = bool(claims["paired_init_seeds"]["verified"])
    out["protocol"] = protocol

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
