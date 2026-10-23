"""Milestone 2 quality replication on the pinned corpus.

Runs the ratified protocol (REPRODUCIBILITY.md §5): both arms, the same
batches in the same order, one or more seeds, held-out loss reported at fixed
intervals, dispersion across seeds. This script produces DATA — curves, per-seed
numbers, dispersion, paired per-seed differences — and no verdicts; the
interpretation belongs to the protocol review (Researcher), not to the runner.

Usage:
    python experiments/run_quality.py --arms HE attention-32h --seeds 1 2 \
        --steps 200 --eval-every 50 --label pilot --out results/quality_pilot.json

Everything needed to re-run the numbers is written into the output JSON:
config, protocol, corpus identity, environment manifest, per-seed curves.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hextractor.corpus import CorpusBundle  # noqa: E402
from hextractor.harness import TrainConfig, run_multi_seed_corpus  # noqa: E402


def env_manifest(device: torch.device) -> dict:
    man = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": device.type,
    }
    if device.type == "cuda":
        man["cuda"] = torch.version.cuda
        man["cudnn"] = torch.backends.cudnn.version()
        man["gpu"] = torch.cuda.get_device_name(device)
        man["gpu_capability"] = list(torch.cuda.get_device_capability(device))
        man["gpu_memory_total_bytes"] = torch.cuda.get_device_properties(device).total_memory
    return man


def paired_stats(arm_a: dict, arm_b: dict) -> dict:
    """Per-seed paired differences at the final eval point (data, not verdicts).

    `arm_a` is the treatment arm (HE), `arm_b` the reference (attention).
    Positive difference = HE's held-out loss is higher (worse) than attention's.
    """
    fa = {r["seed"]: r["final_held_out_loss"] for r in arm_a["runs"]}
    fb = {r["seed"]: r["final_held_out_loss"] for r in arm_b["runs"]}
    seeds = sorted(set(fa) & set(fb))
    diffs = [fa[s] - fb[s] for s in seeds]
    n = len(diffs)
    mean = sum(diffs) / n if n else None
    var = sum((x - mean) ** 2 for x in diffs) / n if n and n > 1 else 0.0
    sd = math.sqrt(var)
    return {
        "seeds": seeds,
        "arm_a_final_held_out_loss": {str(s): fa[s] for s in seeds},
        "arm_b_final_held_out_loss": {str(s): fb[s] for s in seeds},
        "differences_arm_a_minus_arm_b": {str(s): d for s, d in zip(seeds, diffs)},
        "mean_difference": mean,
        "std_difference": sd,
        "n": n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["HE", "attention-32h"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--train-seed", type=int, default=0, help="data-pipeline seed (shared by all arms)")
    ap.add_argument("--label", default="pilot", help="pilot | full | <custom>: labels the run's budget")
    ap.add_argument("--out", default="results/quality.json")
    ap.add_argument("--checkpoint-dir", default="results/checkpoints")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--save-checkpoints", action="store_true")
    # shape overrides (paper defaults unless stated)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--l", type=int, default=128)
    ap.add_argument("--layers", type=int, default=18)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=5000)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = leave default)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    device = torch.device("cuda" if (args.device == "cuda" or
                                     (args.device == "auto" and torch.cuda.is_available()))
                          else "cpu")

    cfg = TrainConfig(d=args.d, l=args.l, n_layers=args.layers, ffn_width=args.ffn,
                      vocab=args.vocab, batch_size=args.batch,
                      eval_every=args.eval_every, eval_batches=args.eval_batches)

    bundle = CorpusBundle.load(root=REPO_ROOT, verify=True, load_tokens=True)
    cut, total = bundle.split_bounds()
    tokens_per_step = cfg.batch_size * cfg.l

    out = {
        "schema": "untransformers-quality-run/1",
        "label": args.label,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "arms": args.arms,
            "seeds": args.seeds,
            "train_seed": args.train_seed,
            "steps": args.steps,
            "eval_every": args.eval_every,
            "eval_batches": args.eval_batches,
            "tokens_per_step": tokens_per_step,
            "tokens_processed_per_run": tokens_per_step * args.steps,
            "epochs_of_our_corpus": (tokens_per_step * args.steps) / max(cut, 1),
            "corpus_tokens_total": total,
            "corpus_tokens_train_region": cut,
            "corpus_tokens_heldout_region": total - cut,
            "same_batches_same_order": True,
            "paired_init_seeds": True,
        },
        "config": {
            "d": cfg.d, "l": cfg.l, "n_layers": cfg.n_layers, "ffn_width": cfg.ffn_width,
            "vocab": cfg.vocab, "batch_size": cfg.batch_size, "lr": cfg.lr,
            "betas": list(cfg.betas), "weight_decay": cfg.weight_decay,
            "dropout": cfg.dropout, "eval_every": cfg.eval_every,
            "eval_batches": cfg.eval_batches, "clip_grad": cfg.clip_grad,
            "weight_decay_no_norm": cfg.weight_decay_no_norm,
        },
        "corpus": {
            "corpus_id": bundle.corpus_id,
            "tokenizer_sha256": bundle.tokenizer_sha256,
            "manifest_sha256": bundle.manifest.get("manifest_sha256"),
            "n_tokens": bundle.n_tokens,
            "books": len(bundle.manifest["books"]),
        },
        "environment": env_manifest(device),
        "results": {},
    }

    ckpt_dir = Path(args.checkpoint_dir)
    if args.save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def flush_results() -> None:
        """Write partial results after every arm so a session killed mid-sweep
        still leaves the completed arms on disk."""
        out["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    for arm in args.arms:
        t0 = time.time()
        print(f"[{arm}] starting, seeds={args.seeds}, steps={args.steps}", flush=True)
        res = run_multi_seed_corpus(arm, cfg, args.seeds, args.steps, bundle,
                                     train_seed=args.train_seed,
                                     checkpoint_dir=str(ckpt_dir) if args.save_checkpoints else None)
        res["wall_seconds"] = time.time() - t0
        if args.save_checkpoints:
            res["checkpoints"] = [str(ckpt_dir / f"{arm}_seed{s}.pt") for s in args.seeds]
        out["results"][arm] = res
        print(f"[{arm}] done in {res['wall_seconds']:.1f}s; "
              f"final held-out loss per seed: "
              f"{ {r['seed']: r['final_held_out_loss'] for r in res['runs']} }", flush=True)
        flush_results()

    if len(args.arms) == 2:
        out["paired_final"] = paired_stats(out["results"][args.arms[0]],
                                           out["results"][args.arms[1]])
        print(f"[paired] {json.dumps(out['paired_final'], indent=2)}", flush=True)

    flush_results()
    print(f"results written: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
