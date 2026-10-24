"""Re-score the paired quality checkpoints (evaluation only — no retraining).

Two measurements per checkpoint:

1. **The run's own eval set** — the first `eval_batches` batches of the
   held-out stream, i.e. the same 512 windows every eval point of the sweep
   was measured on (`harness.py:held_out_loss` draws a fresh stream per eval,
   so the set is fixed). The per-window losses are kept, so the eval-set
   bootstrap of protocol §3.3 can be computed later without re-running a
   model.

2. **The book-aligned set** — the three books that are *fully* held out
   (ebooks 3536, 3795, 3796; 406,723 tokens), swept **exhaustively with
   stride 1** (every window start inside the book), so this metric carries no
   window-sampling noise at all. This is the pre-registered sensitivity
   analysis (protocol §2.3): it is reported *alongside* the primary endpoint
   and may not replace it. The protocol's own phrasing ("~3,177 windows per
   book") is the count of NON-overlapping windows over the three books
   together; the exhaustive stride-1 sweep has one window per token start.

The recomputed eval-set mean must reproduce the final held-out loss the run
reported; the difference is recorded as a cross-check that the checkpoint and
the results file describe the same model.

Usage:
    python experiments/eval_checkpoints.py --checkpoint-dir results/checkpoints \
        --arms HE attention-32h --seeds 1 2 3 --out results/eval_windows.json \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hextractor.corpus import CorpusBundle  # noqa: E402
from hextractor.harness import TrainConfig  # noqa: E402
from hextractor.model import ExtractorTransformer  # noqa: E402

SCHEMA = "untransformers-checkpoint-eval/1"
UNSEEN_BOOKS = (3536, 3795, 3796)


def book_spans(bundle: CorpusBundle) -> dict:
    """Token span of every book in the frozen token array."""
    spans = {}
    start = 0
    for entry in bundle.manifest["tokens"]["per_book"]:
        n = int(entry["n_tokens"])
        spans[int(entry["ebook_id"])] = (start, start + n)
        start += n
    return spans


@torch.no_grad()
def per_window_losses(model, x: np.ndarray, y: np.ndarray, device: torch.device,
                      batch: int = 64) -> np.ndarray:
    """Mean next-token CE per window (each window contributes equally)."""
    model.eval()
    out = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], batch):
        xb = torch.from_numpy(x[start:start + batch]).to(device)
        yb = torch.from_numpy(y[start:start + batch]).to(device)
        logits = model(xb)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1),
                             reduction="none").reshape(yb.shape[0], -1)
        out[start:start + batch] = ce.mean(dim=1).double().cpu().numpy()
    return out


def build_model_from_checkpoint(blob: dict, device: torch.device):
    cfg = blob["config"]
    model = ExtractorTransformer(
        arm=blob["arm"], d=cfg["d"], l=cfg["l"], n_layers=cfg["n_layers"],
        ffn_width=cfg["ffn_width"], vocab=cfg["vocab"], n_heads=cfg["n_heads"],
        dropout=cfg["dropout"])
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval(), TrainConfig(**cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="results/checkpoints")
    ap.add_argument("--arms", nargs="+", default=["HE", "attention-32h"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--out", default="results/eval_windows.json")
    ap.add_argument("--npz", default="", help="optional .npz for the book-aligned per-window losses")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--books", nargs="+", type=int, default=list(UNSEEN_BOOKS))
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" or
                                     (args.device == "auto" and torch.cuda.is_available()))
                          else "cpu")
    bundle = CorpusBundle.load(root=REPO_ROOT, verify=True, load_tokens=True)
    spans = book_spans(bundle)
    tokens = bundle.tokens

    # --- the eval windows the sweep used -----------------------------------
    # (the first `eval_batches` batches of the held-out stream: the same
    #  windows every eval point was measured on). The window shape follows the
    # checkpoint's own config, so it is rebuilt per checkpoint below.

    out = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "device": device.type,
        "torch": torch.__version__,
        "train_seed": args.train_seed,
        "books": list(args.books),
        "book_aligned": {
            "definition": ("every stride-1 window of length l+1 inside each fully "
                           "unseen book; no sampling, so no window-sampling noise"),
            "books": {},
        },
        "eval_set": {},
        "checkpoints": {},
    }
    book_arrays = {}

    for arm in args.arms:
        for seed in args.seeds:
            path = Path(args.checkpoint_dir) / f"{arm}_seed{seed}.pt"
            if not path.exists():
                print(f"[skip] {path} is missing", flush=True)
                continue
            t0 = time.time()
            blob = torch.load(path, map_location="cpu", weights_only=False)
            model, cfg = build_model_from_checkpoint(blob, device)
            l, batch = cfg.l, cfg.batch_size

            # (1) the run's own eval set
            it = bundle.stream(l=l, batch=batch, train_seed=args.train_seed, held_out=True)
            xs, ys = [], []
            for _ in range(args.eval_batches):
                x, y = next(it)
                xs.append(x)
                ys.append(y)
            eval_x = np.concatenate(xs, axis=0)
            eval_y = np.concatenate(ys, axis=0)
            eval_losses = per_window_losses(model, eval_x, eval_y, device, args.batch)
            reported = blob.get("final_held_out_loss")
            recomputed = float(eval_losses.mean())
            out["eval_set"][f"{arm}_seed{seed}"] = {
                "n_windows": int(eval_losses.size),
                "target_tokens": int(eval_losses.size * l),
                "mean_loss": recomputed,
                "reported_final_held_out_loss": reported,
                "recomputed_minus_reported": (None if reported is None
                                              else recomputed - float(reported)),
                "per_window_loss": [float(v) for v in eval_losses],
            }

            # (2) the book-aligned set
            per_book = {}
            book_losses = []
            for ebook_id in args.books:
                start, end = spans[ebook_id]
                region = tokens[start:end]
                n = region.size - l          # stride-1 windows of l+1 tokens
                if n <= 0:
                    continue
                wins = np.stack([region[s:s + l + 1] for s in range(n)]).astype(np.int64)
                bx, by = wins[:, :-1], wins[:, 1:]
                losses = per_window_losses(model, bx, by, device, args.batch)
                per_book[str(ebook_id)] = {
                    "n_tokens": int(region.size),
                    "n_windows": int(losses.size),
                    "mean_loss": float(losses.mean()),
                }
                book_losses.append(losses)
            all_losses = np.concatenate(book_losses) if book_losses else np.zeros(0)
            out["book_aligned"]["books"][f"{arm}_seed{seed}"] = per_book
            book_arrays[f"{arm}_seed{seed}"] = all_losses.astype(np.float32)

            out["checkpoints"][f"{arm}_seed{seed}"] = {
                "arm": blob.get("arm"), "seed": blob.get("seed"),
                "steps": blob.get("steps"),
                "final_held_out_loss": blob.get("final_held_out_loss"),
                "checkpoint": str(path),
                "wall_seconds": time.time() - t0,
            }
            print(f"[eval] {arm} seed {seed}: eval-set mean={recomputed:.6f} "
                  f"(reported {reported}), book-aligned mean={float(all_losses.mean()):.6f} "
                  f"over {all_losses.size} windows in {time.time() - t0:.1f}s", flush=True)
            del model, blob

    out["book_aligned"]["overall"] = {
        key: {"n_windows": int(arr.size), "mean_loss": float(arr.mean())}
        for key, arr in book_arrays.items()
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"written: {out_path}", flush=True)

    if args.npz and book_arrays:
        np.savez_compressed(args.npz, **book_arrays)
        print(f"written: {args.npz}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
