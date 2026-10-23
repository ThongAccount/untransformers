#!/usr/bin/env python3
"""Builds the checkpoint manifest for a quality sweep.

The paired checkpoints themselves are large (one .pt per arm per seed) and are
reproducible outputs: same commit, same pinned corpus, same seeds, same
batches => same bytes. What belongs in the repository is the manifest — which
checkpoint, which arm, which seed, how many steps, what held-out loss, how
many bytes, what SHA-256 — so that the systems numbers can be tied to a stated
quality level without bloating the git history with a hundred megabytes of
float32 weights.

Usage:
    python tools/make_checkpoint_manifest.py --dir results/checkpoints \
        --out results/checkpoints/MANIFEST.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="full")
    args = ap.parse_args()

    ckdir = Path(args.dir)
    out_path = Path(args.out)
    entries = []
    for p in sorted(ckdir.glob("*.pt")):
        try:
            import torch
            blob = torch.load(p, map_location="cpu", weights_only=False)
            arm = blob.get("arm")
            seed = blob.get("seed")
            steps = blob.get("steps")
            final = blob.get("final_held_out_loss")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {p.name}: could not read the payload: {exc}")
            arm = seed = steps = final = None
        entries.append({
            "file": p.name,
            "arm": arm,
            "seed": seed,
            "steps": steps,
            "final_held_out_loss": final,
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
        print(f"[info] {p.name}: arm={arm} seed={seed} steps={steps} "
              f"final_held_out_loss={final} sha256={entries[-1]['sha256'][:16]}…")

    manifest = {
        "schema": "untransformers-checkpoint-manifest/1",
        "label": args.label,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "The .pt files themselves are not committed: they are deterministic "
                "outputs of the pinned commit, corpus and seeds. The sha256 of each "
                "file lets anyone who regenerates them prove the match.",
        "checkpoints": entries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest written: {out_path} ({len(entries)} checkpoints)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
