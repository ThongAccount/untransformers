"""Print the SHA-256 fingerprint of the first N batches of the corpus stream.

Run as a script (used by T14 to compare fingerprints across *processes*):

    python tests/_stream_fingerprint.py --seed 0 --batches 100 [--held-out]

Prints one line: "<sha256> <n_batches> <batch> <l> <seed> <split>".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hextractor.corpus import CorpusBundle, stream_sha256  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--l", type=int, default=128)
    ap.add_argument("--held-out", action="store_true")
    args = ap.parse_args()

    bundle = CorpusBundle.load(root=REPO_ROOT, verify=True, load_tokens=True)
    digest = stream_sha256(bundle, n_batches=args.batches, l=args.l, batch=args.batch,
                            train_seed=args.seed, held_out=args.held_out)
    split = "heldout" if args.held_out else "train"
    print(f"{digest} {args.batches} {args.batch} {args.l} {args.seed} {split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
