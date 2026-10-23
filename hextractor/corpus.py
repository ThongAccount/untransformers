"""Corpus-backed deterministic data pipeline (T14).

Replaces the synthetic stream (deviation D-17) with the pinned Gutenberg
snapshot: manifest -> verified book files -> tokenizer -> one frozen token
array -> deterministic batch stream.

Determinism contract (REPRODUCIBILITY.md §5): batches come from a
SHA-256-seeded `numpy.random.Generator` over the frozen token array, so both
arms see identical batches in identical order by construction — not by
replaying a log. The seed material mixes the data-pipeline seed, the split,
the corpus identity and the tokenizer identity, so changing any of them
changes the stream (and T14 asserts that).

Held-out split (D-02): the last 5% of the token stream is never trained on and
never used for selection; it is reported only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
HELD_OUT_FRACTION = 0.05  # last 5% of the token stream (REPRODUCIBILITY.md §5)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_corpus_id(manifest: dict) -> str:
    payload = json.dumps(
        [[b["ebook_id"], b["sha256_clean"]] for b in manifest["books"]],
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


@dataclass
class CorpusBundle:
    """Manifest + verified files + tokenizer + frozen token array."""

    root: Path
    manifest: dict
    tokens: np.ndarray

    @property
    def manifest_path(self) -> Path:
        return self.root / "data" / "manifest.json"

    @property
    def corpus_dir(self) -> Path:
        return self.root / "data" / "corpus"

    @property
    def tokenizer_path(self) -> Path:
        return self.root / "data" / self.manifest["tokenizer"]["file"]

    @property
    def corpus_id(self) -> str:
        return self.manifest.get("corpus_id") or canonical_corpus_id(self.manifest)

    @property
    def tokenizer_sha256(self) -> str:
        return self.manifest["tokenizer"]["sha256"]

    @property
    def n_tokens(self) -> int:
        return int(self.tokens.size)

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, root: Path | str = DEFAULT_ROOT, verify: bool = True,
             load_tokens: bool = True) -> "CorpusBundle":
        root = Path(root)
        manifest_path = root / "data" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if verify:
            cls.verify_files(root, manifest)

        tokens = np.zeros(0, dtype=np.uint16)
        if load_tokens:
            tok_meta = manifest["tokens"]
            tokens_path = root / "data" / tok_meta["file"]
            if not tokens_path.exists():
                raise FileNotFoundError(
                    f"{tokens_path} missing — run tools/build_tokenizer.py first")
            if verify:
                got = sha256_file(tokens_path)
                if got != tok_meta["sha256"]:
                    raise RuntimeError(
                        f"token array sha256 mismatch: manifest={tok_meta['sha256']} file={got}")
            tokens = np.load(tokens_path)
        return cls(root=root, manifest=manifest, tokens=tokens)

    @staticmethod
    def verify_files(root: Path, manifest: dict) -> None:
        """Every committed corpus file must match its manifest checksum."""
        corpus_dir = root / "data" / "corpus"
        problems = []
        for book in manifest["books"]:
            path = corpus_dir / book["file"]
            if not path.exists():
                problems.append(f"missing corpus file: {path}")
                continue
            got = sha256_file(path)
            if got != book["sha256_clean"]:
                problems.append(
                    f"sha256 mismatch for {path}: manifest={book['sha256_clean']} file={got}")
        if problems:
            raise RuntimeError("corpus verification failed:\n" + "\n".join(problems))

    # ----------------------------------------------------------------- splits
    def split_bounds(self, held_out_fraction: float = HELD_OUT_FRACTION) -> Tuple[int, int]:
        """(train_end, total) in token positions. Train = first (1-f) of the
        stream, held-out = last f — never trained, never used for selection."""
        total = self.n_tokens
        cut = int(total * (1.0 - held_out_fraction))
        return cut, total

    def region(self, held_out: bool = False,
               held_out_fraction: float = HELD_OUT_FRACTION) -> np.ndarray:
        cut, total = self.split_bounds(held_out_fraction)
        if held_out:
            return self.tokens[cut:total]
        return self.tokens[:cut]

    # ----------------------------------------------------------------- stream
    def seed_material(self, train_seed: int, held_out: bool) -> bytes:
        split = "heldout" if held_out else "train"
        return f"{train_seed}|{split}|{self.corpus_id}|{self.tokenizer_sha256}".encode("utf-8")

    def stream(self, l: int = 128, batch: int = 64, train_seed: int = 0,
               held_out: bool = False,
               held_out_fraction: float = HELD_OUT_FRACTION) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Infinite deterministic stream of (x, y) token windows.

        y[i] = x[i+1] inside each window; windows are contiguous l+1-token
        slices of the frozen token array, sampled with replacement by a
        SHA-256-seeded numpy Generator.
        """
        region = self.region(held_out=held_out, held_out_fraction=held_out_fraction)
        if region.size < l + 1:
            raise ValueError(f"region too small for l={l}: {region.size} tokens")
        digest = hashlib.sha256(self.seed_material(train_seed, held_out)).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        n_start = region.size - l - 1
        while True:
            starts = rng.integers(0, n_start, size=batch, endpoint=False)
            windows = np.stack([region[s:s + l + 1] for s in starts]).astype(np.int64)
            yield windows[:, :-1], windows[:, 1:]


def stream_sha256(bundle: CorpusBundle, n_batches: int = 100, l: int = 128,
                   batch: int = 64, train_seed: int = 0,
                   held_out: bool = False) -> str:
    """SHA of the first n_batches of token tensors — the T14 fingerprint."""
    h = hashlib.sha256()
    it = bundle.stream(l=l, batch=batch, train_seed=train_seed, held_out=held_out)
    for _ in range(n_batches):
        x, y = next(it)
        h.update(np.ascontiguousarray(x, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(y, dtype=np.int64).tobytes())
    return h.hexdigest()
