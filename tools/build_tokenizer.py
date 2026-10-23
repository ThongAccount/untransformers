"""Train the pinned BPE tokenizer on the corpus snapshot (REPRODUCIBILITY.md §4).

Paper-pinned: HF `tokenizers`, BPE, vocab_size = 5000 **[P]**.
Ours (deviation D-04): byte-level pre-tokenizer (GPT-2 style, no prefix space),
no normalizer, `<|endoftext|>` as the sole special token, trained on the pinned
corpus in manifest order.

Deterministic: given the same corpus bytes, the same library version and the
same trainer configuration, the resulting tokenizer.json is byte-identical —
which is why we commit the JSON *and* its SHA-256 (the manifest records both).

Usage:
    python tools/build_tokenizer.py --corpus data/corpus --manifest data/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB_SIZE = 5000
SPECIAL_TOKEN = "<|endoftext|>"
SEED = 0  # recorded; the BPE trainer is deterministic given data order + version


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_id(manifest: dict) -> str:
    """Stable identity of the corpus CONTENT (independent of later manifest
    sections): sha256 over the canonical (ebook_id, sha256_clean) list."""
    payload = json.dumps(
        [[b["ebook_id"], b["sha256_clean"]] for b in manifest["books"]],
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--out", default="data/tokenizer.json")
    ap.add_argument("--tokens-out", default="data/tokens.npy")
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = ap.parse_args()

    import tokenizers  # version recorded in the manifest

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_dir = Path(args.corpus)

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.normalizer = None
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[SPECIAL_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=False,
    )

    def text_iterator():
        for book in manifest["books"]:
            yield (corpus_dir / book["file"]).read_text(encoding="utf-8")

    tok.train_from_iterator(text_iterator(), trainer=trainer)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    tokenizer_sha = sha256_file(out)

    # Encode the corpus into one frozen token array (documents separated by the
    # special token, so a window never straddles a book boundary silently).
    eos_id = tok.token_to_id(SPECIAL_TOKEN)
    pieces: list[np.ndarray] = []
    per_book: list[dict] = []
    for book in manifest["books"]:
        text = (corpus_dir / book["file"]).read_text(encoding="utf-8")
        enc = tok.encode(text)
        ids = np.asarray(enc.ids + [eos_id], dtype=np.uint16)
        pieces.append(ids)
        per_book.append({"ebook_id": book["ebook_id"], "file": book["file"], "n_tokens": int(ids.size)})
        print(f"{book['file']}: {ids.size} tokens", flush=True)

    tokens = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.uint16)
    tokens_out = Path(args.tokens_out)
    tokens_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(tokens_out, tokens)
    tokens_sha = sha256_file(tokens_out)

    manifest["tokenizer"] = {
        "file": out.name,
        "sha256": tokenizer_sha,
        "library": "tokenizers",
        "library_version": tokenizers.__version__,
        "model": "BPE",
        "vocab_size": args.vocab_size,
        "special_tokens": [SPECIAL_TOKEN],
        "normalizer": None,
        "pre_tokenizer": "ByteLevel(add_prefix_space=False, use_regex=True)",
        "decoder": "ByteLevel",
        "trainer": "BpeTrainer(min_frequency=2, initial_alphabet=ByteLevel.alphabet())",
        "trained_on": "pinned corpus snapshot (manifest books, manifest order)",
        "seed": SEED,
        "document_separator": f"{SPECIAL_TOKEN} appended after every book",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest["tokens"] = {
        "file": tokens_out.name,
        "sha256": tokens_sha,
        "dtype": "uint16",
        "count": int(tokens.size),
        "per_book": per_book,
    }
    manifest["corpus_id"] = corpus_id(manifest)
    manifest["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "tokenizers": tokenizers.__version__,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"tokenizer: {out} sha256={tokenizer_sha}")
    print(f"tokens: {tokens.size} -> {tokens_out} sha256={tokens_sha}")
    print(f"corpus_id: {manifest['corpus_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
