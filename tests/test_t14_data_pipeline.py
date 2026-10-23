"""Suite T14 — data pipeline determinism (corpus manifest + token stream).

TEST_SPEC.md §T14:
  * same seed => byte-identical batch stream across two processes
  * tokenizer round-trip stable
  * manifest checksums match the committed corpus files
  * pass: identical SHA of the first 100 batches' token tensors

Negative controls (the suite must be able to redden):
  * mutating a corpus file must break the checksum verification
  * a different data-pipeline seed must produce a different stream fingerprint
  * mutating the seed material (split/corpus/tokenizer) must change the stream
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hextractor.corpus import (  # noqa: E402
    CorpusBundle, canonical_corpus_id, sha256_file, stream_sha256,
)

MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
FINGERPRINT = REPO_ROOT / "tests" / "_stream_fingerprint.py"

pytestmark = pytest.mark.skipif(
    not MANIFEST_PATH.exists(), reason="corpus manifest not present (run tools/fetch_corpus.py)")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bundle() -> CorpusBundle:
    return CorpusBundle.load(root=REPO_ROOT, verify=True, load_tokens=True)


# ------------------------------------------------------------------ manifest
def test_manifest_pins_the_selection_rule(manifest):
    assert manifest["source"] == "Project Gutenberg"
    assert manifest["shelf"]["id"] == 20
    assert manifest["shelf"]["books_in_shelf"] >= 100
    assert len(manifest["shelf"]["listing_sha256"]) == 64
    assert "ebook ID" in manifest["selection_rule"]
    ids = [b["ebook_id"] for b in manifest["books"]]
    assert len(ids) == 100
    assert len(set(ids)) == 100, "duplicate ebook id in the manifest"
    assert ids == sorted(ids), "selection must be ascending by numeric ebook ID"
    assert manifest["corpus_id"] == canonical_corpus_id(manifest)


def test_shelf_snapshot_matches_the_manifest(manifest):
    """The shelf listing snapshot proves which pool the selection came from."""
    snap_path = REPO_ROOT / "data" / "shelf20_listing.json"
    assert snap_path.exists(), "missing data/shelf20_listing.json"
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    assert snap["shelf_id"] == 20
    assert snap["listing_sha256"] == manifest["shelf"]["listing_sha256"]
    assert snap["books_in_shelf"] == manifest["shelf"]["books_in_shelf"]
    pool = snap["listing_order_ebook_ids"]
    assert len(pool) == snap["books_in_shelf"] == len(set(pool))
    selected = [b["ebook_id"] for b in manifest["books"]]
    assert snap["selected_ebook_ids"] == selected
    # the selection is exactly the 100 smallest ids of the snapshot pool
    assert selected == sorted(pool)[:100]


def test_manifest_checksums_match_committed_corpus_files(manifest):
    """Every committed corpus file must hash to its manifest checksum."""
    for book in manifest["books"]:
        path = CORPUS_DIR / book["file"]
        assert path.exists(), f"missing corpus file {path}"
        assert sha256_file(path) == book["sha256_clean"], f"sha256 mismatch: {path}"
        assert path.stat().st_size == book["bytes_clean"]


def test_tokenizer_and_token_array_match_manifest(manifest):
    tok_path = REPO_ROOT / "data" / manifest["tokenizer"]["file"]
    assert tok_path.exists()
    assert sha256_file(tok_path) == manifest["tokenizer"]["sha256"]
    assert manifest["tokenizer"]["vocab_size"] == 5000
    assert manifest["tokenizer"]["special_tokens"] == ["<|endoftext|>"]
    tokens_path = REPO_ROOT / "data" / manifest["tokens"]["file"]
    assert tokens_path.exists()
    assert sha256_file(tokens_path) == manifest["tokens"]["sha256"]
    per_book_total = sum(b["n_tokens"] for b in manifest["tokens"]["per_book"])
    assert manifest["tokens"]["count"] == per_book_total
    assert manifest["tokens"]["count"] > 0


# ------------------------------------------------------------------ tokenizer
def test_tokenizer_roundtrip_is_stable(manifest, bundle):
    """Encode -> decode -> encode must be a fixed point (byte-level BPE)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(REPO_ROOT / "data" / manifest["tokenizer"]["file"]))
    sample = ("It was the best of times, it was the worst of times.\n"
              "Chapter 1.  The quick brown fox jumps over 13 lazy dogs.\n")
    ids = tok.encode(sample).ids
    decoded = tok.decode(ids)
    ids2 = tok.encode(decoded).ids
    assert ids2 == ids, "tokenizer round-trip is not stable"
    assert max(ids) < manifest["tokenizer"]["vocab_size"]


def test_eos_separator_present_between_books(manifest, bundle):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(REPO_ROOT / "data" / manifest["tokenizer"]["file"]))
    eos = tok.token_to_id("<|endoftext|>")
    assert eos is not None
    counts = {b["file"]: b["n_tokens"] for b in manifest["tokens"]["per_book"]}
    assert len(counts) == len(manifest["books"])
    # every book ends with the separator token
    tokens = bundle.tokens
    first = counts[manifest["books"][0]["file"]]
    assert tokens[first - 1] == eos
    assert tokens[first] != eos or first < 10


# --------------------------------------------------------------- batch stream
def _fingerprint_in_subprocess(seed: int, held_out: bool = False, batches: int = 100) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    cmd = [sys.executable, str(FINGERPRINT), "--seed", str(seed), "--batches", str(batches)]
    if held_out:
        cmd.append("--held-out")
    out = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert out.returncode == 0, f"fingerprint subprocess failed: {out.stderr[-2000:]}"
    return out.stdout.strip().split()[0]


def test_same_seed_identical_batch_stream_across_processes(bundle):
    """Two separate processes, same seed => byte-identical first-100-batch SHA."""
    a = _fingerprint_in_subprocess(seed=0)
    b = _fingerprint_in_subprocess(seed=0)
    assert a == b, "same seed produced different batch streams across processes"
    assert len(a) == 64
    # and the in-process fingerprint agrees with the subprocess one
    assert a == stream_sha256(bundle, n_batches=100, train_seed=0)


def test_different_seed_changes_the_stream(bundle):
    """Negative control: a seed-oblivious stream would fail here."""
    a = stream_sha256(bundle, n_batches=20, train_seed=0)
    b = stream_sha256(bundle, n_batches=20, train_seed=1)
    assert a != b


def test_held_out_stream_is_disjoint_from_training_stream(bundle):
    cut, total = bundle.split_bounds()
    assert cut < total
    train_region = bundle.tokens[:cut]
    held_region = bundle.tokens[cut:total]
    assert train_region.size == cut and held_region.size == total - cut
    # position-level disjointness: the held-out token positions are never part
    # of the training region (they may share *values*, which is expected).
    assert not np.shares_memory(train_region, held_region)
    train_fp = stream_sha256(bundle, n_batches=20, train_seed=0, held_out=False)
    held_fp = stream_sha256(bundle, n_batches=20, train_seed=0, held_out=True)
    assert train_fp != held_fp


def test_windows_are_next_token_pairs(bundle):
    it = bundle.stream(l=128, batch=8, train_seed=3)
    x, y = next(it)
    assert x.shape == (8, 128) and y.shape == (8, 128)
    assert (y[:, :-1] == x[:, 1:]).all(), "y must be x shifted by one token"
    assert int(x.min()) >= 0 and int(x.max()) < 5000


def test_stream_depends_on_corpus_and_tokenizer_identity(bundle):
    """Negative control: the seed material must actually bind the identity."""
    digest = hashlib.sha256(bundle.seed_material(0, False)).hexdigest()
    other = hashlib.sha256(bundle.seed_material(0, True)).hexdigest()
    assert digest != other
    # a different corpus id changes the seed material
    broken = CorpusBundle(root=bundle.root, manifest=dict(bundle.manifest), tokens=bundle.tokens)
    broken.manifest = dict(bundle.manifest)
    broken.manifest["corpus_id"] = "0" * 64
    assert broken.seed_material(0, False) != bundle.seed_material(0, False)


def test_mutated_corpus_file_fails_verification(tmp_path):
    """Negative control: checksum verification must be able to redden."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    book = manifest["books"][0]
    src = CORPUS_DIR / book["file"]
    fake_root = tmp_path / "repo"
    (fake_root / "data" / "corpus").mkdir(parents=True)
    shutil.copy(src, fake_root / "data" / "corpus" / book["file"])
    # mutate one byte of the copy
    target = fake_root / "data" / "corpus" / book["file"]
    data = bytearray(target.read_bytes())
    data[0] = (data[0] + 1) % 256
    target.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        CorpusBundle.verify_files(fake_root, manifest)
