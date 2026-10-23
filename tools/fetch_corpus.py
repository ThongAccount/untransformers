"""Fetch the pinned corpus snapshot (T14).

Selection rule (fixed by Milestone 0, REPRODUCIBILITY.md §3):
  pool         = Project Gutenberg "Children's Literature" bookshelf 20
                 (https://www.gutenberg.org/ebooks/bookshelf/20)
  selection    = plain-text English ebooks of the shelf, sorted by Gutenberg
                 ebook ID ASCENDING, first N (default 100)
  snapshot     = every book's plain text + the shelf listing, hashed

The manifest — not the live shelf — is the corpus. Shelf membership drifts
(volunteer-curated), so re-running this script later may select a different
book set; that is why every file carries a SHA-256 and the manifest records
the listing hash.

Usage:
    python tools/fetch_corpus.py --out data/corpus --manifest data/manifest.json
    python tools/fetch_corpus.py --verify-only      # re-hash files, check manifest

No training, no interpretation — data plumbing only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

USER_AGENT = "untransformers-replication/0.1 (research corpus fetch; contact: repo owner)"
SHELF_URL = "https://www.gutenberg.org/ebooks/bookshelf/20"
PAGE_SIZE = 25

START_MARK = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
END_MARK = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)


def http_get(url: str, retries: int = 4, timeout: int = 90) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:  # noqa: PERF203
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_shelf_listing() -> tuple[list[int], str]:
    """Walk every page of the shelf listing, return (ebook ids in listing order,
    sha256 of the concatenated listing pages)."""
    ids: list[int] = []
    seen: set[int] = set()
    pages: list[bytes] = []
    start = 1
    while True:
        url = f"{SHELF_URL}?start_index={start}"
        raw = http_get(url)
        pages.append(raw)
        html = raw.decode("utf-8", errors="replace")
        found = re.findall(r'href="/ebooks/(\d+)\.html"', html)
        if not found:
            found = re.findall(r'href="/ebooks/(\d+)"', html)
        for blob in found:
            eid = int(blob)
            if eid not in seen:
                seen.add(eid)
                ids.append(eid)
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
        if "Next" not in text:
            break
        start += PAGE_SIZE
        time.sleep(0.4)
    listing_hash = sha256_bytes(b"".join(pages))
    return ids, listing_hash


def book_urls(ebook_id: int) -> list[str]:
    return [
        f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt",
        f"https://www.gutenberg.org/ebooks/{ebook_id}.txt.utf-8",
        f"https://www.gutenberg.org/ebooks/{ebook_id}.txt",
    ]


def fetch_book(ebook_id: int) -> tuple[bytes, str]:
    errs = []
    for url in book_urls(ebook_id):
        try:
            return http_get(url), url
        except RuntimeError as exc:
            errs.append(str(exc))
    raise RuntimeError(f"book {ebook_id}: no source reachable; {errs}")


def strip_gutenberg_boilerplate(text: str) -> tuple[str, bool]:
    """Remove the standard PG header/footer blocks. Returns (clean_text, stripped)."""
    start = START_MARK.search(text)
    end = END_MARK.search(text)
    if start and end and end.start() > start.end():
        body = text[start.end():end.start()]
        body = body.lstrip("\r\n")
        return body, True
    return text, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus", help="directory for cleaned book texts")
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--limit", type=int, default=100, help="books to select (by ebook ID ascending)")
    ap.add_argument("--verify-only", action="store_true", help="re-hash committed files against the manifest")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out)

    if args.verify_only:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bad = []
        for book in manifest["books"]:
            path = out_dir / book["file"]
            if not path.exists():
                bad.append(f"missing: {path}")
                continue
            got = sha256_file(path)
            if got != book["sha256_clean"]:
                bad.append(f"sha mismatch: {path} manifest={book['sha256_clean']} file={got}")
        if bad:
            print("\n".join(bad))
            return 1
        print(f"OK: {len(manifest['books'])} files match manifest checksums")
        return 0

    listing_ids, listing_hash = fetch_shelf_listing()
    print(f"shelf listing: {len(listing_ids)} books, listing sha256={listing_hash}")

    selected = sorted(listing_ids)[: args.limit]
    print(f"selected by ebook ID ascending: {selected[:5]} ... {selected[-3:]}")

    # Snapshot the shelf membership itself: the pool the selection was made
    # from, in listing order, with the listing-page hash.
    listing_snapshot = {
        "schema": "untransformers-shelf-snapshot/1",
        "shelf_id": 20,
        "shelf_name": "Children's Literature",
        "shelf_url": SHELF_URL,
        "listing_sha256": listing_hash,
        "books_in_shelf": len(listing_ids),
        "listing_order_ebook_ids": listing_ids,
        "selection_rule": (
            "All ebook IDs of bookshelf 20 at snapshot time, sorted ascending by "
            f"numeric ebook ID, first {args.limit}."
        ),
        "selected_ebook_ids": selected,
    }
    listing_path = manifest_path.parent / "shelf20_listing.json"
    listing_path.write_text(json.dumps(listing_snapshot, indent=2) + "\n", encoding="utf-8")
    print(f"listing snapshot written: {listing_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    books = []
    for i, eid in enumerate(selected, start=1):
        raw, url = fetch_book(eid)
        text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        clean, stripped = strip_gutenberg_boilerplate(text)
        clean = clean.strip("\n") + "\n"
        fname = f"{eid:05d}.txt"
        (out_dir / fname).write_text(clean, encoding="utf-8")
        books.append({
            "ebook_id": eid,
            "title": None,
            "url": url,
            "file": fname,
            "bytes_raw": len(raw),
            "sha256_raw": sha256_bytes(raw),
            "bytes_clean": len(clean.encode("utf-8")),
            "sha256_clean": sha256_bytes(clean.encode("utf-8")),
            "chars": len(clean),
            "boilerplate_stripped": stripped,
        })
        print(f"[{i}/{len(selected)}] {fname} raw={len(raw)}B clean={books[-1]['bytes_clean']}B "
              f"stripped={stripped}", flush=True)
        time.sleep(0.25)

    manifest = {
        "schema": "untransformers-corpus-manifest/1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Project Gutenberg",
        "shelf": {"id": 20, "name": "Children's Literature", "url": SHELF_URL,
                  "listing_sha256": listing_hash, "books_in_shelf": len(listing_ids)},
        "selection_rule": (
            "All ebook IDs of bookshelf 20 at snapshot time, sorted ascending by "
            f"numeric ebook ID, first {args.limit}."
        ),
        "preprocessing": (
            "UTF-8 decode, CRLF->LF, standard PG '*** START OF ... ***' / "
            "'*** END OF ... ***' header and footer blocks removed, body "
            "stripped of leading/trailing newlines, trailing newline added."
        ),
        "counts": {
            "books": len(books),
            "bytes_clean_total": sum(b["bytes_clean"] for b in books),
            "chars_total": sum(b["chars"] for b in books),
            "books_with_boilerplate_stripped": sum(1 for b in books if b["boilerplate_stripped"]),
        },
        "books": books,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"manifest written: {manifest_path} ({manifest['counts']['bytes_clean_total']} bytes of text)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
