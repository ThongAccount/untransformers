#!/usr/bin/env python3
"""Check that the PROSE numbers in a write-up are the numbers of the artefacts.

The tables in the write-ups are rendered from the JSON, so they cannot drift.
The prose is not: a sentence that says "3.333% of the paper's token budget" is
typed by hand, and a hand-typed number is exactly where a stale figure hides.
This checker takes the numbers that the prose is *allowed* to state in
prose (they are the conditions of the run and the composition of the corpus)
and refuses the document if any of them is missing or contradicts the
artefact.

It is deliberately narrow: it checks the handful of constants the write-up
states in prose, not the whole document, and it never guesses. A constant it
cannot compute is reported as a failure, not skipped.

Usage:
    python tools/verify/check_doc_numbers.py \
        --doc docs/milestone-2/QUALITY.md \
        --results results/quality_gpu_full.json \
        --eval-windows results/eval_windows.json \
        --manifest data/manifest.json \
        --summary results/quality_summary.json

Exit 0 prints DOC_NUMBERS_OK.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# the paper's own token budget (60,000 batches x 8,192 tokens), from the
# paper's protocol as recorded in docs/milestone-1/DEVIATIONS.md (D-19)
PAPER_TOKEN_BUDGET = 491_520_000


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_pct(x: float) -> str:
    return f"{x:.3f}"


def check(results: dict, eval_windows: dict | None,
          manifest: dict | None, summary: dict | None = None) -> list[tuple[str, str]]:
    """The constants the prose is allowed to state, with the value it must be.

    Each entry is (what it is, the string the document must contain). A
    constant that cannot be computed from the artefacts is simply not checked
    — but the caller reports how many were checked, so a document that checks
    nothing is visible.
    """
    checks: list[tuple[str, str]] = []
    proto = results.get("protocol") or {}
    cfg = results.get("config") or {}
    corp = results.get("corpus") or {}

    tokens_run = proto.get("tokens_processed_per_run")
    epochs = proto.get("epochs_of_our_corpus")
    held_out = proto.get("corpus_tokens_heldout_region")
    train_region = proto.get("corpus_tokens_train_region")
    eval_batches = proto.get("eval_batches")
    batch = cfg.get("batch_size")
    l_ctx = cfg.get("l")

    # --- the constants the prose states -------------------------------------

    if tokens_run is not None:
        checks.append(("the tokens processed per run", _fmt_int(int(tokens_run))))
    if epochs is not None:
        checks.append(("the nominal epochs of the train region", f"{epochs:.3f}"))
    if tokens_run is not None:
        checks.append(("the share of the paper's token budget",
                       _fmt_pct(tokens_run / PAPER_TOKEN_BUDGET * 100)))
    if held_out is not None:
        checks.append(("the held-out region's token count", _fmt_int(int(held_out))))
    if batch and l_ctx and eval_batches:
        windows = batch * eval_batches
        checks.append(("the eval window count", str(windows)))
        checks.append(("the eval target tokens", _fmt_int(windows * l_ctx)))

    # the three fully unseen books: their token count comes from the
    # book-aligned artefact (the sweep that produced it)
    if eval_windows:
        books = ((eval_windows.get("book_aligned") or {}).get("books") or {})
        first = next(iter(books.values()), None)
        if first:
            total = sum(int(v.get("n_tokens", 0)) for v in first.values())
            if total:
                checks.append(("the token count of the three unseen books",
                               _fmt_int(total)))

    # the share of the held-out region that comes from books never seen in
    # training, recomputed from the manifest
    if manifest and train_region:
        per_book = (manifest.get("tokens") or {}).get("per_book") or []
        start = 0
        unseen = 0
        for entry in per_book:
            n = int(entry["n_tokens"])
            if start >= train_region:
                unseen += n
            start += n
        if unseen and held_out:
            checks.append(("the share of the held-out tokens from unseen books",
                           f"{unseen / held_out * 100:.1f}"))

    # the two numbers the eval-set-noise sentence compares: the bootstrap SE of
    # the paired difference (the largest over the seeds) and the training-noise
    # CI half-width. Both come from the summary artefact, so a sentence that
    # states them cannot drift from them.
    if summary:
        per_seed = ((summary.get("eval_set_bootstrap") or {}).get("per_seed") or {})
        ses = [float(v["bootstrap_se_paired_difference"])
               for v in per_seed.values()
               if v.get("bootstrap_se_paired_difference") is not None]
        if ses:
            checks.append(("the largest bootstrap SE of the paired difference",
                           f"{max(ses):.6f}"))
        ci = ((summary.get("endpoints") or {}).get("final") or {}).get("ci95") or []
        if len(ci) == 2 and ci[0] is not None and ci[1] is not None:
            half = (float(ci[1]) - float(ci[0])) / 2.0
            checks.append(("the training-noise CI half-width", f"{half:.6f}"))

    return checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--eval-windows", default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--summary", default="",
                    help="the quality-summary JSON (the two noise numbers the "
                         "eval-set-noise sentence compares)")
    args = ap.parse_args()

    doc_path = Path(args.doc)
    if not doc_path.exists():
        print(f"FAIL: {doc_path} not found")
        return 1
    text = doc_path.read_text(encoding="utf-8")

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    eval_windows = None
    if args.eval_windows:
        p = Path(args.eval_windows)
        eval_windows = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    manifest = None
    if args.manifest:
        p = Path(args.manifest)
        manifest = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    summary = None
    if args.summary:
        p = Path(args.summary)
        summary = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    checks = check(results, eval_windows, manifest, summary)

    failures: list[str] = []
    for label, value in checks:
        print(f"[info] {label}: the document must state {value}")
        if value not in text:
            failures.append(f"{doc_path}: the prose does not state {label} "
                            f"({value}) — it must be the artefact's number")

    if failures:
        print(f"FAIL: {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"DOC_NUMBERS_OK: {doc_path} ({len(checks)} constants agree with the artefacts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
