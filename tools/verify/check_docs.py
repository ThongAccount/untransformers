#!/usr/bin/env python3
"""Gate check for the Milestone-3 write-up and the scope decision (G8).

Checks that both documents exist, carry the sections the milestone requires,
and contain no unresolved placeholders. It does not grade the prose — the
reviewer (Researcher) does that. Exit 0 prints DOCS_OK.

Usage:
    python tools/verify/check_docs.py --systems docs/milestone-3/SYSTEMS.md \
        --scope docs/milestone-3/SCOPE.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# (label, [required markers, case-insensitive substrings])
SYSTEMS_SECTIONS = [
    ("conditions", ("condition",)),
    ("environment", ("environment", "hardware", "gpu", "cpu")),
    ("model shape", ("shape", "configuration", "d=", "l=")),
    ("training step", ("training step", "train step", "step time")),
    ("prefill", ("prefill",)),
    ("incremental decoding", ("incremental",)),
    ("sublayer scaling", ("sublayer",)),
    ("weight traffic", ("weight traffic", "traffic")),
    ("paired quality", ("paired quality", "paired_quality")),
    ("context lengths", ("context length", "context_lengths", "contexts")),
]

SCOPE_SECTIONS = [
    ("option execution-only", ("execution-only", "execution only")),
    ("option broaden", ("broaden",)),
    ("option stop", ("stop",)),
    ("decision", ("decision",)),
    ("rationale", ("rationale", "reasoning", "because")),
    ("evidence", ("evidence", "measured", "results/")),
    ("what would change it", ("falsifier", "would change", "revisit", "trigger")),
]

PLACEHOLDERS = ("TODO", "TBD", "FIXME", "XXX", "PLACEHOLDER", "Lorem ipsum")
_PLACEHOLDER_RE = re.compile(
    r"\b(" + "|".join(p.replace(" ", r"\s+") for p in PLACEHOLDERS) + r")\b",
    flags=re.IGNORECASE)


def _check_doc(path: Path, sections, failures: list[str], notes: list[str]) -> None:
    if not path.exists():
        failures.append(f"{path}: file not found")
        return
    text = path.read_text(encoding="utf-8")
    low = text.lower()
    if len(text) < 400:
        failures.append(f"{path}: suspiciously short ({len(text)} chars) — a write-up, not a stub")
    if not re.search(r"^#{1,6}\s+\S", text, flags=re.MULTILINE):
        failures.append(f"{path}: no markdown headings")
    for label, markers in sections:
        if not any(m in low for m in markers):
            failures.append(f"{path}: missing section/term for '{label}' "
                            f"(looked for {markers})")
    for ph in _PLACEHOLDER_RE.findall(text):
        failures.append(f"{path}: unresolved placeholder {ph!r}")
    notes.append(f"{path}: {len(text)} chars, "
                 f"{len(re.findall(r'^#{1,6} ', text, flags=re.MULTILINE))} headings")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", required=True)
    ap.add_argument("--scope", required=True)
    args = ap.parse_args()

    failures: list[str] = []
    notes: list[str] = []

    _check_doc(Path(args.systems), SYSTEMS_SECTIONS, failures, notes)
    _check_doc(Path(args.scope), SCOPE_SECTIONS, failures, notes)

    for note in notes:
        print(f"[info] {note}")
    if failures:
        print(f"FAIL: {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"DOCS_OK: {args.systems} + {args.scope}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
