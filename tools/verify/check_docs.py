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

QUALITY_SECTIONS = [
    ("conditions", ("condition",)),
    ("the corpus", ("corpus", "corpus_id")),
    ("the held-out definition", ("held-out", "held out")),
    ("the primary endpoint", ("primary endpoint", "final step")),
    ("the paired differences", ("paired difference", "differences")),
    ("the confidence interval", ("95% ci", "ci95", "confidence interval")),
    ("the minimum detectable difference", ("mdd", "minimum detectable")),
    ("the eval-set bootstrap", ("bootstrap",)),
    ("the book-aligned sensitivity analysis", ("book-aligned", "sensitivity")),
    ("the disclosures", ("d-a", "d-b", "d-c", "disclosure")),
    ("the limitations", ("limitation",)),
    ("the token budget", ("token budget", "tokens processed", "nominal epoch")),
    ("the pilot is not a result", ("smoke test", "pilot")),
]

PLACEHOLDERS = ("TODO", "TBD", "FIXME", "XXX", "PLACEHOLDER", "Lorem ipsum",
                "filled in", "to be written", "to be filled")
_PLACEHOLDER_RE = re.compile(
    # a word boundary is not enough: markdown italics put an underscore right
    # against the word ("_filled in below_"), and "_" is a word character
    r"(?<![A-Za-z0-9])(" +
    "|".join(p.replace(" ", r"\s+") for p in PLACEHOLDERS) + r")(?![A-Za-z0-9])",
    flags=re.IGNORECASE)

# The scope decision must exist: §4 has to name one of the three options and
# carry a body. Without this, the section markers ("decision", "rationale",
# "evidence", "falsifier") are satisfied by the surrounding prose and the gate
# passes on a document that decides nothing.
SCOPE_OPTIONS = ("execution-only", "execution only", "broaden", "stop")
SCOPE_DECISION_MIN_CHARS = 400


def _scope_decision_sections(text: str) -> tuple[str, str]:
    """The decision section's body and the rationale that goes with it.

    The decision section is found by its heading, with a priority: the heading
    that carries the rationale first, then a heading numbered 4, then a heading
    that is the decision. The surrounding sections ("the criteria the decision
    is made against") must not be mistaken for the decision itself.

    The rationale may live in the same section or in the section right after
    it, so both are returned and checked together.
    """
    headings = list(re.finditer(r"^#{2,6}\s+(?P<title>.+)$", text, flags=re.MULTILINE))

    def body_of(index: int) -> str:
        h = headings[index]
        rest = text[h.end():]
        nxt = re.search(r"^#{2,6}\s+", rest, flags=re.MULTILINE)
        return (rest[:nxt.start()] if nxt else rest).strip()

    chosen = None
    for predicate in (
        lambda t: "rationale" in t,
        lambda t: re.match(r"^(?:§\s*)?4[.)]\s", t) is not None,
        lambda t: t.strip().startswith("decision"),
    ):
        for i, h in enumerate(headings):
            if predicate(h.group("title").lower()):
                chosen = i
                break
        if chosen is not None:
            break
    if chosen is None:
        return "", ""

    title = headings[chosen].group("title").lower()
    decision = body_of(chosen)
    rationale = ""
    if "rationale" not in title and chosen + 1 < len(headings):
        nxt_title = headings[chosen + 1].group("title").lower()
        if "rationale" in nxt_title or "reason" in nxt_title:
            rationale = body_of(chosen + 1)
    return decision, rationale


def _check_scope_decision(text: str, path: Path, failures: list[str]) -> None:
    """The decision section must decide, not defer."""
    decision, rationale = _scope_decision_sections(text)
    if not decision:
        failures.append(f"{path}: no decision section (it must carry the "
                        f"decision and its rationale)")
        return
    if len(decision) < SCOPE_DECISION_MIN_CHARS:
        failures.append(f"{path}: the decision section is {len(decision)} chars — "
                        f"a decision needs at least {SCOPE_DECISION_MIN_CHARS}")
    low = (decision + "\n" + rationale).lower()
    if not any(o in low for o in SCOPE_OPTIONS):
        failures.append(f"{path}: the decision section does not name an option "
                        f"(one of {', '.join(SCOPE_OPTIONS)})")
    if not any(k in low for k in ("because", "rationale", "reasoning", "since")):
        failures.append(f"{path}: the decision section gives no rationale")


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
    ap.add_argument("--quality", default="",
                    help="the quality write-up (checked when given)")
    args = ap.parse_args()

    failures: list[str] = []
    notes: list[str] = []

    _check_doc(Path(args.systems), SYSTEMS_SECTIONS, failures, notes)
    _check_doc(Path(args.scope), SCOPE_SECTIONS, failures, notes)
    scope_path = Path(args.scope)
    if scope_path.exists():
        _check_scope_decision(scope_path.read_text(encoding="utf-8"), scope_path, failures)
    if args.quality:
        _check_doc(Path(args.quality), QUALITY_SECTIONS, failures, notes)

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
