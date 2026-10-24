#!/usr/bin/env python3
"""Fill the table markers of a write-up from the measured artefacts.

The write-ups must not contain retyped numbers: every table is rendered from
the JSON the run produced. This tool replaces the markers

    <!-- TABLES:<name> -->

in a markdown file with the rendered section for that name, so the document
can be regenerated from the artefacts instead of edited by hand.

Usage:
    python tools/fill_doc_tables.py --doc docs/milestone-2/QUALITY.md \
        --quality results/quality_gpu_full.json \
        --summary results/quality_summary.json \
        --systems results/systems.json \
        --write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_results_tables as rrt  # noqa: E402

MARKER = re.compile(r"<!--\s*TABLES:([a-z0-9\-]+)\s*-->")


def build_sections(quality: Path | None, summary: Path | None,
                   systems: Path | None) -> dict:
    sections: dict = {}
    if quality is not None:
        sections["quality-conditions"] = rrt.render_quality_conditions(quality)
        sections["quality-holdout-loss"] = rrt.render_quality(quality)
    if summary is not None:
        d = json.loads(summary.read_text(encoding="utf-8"))
        sections["quality-summary-conditions"] = rrt.summary_conditions(d)
        sections["quality-summary"] = rrt.summary_endpoints(d)
        sections["quality-bootstrap"] = rrt.summary_bootstrap(d)
        sections["quality-sensitivity"] = rrt.summary_sensitivity(d)
        sections["quality-curves"] = rrt.summary_curves(d)
    if systems is not None:
        sections["systems"] = rrt.render_systems(systems)
    return sections


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--quality", default="")
    ap.add_argument("--summary", default="")
    ap.add_argument("--systems", default="")
    ap.add_argument("--write", action="store_true",
                    help="rewrite the document in place (default: print)")
    args = ap.parse_args()

    doc_path = Path(args.doc)
    if not doc_path.exists():
        print(f"FAIL: {doc_path} not found")
        return 1
    sections = build_sections(
        Path(args.quality) if args.quality else None,
        Path(args.summary) if args.summary else None,
        Path(args.systems) if args.systems else None)
    if not sections:
        print("FAIL: no artefacts were given")
        return 1

    text = doc_path.read_text(encoding="utf-8")
    missing: list[str] = []
    empty: list[str] = []

    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name not in sections:
            missing.append(name)
            return m.group(0)
        rendered = sections[name].strip()
        if not rendered:
            # a section that renders empty would silently delete a required
            # part of the write-up (the bootstrap, the sensitivity analysis):
            # that is a missing artefact, not a finished document
            empty.append(name)
            return m.group(0)
        return rendered

    filled = MARKER.sub(repl, text)
    if missing:
        print(f"FAIL: no section for the marker(s): {sorted(set(missing))}")
        return 1
    if empty:
        print(f"FAIL: the section(s) {sorted(set(empty))} rendered empty — "
              f"the artefact they need is missing or incomplete")
        return 1
    left = MARKER.findall(filled)
    if left:
        print(f"FAIL: the document still carries {sorted(set(left))}")
        return 1
    if args.write:
        doc_path.write_text(filled, encoding="utf-8")
        print(f"written: {doc_path}")
    else:
        print(filled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
