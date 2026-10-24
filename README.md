# UnTransformers

*Transformers is popular. But it consumes countless GPUs and dedicated chips just to keep up with the pace. This repo goes the other way around.*

**Hugging Face Transformers, with the arrow reversed.**

---

## The other way around

Every trick the field reached for to keep transformers affordable points the same direction:

> **Shrink the model until it fits the machine.**

Quantize to 4 bits. Prune the experts. Distill 70B into 7B. Truncate the context to 4K. Serve a different
model than the one that was trained, and call it "good enough."

UnTransformers reverses the axis:

> **Bend the model until it fits the machine.**

---

## What this repository contains

The repository is also a research log: a from-scratch study of one idea from
the HE paper — *the HE Extractor sublayer against a 32-head self-attention
sublayer* — measured under a single pinned protocol, with every artefact,
every gate and every write-up committed next to the code that produced them.

| what | where |
|---|---|
| the plan, the protocol and the decisions | [`PLAN.md`](PLAN.md), [`docs/milestone-3/PROTOCOL.md`](docs/milestone-3/PROTOCOL.md) (pre-registered before any number existed), [`docs/milestone-1/DEVIATIONS.md`](docs/milestone-1/DEVIATIONS.md) |
| the corpus, the tokenizer and the token array | [`docs/milestone-2/CORPUS.md`](docs/milestone-2/CORPUS.md), [`data/manifest.json`](data/manifest.json) |
| the measured artefacts | [`results/`](results) — the merged quality runs, the systems comparison, the checkpoint digests |
| the write-ups | [`docs/milestone-2/QUALITY.md`](docs/milestone-2/QUALITY.md), [`docs/milestone-3/SYSTEMS.md`](docs/milestone-3/SYSTEMS.md), [`docs/milestone-3/SCOPE.md`](docs/milestone-3/SCOPE.md) |
| the gates and their evidence | [`GATES.md`](GATES.md), [`docs/milestone-3/EVIDENCE.md`](docs/milestone-3/EVIDENCE.md) |
| the reviewer's independent re-computation | [`tools/review_recompute.py`](tools/review_recompute.py) — the Job-B verdict is the reviewer's |

The rules the round was run under, in one line each: every number carries its
conditions; nothing is interpreted before it lands; the statistics and the
decision rule were pre-registered; the tables are rendered from the artefacts
rather than retyped; and the gates are commands with exit codes, so a gate
that cannot fail the right way is a defect rather than a pass.

Reproduce it from the manifest alone: the corpus pipeline
(`tools/build_tokenizer.py`, `hextractor/corpus.py`), the runners
(`experiments/run_quality.py`, `experiments/run_systems.py`), the checkers
(`tools/verify/`) and the test suite (`tests/`) are all in the repository, and
the suite is the first thing to run:

```bash
python -m pytest -q
```

What the round measured — and what it does not support — is in the write-ups,
not in this file.
