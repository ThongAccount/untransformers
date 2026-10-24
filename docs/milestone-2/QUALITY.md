# Milestone 2 — the quality comparison: HE Extractor vs 32-head self-attention

**Status:** the write-up is assembled from the measured artefacts in
`results/`; every table below is rendered straight from the JSON by
`tools/render_results_tables.py`, so a number in this document is by
construction the number in the artefact.

**What this is.** A within-protocol, same-corpus comparison of two sublayer-1
implementations under one fixed training recipe, on a substitute corpus, at
3.333% of the paper's token budget. **What this is not:** a reproduction of
the paper's numbers. The paper's corpus is unrecoverable, the paper's metric
is the average training loss over a batch rather than a held-out loss, and its
"0.457 epochs" label is not reproducible under any standard reading
(`docs/milestone-1/DEVIATIONS.md` D-19). Absolute losses here are therefore
**not comparable to the paper's**, and no such comparison is made anywhere in
this document.

**Protocol:** `docs/milestone-3/PROTOCOL.md` (pre-registered at `dbc08fc`,
before any quality number existed). The statistics, the decision rule, the
disclosures and the forbidden claims are taken from there; this document
reports, it does not re-decide.

---

## 1. The conditions of the comparison

### Conditions

| field | value |
|---|---|
| label | full |
| commit | 5b296cad33e8e9911bcb5da0336f52b173054e37 |
| corpus_id | `fd567893d71494fdf5f733c13b4d4ac5e9ecf81c886ab298e1003d4103a5e2d7` |
| tokenizer sha256 | `4e6abaa775edf70839cc6b0b881b7846e3e558c577c474fcaf22516426f6c647` |
| books | 100 |
| tokens: total / train region / held-out region | 9514186 / 9038476 / 475710 |
| held-out definition | the last 5% of the frozen token array (475710 tokens); never trained on, never used for selection |
| model shape | d=128 l=128 layers=18 ffn=512 vocab=5000 heads=32 |
| steps | 2000 |
| tokens per step | 8192 |
| tokens processed per run | 16384000 |
| nominal epochs of the train region | 1.813 (windows are sampled with replacement: repeats included) |
| token budget vs the paper | 3.333% of the paper's 491,520,000 tokens (60,000 batches x 8,192) |
| seeds | [1, 2, 3] |
| eval every / eval batches | 250 / 8 |
| eval windows / target tokens per eval point | 512 windows / 65536 target tokens |
| train seed | 0 |
| device | cuda |
| hardware | Tesla T4 |
| torch | 2.11.0+cu130 |
| python | 3.13.16 |
| threads | 1 |

Every row above is the runner's own record of the run that produced
`results/quality_gpu_full.json`; nothing in it is retyped by hand.

## 2. The corpus, and what the held-out split does and does not measure

The corpus is the pinned Gutenberg "Children's Literature" bookshelf-20
snapshot (first 100 ebook IDs in ascending order), tokenised with a 5,000-token
byte-level BPE fitted on the whole snapshot. `data/manifest.json` carries the
listing digest, the per-book digests, the token digests and the `corpus_id`;
the token array is frozen in `data/tokens.npy`.

The split is the last 5% of the frozen token array (475,710 tokens) held out
from training. The evaluation windows are drawn from that region only, and the
stream is seeded by `SHA-256(train_seed | split | corpus_id | tokenizer_sha256)`,
so **every arm, every seed and every eval point is scored on exactly the same
windows**.

Three disclosures are binding on this write-up:

* **D-A — the cut is mid-book.** The boundary falls inside ebook 3499, so the
  held-out region mixes two different things: unseen books (85.5% of the
  held-out tokens) and the unseen tail of a book whose beginning was in
  training (14.5%). The two are reported separately in §6 (the book-aligned
  sensitivity analysis), never merged into one "held-out loss" claim.
* **D-B — n = 3 books.** The unseen-book evidence rests on exactly three books
  (ebooks 3536, 3795, 3796). Book-level variance is unmeasured. This document
  says "on three unseen books", never "on unseen books".
* **D-C — the tokenizer saw the held-out text.** `tools/build_tokenizer.py`
  fits the BPE merges on every book in the manifest, the held-out region
  included. This is a mild leak, it affects both arms identically, and it is
  recorded as deviation **D-22**. It cannot be removed without re-tokenising
  and re-running, which would break the provenance of the run.

Two further conditions of the corpus, from the protocol review:

* **C-1 — the selection is not random.** "First 100 by ascending ebook ID" is
  deterministic but arbitrary; low IDs are older Gutenberg records, so the
  corpus skews toward 19th-century children's books. This is a
  representativeness limitation of the substitute corpus, not of the protocol.
* **C-4 — no absolute comparison to the paper.** Neither the losses nor the
  perplexities below may be compared with the paper's numbers.

## 3. The budget, and the word "epoch"

Windows are sampled **with replacement**, so the epoch figures are *nominal*:
they are tokens processed divided by the size of the train region, repeats
included. This run processed 16,384,000 tokens per run — **1.813 nominal
epochs** of the 9,038,476-token train region, and **3.333%** of the paper's
491,520,000-token budget (60,000 batches × 8,192 tokens). The paper's own
"0.457 epochs" label is not reproducible (D-19), so every budget comparison
here is made in **tokens**, and the token budget is in the header of every
table.

## 4. The endpoints, and the decision rule that was fixed in advance

* **Primary endpoint:** the held-out loss at the final step (2,000).
* **Secondary endpoint:** the mean of the last four eval points (1,250, 1,500,
  1,750, 2,000).
* **The comparison is reported as paired per-seed differences**
  `d_s = L_HE(s) − L_attention-32h(s)`, with the sign convention stated in
  the table: a positive `d̄` means HE's held-out loss is *higher* (worse).
* The sample SD uses **ddof = 1**; the 95% CI is `d̄ ± t₀.₉₇₅,₂ · s_d/√3`
  (`t₀.₉₇₅,₂ = 4.3027`); the **MDD at 80% power** is
  `(t₀.₉₇₅,₂ + t₀.₈₀,₂) · s_d/√3` with `t₀.₈₀,₂ = 1.0607`.
* **Decision rule, fixed before the numbers existed:** the comparison is
  *resolved* iff the 95% CI for `d̄` excludes 0. If it does not, the result is
  "not resolved at n = 3", with the MDD printed as the smallest effect this
  protocol could have detected, and the number of seeds that would have been
  needed at 80% power for the effect that was observed. "Not resolved" is a
  legitimate result and is reported as one.
* No p-value appears in the headline. The harness's own `std` (a population SD,
  ddof = 0, and per-arm rather than paired) is descriptive only and is never
  quoted as the dispersion.

### Endpoint: primary: held-out loss at the final step

Sign convention: d_s = L_HE(s) - L_attn(s); positive means HE's loss is HIGHER (worse)

| seed | HE | attention-32h | d = HE - attention |
|---|---|---|---|
| 1 | 4.138239 | 4.243823 | -0.105584 |
| 2 | 4.149547 | 4.256581 | -0.107034 |
| 3 | 4.151253 | 4.250221 | -0.098967 |
| **mean** | 4.146346 | 4.250208 | -0.103862 |
| **sample SD (ddof=1)** | 0.007073 | 0.006379 | 0.004300 |

| statistic | value |
|---|---|
| standard error | 0.002483 |
| 95% CI for d-bar | [-0.114544, -0.093180] |
| CI excludes 0 | True |
| MDD at 80% power | 0.013315 |
| perplexity ratio exp(d-bar) | 0.901350 |
| perplexity ratio, 95% CI | [0.891773, 0.911030] |
| verdict | resolved (the 95% CI for d̄ excludes 0) |
| seeds needed for 80% power at the observed effect | 3 |

### Endpoint: secondary: mean of the last four eval points

Sign convention: d_s = L_HE(s) - L_attn(s); positive means HE's loss is HIGHER (worse)

| seed | HE | attention-32h | d = HE - attention |
|---|---|---|---|
| 1 | 4.201763 | 4.310400 | -0.108637 |
| 2 | 4.213751 | 4.321823 | -0.108072 |
| 3 | 4.211961 | 4.322089 | -0.110129 |
| **mean** | 4.209158 | 4.318104 | -0.108946 |
| **sample SD (ddof=1)** | 0.006467 | 0.006673 | 0.001062 |

| statistic | value |
|---|---|
| standard error | 0.000613 |
| 95% CI for d-bar | [-0.111585, -0.106307] |
| CI excludes 0 | True |
| MDD at 80% power | 0.003290 |
| perplexity ratio exp(d-bar) | 0.896779 |
| perplexity ratio, 95% CI | [0.894415, 0.899149] |
| verdict | resolved (the 95% CI for d̄ excludes 0) |
| seeds needed for 80% power at the observed effect | 3 |

## 5. The eval-set bootstrap (a different kind of noise)

All arms and seeds are scored on the identical 512 windows (65,536 target
tokens, 13.8% of the held-out region), so the eval-set noise cancels in the
paired difference to a large extent. It is **not** zero for the absolute
per-arm loss, and it is reported separately: the 512 windows are resampled
with replacement (1,000 resamples, seeded), the same resample is used for both
arms, and the bootstrap SE of each arm's mean loss and of the paired
difference is printed. This SE is **not** included in the training-noise CI
above, and the two are not simply additive.

**How large is the eval-set noise?** The largest bootstrap SE of the paired
difference over the three seeds is **0.003848**, against a training-noise CI
half-width of **0.010682** — the eval-set noise is the smaller of the two, so
the interval above is dominated by the run-to-run variance of the training
runs and not by the sampling of the eval windows.

**The conditions of the eval-set re-scoring.** The 512 windows were re-scored
from the six checkpoints on the 1-core CPU host (float32, torch 2.14.1+cpu),
while the training runs themselves were on a Tesla T4 (cuda, torch
2.11.0+cu130): Colab's T4 pool was saturated for the whole window this was
attempted in, so this half was measured on the CPU and the book-aligned half
was not run at all (§6). The re-scoring is a cross-check as well as an input:
each checkpoint's recomputed eval-set mean reproduces the final held-out loss
that its run reported, to within 3 × 10⁻⁷ — the float32 CPU-vs-GPU difference,
not a modelling difference.

### Eval-set bootstrap (1,000 resamples of the 512 eval windows)

| seed | bootstrap SE, HE | bootstrap SE, attention-32h | bootstrap SE, paired difference |
|---|---|---|---|
| 1 | 0.013329 | 0.013799 | 0.003848 |
| 2 | 0.013464 | 0.013600 | 0.003675 |
| 3 | 0.013471 | 0.013825 | 0.003564 |

_this SE describes the sampling of the eval windows only; it is NOT included in the training-noise CI (2.484·s_d) and the two are not simply additive_

## 6. The book-aligned sensitivity analysis (post-hoc, three unseen books) — not run

The sensitivity analysis pre-registered in `docs/milestone-3/PROTOCOL.md` §2.3
was **not run**, and this write-up makes **no** sensitivity claim. The three
fully unseen books (ebooks 3536, 3795, 3796) hold **406,723 tokens**; the
stride-1 sweep would score **406,339 windows** per checkpoint (a window start
at every token position inside each book), each scoring 128 target positions —
**52,011,392 token-predictions** per checkpoint, which is a count of
token-predictions and not of distinct tokens (the windows overlap, and the
first token of each book is never a target).

It was not run because the measurement needs a GPU and none was available: the
re-scoring leg was queued on a Colab Tesla T4 from 2026-10-24 07:10 UTC, the
allocation endpoint kept answering `Service Unavailable`, and the leg was
stopped rather than polled (the user's instruction, on cost grounds). The CPU
fallback was measured rather than assumed — the book-aligned path costs
**18–20 ms per window** for the HE arm and **62–65 ms per window** for the
attention arm on this 1-core host, i.e. **2.1–2.2 h** and **7.0–7.4 h** per
checkpoint and ≈ 28 h for the six — and was rejected. The eval-set half of the
same re-scoring *was* run, on the CPU (§5).

What that leaves: the primary endpoint stands on its own, and the **absence**
of the sensitivity analysis is a limit on what may be claimed from it, not a
neutral detail. The protocol's rule for this metric — that it is reported
*alongside* the primary endpoint and may not replace it, and that a
disagreement between the two orderings would be reported as the finding —
could not be exercised, so this write-up says nothing about whether the arm
ordering on the three unseen books agrees with the ordering on the run's own
metric. Recorded as deviation **D-27**.

## 7. What this comparison establishes, and what it does not

**Does not establish:**

* anything about the paper's books, the paper's corpus size, or the paper's
  absolute numbers (the corpus is a substitute, the metric differs, and the
  budget is 3.333% of the paper's);
* anything about generalisation beyond these three books (D-B);
* a data-order variance estimate: **all three seeds share one batch
  sequence** (`corpus_stream_factory` seeds from `train_seed` only), so `s_d`
  covers initialisation and dropout variance only and is a **lower bound** on
  run-to-run variance; the MDD derived from it is therefore optimistic
  (protocol §3.4, limitation L-01 in `docs/milestone-3/SCOPE.md`);
* anything about the arm ordering on the three unseen books: the
  pre-registered book-aligned sensitivity analysis was **not run** (§6,
  deviation D-27), so the protocol's rule for it — report it alongside the
  primary endpoint, and report a disagreement between the two orderings as the
  finding — could not be exercised;
* anything about the tokenizer's generalisation (D-C).

**Does establish, within this protocol:**

* that the paired difference between the two arms is **resolved at n = 3**
  under this protocol: the 95% CI for d̄ [-0.114544, -0.093180] excludes 0, and
  on this corpus at this budget the HE Extractor's held-out loss is the lower
  one (d̄ = -0.103862, a perplexity ratio of 0.901350, an MDD of 0.013315 at
  80% power);
* the size of the eval-set noise separately from the training-run noise.

## 8. The CPU pilot is a protocol smoke test, not a result

`results/quality_pilot.json` is the 50-step, 2-seed CPU pilot (409,600 tokens
per run = 0.045 nominal epochs). It exists to prove the pipeline end-to-end on
the pinned corpus; its numbers are **not** a quality result and are not
compared with anything. The full sweep above is the quality run.

## 9. Reproducing this

```bash
# the run (one invocation per arm and seed, exactly as the session did)
python experiments/run_quality.py --arms HE --seeds 1 --steps 2000 \
    --eval-every 250 --eval-batches 8 --train-seed 0 --label full \
    --out runs/HE_seed1.json --checkpoint-dir checkpoints \
    --device cuda --save-checkpoints

# the merge, the pairing evidence, the evaluation-only re-scoring, the statistics
python experiments/merge_quality_runs.py --runs runs --out results/quality_gpu_full.json \
    --label full --checkpoint-dir checkpoints --commit <sha>
python tools/verify/verify_pairing_evidence.py --results results/quality_gpu_full.json \
    --checkpoint-dir checkpoints --out results/pairing_evidence_gpu_full.json \
    --merge-into results/quality_gpu_full.json
python experiments/eval_checkpoints.py --checkpoint-dir checkpoints \
    --arms HE attention-32h --seeds 1 2 3 --out results/eval_windows.json \
    --device cpu --skip-book-aligned --note "<what this run is>"
python tools/summarise_quality.py --file results/quality_gpu_full.json \
    --eval-windows results/eval_windows.json --out results/quality_summary.json
```

The committed `results/eval_windows.json` is the output of exactly that
command: the eval-set half, scored on the CPU, **without** the book-aligned
sweep. The sweep is the same script without `--skip-book-aligned` (and on a
GPU) — that is the command the protocol asks for, and it was **not run**,
because no GPU was available (§6, deviation D-27). `--skip-book-aligned`
makes the artefact record the book-aligned block as `not run` with the reason,
so a missing sensitivity analysis can never be read as a measured one.

The gates that certify this document are `G3`, `G3a`, `G4`, `G4a`, `G4b` and
`G8b` in `GATES.md`; their evidence is recorded in
`docs/milestone-3/EVIDENCE.md`.
