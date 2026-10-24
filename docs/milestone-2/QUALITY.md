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

<!-- TABLES:quality-conditions -->

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

<!-- TABLES:quality-summary -->

## 5. The eval-set bootstrap (a different kind of noise)

All arms and seeds are scored on the identical 512 windows (65,536 target
tokens, 13.8% of the held-out region), so the eval-set noise cancels in the
paired difference to a large extent. It is **not** zero for the absolute
per-arm loss, and it is reported separately: the 512 windows are resampled
with replacement (1,000 resamples, seeded), the same resample is used for both
arms, and the bootstrap SE of each arm's mean loss and of the paired
difference is printed. This SE is **not** included in the training-noise CI
above, and the two are not simply additive.

<!-- TABLES:quality-bootstrap -->

## 6. The book-aligned sensitivity analysis (post-hoc, three unseen books)

Every checkpoint is re-scored, without any retraining, on a **book-aligned**
set: the three fully unseen books (ebooks 3536, 3795, 3796), swept
exhaustively with stride 1 (a window start at every token position inside each
book), so this metric carries no window-sampling noise at all.

**What the coverage is, precisely.** The windows overlap: each window scores
`l` target positions, and consecutive windows share all but one of them. The
count of scored positions is therefore a count of **token-predictions**, not of
distinct tokens, and the first token of each book is never a target (a window
needs `l + 1` tokens to predict its last one). The coverage table below prints
the books' token counts, the window counts, the stride and the token-prediction
counts, so the coverage cannot be read as larger than it is.

This is a **post-hoc sensitivity analysis**. The primary endpoint stays the
run's own metric; this number is reported *alongside* it and may not replace
it. If the two orderings disagree, the disagreement is the finding — the
favourable one is not picked.

<!-- TABLES:quality-sensitivity -->

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
* anything about the tokenizer's generalisation (D-C).

**Does establish, within this protocol:**

* the paired per-seed difference between the two arms, with its uncertainty
  and its resolution status;
* the same comparison restricted to the three unseen books;
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
    --arms HE attention-32h --seeds 1 2 3 --out results/eval_windows.json --device cuda
python tools/summarise_quality.py --file results/quality_gpu_full.json \
    --eval-windows results/eval_windows.json --out results/quality_summary.json
```

The gates that certify this document are `G3`, `G3a`, `G4`, `G4a`, `G4b` and
`G8b` in `GATES.md`; their evidence is recorded in
`docs/milestone-3/EVIDENCE.md`.
