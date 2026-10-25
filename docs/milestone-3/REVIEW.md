# Job B — results review for overclaiming (Milestone 3)

This is the reviewer's independent verdict on the measured round. It is written
*after* the gates landed, against the artefacts in `results/`, and it does not
re-decide anything the protocol pre-registered — it checks whether what was
claimed matches what was measured, and states plainly what is not established.

The review is a command with an exit code, not prose:

```
/home/node/.venv/bin/python tools/review_recompute.py \
    --file results/quality_gpu_full.json \
    --systems results/systems.json \
    --eval-windows results/eval_windows.json \
    --summary results/quality_summary.json \
    --out results/review_recompute_gpu_full.json
```

Exit code 0: `REVIEW_RECOMPUTE_OK: 162 checks`. Every check compares the
reviewer's independent recomputation against the artefact and against the
builder's summary, digit for digit where numbers are involved. The full output
is committed as `docs/milestone-3/logs/review-recompute.txt`; `/workspace/temp`
is cleared when this task ends, so the committed copy is the only source the
evidence ledger can quote.

## The numbers (recomputed independently, agreed with the artefact)

Quality leg: commit `5b296ca`, Tesla T4, cuda, 2000 steps,
16,384,000 tokens per run (1.813 nominal epochs of the 9,038,476-token train
region = 3.333% of the paper's 491,520,000-token budget), seeds [1, 2, 3],
512 eval windows per eval point (65,536 target tokens, 13.8% of the held-out
region). Corpus: `corpus_id fd567893...`, tokenizer `sha256
4e6abaa7...` (fitted on the whole corpus — disclosure D-C / deviation D-22).

Primary endpoint = held-out loss at step 2000.
Secondary endpoint = mean of the last four eval points (1250/1500/1750/2000).
Sign convention: `d_s = L_HE(s) − L_attention-32h(s)`; positive means HE's loss
is HIGHER (worse).

| endpoint | seed | HE | attention-32h | d = HE − attention |
|---|---|---|---|---|
| primary (step 2000) | 1 | 4.138239 | 4.243823 | −0.105584 |
| primary (step 2000) | 2 | 4.149547 | 4.256581 | −0.107034 |
| primary (step 2000) | 3 | 4.151253 | 4.250221 | −0.098967 |
| secondary (last 4) | 1 | 4.201763 | 4.310400 | −0.108637 |
| secondary (last 4) | 2 | 4.213751 | 4.321823 | −0.108072 |
| secondary (last 4) | 3 | 4.211961 | 4.322089 | −0.110129 |

| statistic | primary | secondary |
|---|---|---|
| mean difference d̄ | −0.103862 | −0.108946 |
| sample SD s_d (ddof=1) | 0.004300 | 0.001062 |
| standard error s_d/√3 | 0.002483 | 0.000613 |
| 95% CI for d̄ | [−0.114544, −0.093180] | [−0.111585, −0.106307] |
| CI excludes 0 | yes | yes |
| MDD at 80% power | 0.013315 | 0.003290 |
| seeds needed at 80% power (observed effect) | 3 | 3 |
| perplexity ratio exp(d̄) | 0.901350 | 0.896779 |
| verdict | **resolved at n = 3** | **resolved at n = 3** |

The reviewer's recomputation agrees with the artefact digit for digit:
`abs_diff = 0.000e+00` on every numeric check. The harness's own `std` (a
population SD, ddof = 0) is classified as descriptive only; the CI uses the
sample SD with ddof = 1, as the protocol requires. The eval-set bootstrap SE
(0.003848 at seed 1, largest over the three seeds) is reported separately from
the training-noise CI half-width (0.010682); the two are not additive, and the
bootstrap block never enters the CI.

## Claim-by-claim verdict table

| # | claim | where it is made | what the artefact says | reviewer's recomputation | verdict |
|---|---|---|---|---|---|
| Q1 | The paired quality comparison is resolved at n = 3 under the pre-registered rule (CI excludes 0). | QUALITY.md §4/§7; SCOPE.md §3 criterion 1 | Primary CI [−0.114544, −0.093180] excludes 0; secondary CI [−0.111585, −0.106307] excludes 0 | reviewer d̄ = −0.103862, CI [−0.114544, −0.093180], MDD 0.013315, seeds needed = 3; exact digit-for-digit agreement | **RESOLVED** |
| Q2 | HE's held-out loss is lower than attention-32h's (d̄ < 0). | QUALITY.md §7 | All three per-seed differences negative (−0.105584, −0.107034, −0.098967); d̄ = −0.103862 | agrees, sign and magnitude | **RESOLVED** (within this protocol) |
| Q3 | The systems differences exceed the run-order drift floor of their context. | SYSTEMS.md §1 "Arm-to-arm differences against the drift floor" | HE vs attention-32h resolvable in every block/context; HE vs attention-32h-sdpa resolvable in every block/context; attention-32h vs attention-32h-sdpa resolvable at contexts ≥ 32 in prefill and ≥ 8 in decode, not resolvable at the smallest contexts | reviewer's per-(arm, context) recomputation matches the artefact's floors abs_diff = 0.0; every difference carries its floor and verdict | **RESOLVED** (where the table says resolvable) |
| Q4 | The wall-clock differences measure the HE *mechanism*. | SYSTEMS.md intro, §3 "Not supported" | Explicitly disclaimed: HE is an eager Python offset loop; attention arms may dispatch fused SDPA; wall-clock measures implementation maturity under these conditions | N/A — the document states this itself | **NOT CLAIMED** (correctly disclaimed) |
| Q5 | Anything about the paper's own numbers (corpus, absolute loss/perplexity, "0.457 epochs"). | QUALITY.md §1/§9 "Does not establish"; SCOPE.md §3 criterion 3 | Explicitly disclaimed: substitute corpus, different metric, 3.333% of paper's budget; "no such comparison is made anywhere" | N/A | **NOT CLAIMED** (correctly disclaimed) |
| Q6 | The book-aligned sensitivity analysis ran and agrees with the primary endpoint. | PROTOCOL.md §2.3; QUALITY.md §6 | `results/eval_windows.json` records `book_aligned` as `not run` (deviation D-27); no T4 was available; CPU cost measured ≈ 28 h for six checkpoints and rejected on cost | book-aligned block is null on both sides ("no paired seeds"); the two verdicts agree because neither exists | **NOT RUN** — recorded as not run, never as measured |
| Q7 | The eval-set bootstrap SE is part of the training-noise CI. | QUALITY.md §5 | Printed separately (largest paired-difference SE 0.003848 vs CI half-width 0.010682); stated "NOT included in the training-noise CI" | reviewer reproduces both and keeps them separate | **CORRECT** (kept separate) |
| Q8 | s_d covers initialisation/dropout variance only; it is a lower bound on run-to-run variance because all seeds share one batch sequence. | QUALITY.md §7 "Does not establish"; SCOPE.md §5 L-01 | Stated as a limitation, not a measurement | N/A — the document states this itself | **CORRECT** (declared limitation) |
| Q9 | The tokenizer saw the held-out text (mild leak, affects both arms equally). | QUALITY.md §2 disclosure D-C; deviation D-22 | Stated as a disclosure; `tools/build_tokenizer.py` fits on every manifest book | N/A — disclosed, not measured | **CORRECT** (declared disclosure) |
| Q10 | The held-out cut is mid-book (inside ebook 3499); unseen-book and seen-book-tail behaviour are reported separately. | QUALITY.md §2 disclosure D-A | 85.5% of held-out tokens from three unseen books; 14.5% from the tail of ebook 3499 | N/A — disclosed | **CORRECT** (declared disclosure) |
| Q11 | The unseen-book evidence rests on n = 3 books (3536, 3795, 3796). | QUALITY.md §2 disclosure D-B | Stated as a disclosure | N/A — disclosed | **CORRECT** (declared disclosure) |
| Q12 | The CPU pilot is a smoke test, not a quality result; no pilot number appears in any comparison. | QUALITY.md §8; PLAN.md Milestone 0 | Pilot = 50 steps, 2 seeds, CPU; §8 says "its numbers are not a quality result and are not compared with anything" | The reviewer's tool reads only `quality_gpu_full.json`; the pilot file is untouched by the verdict | **CORRECT** (pilot excluded) |
| Q13 | The per-seed pairing evidence holds (same batches, same order; paired init seeds). | EVIDENCE.md G3a/G4a | `same_batches_same_order: true`, `paired_init_seeds: true`, verified on a clean tree at 41a41ec | verifier gate G4a MET (exit 0, `verified=True`) | **MET** |
| Q14 | The systems artefact was measured at the commit it names, and that commit's source carries the same-context null. | review's own provenance check | `environment.git_commit = b2a3221...`; `git show b2a3221:experiments/run_systems.py` contains `aa_null_by_context` (4×); artefact itself carries `aa_null_by_context` | artefact and recorded commit agree; no cross-context null in either | **MET** |

## EVIDENCE.md rows (one per command)

| gate | command | exit | matched token | wall | source |
|---|---|---|---|---|---|
| G2 | `python -m pytest -q` | 0 | `154 passed, 1 skipped` | 15.81 s | `docs/milestone-3/logs/gate_G2.txt` |
| G4 | `tools/verify/check_quality_results.py --file results/quality_gpu_full.json --label full --arms HE attention-32h --seeds 1 2 3` | 0 | `FULL_OK` | 0.06 s | `docs/milestone-3/logs/gate_G4.txt` |
| G4a | `tools/verify/verify_pairing_evidence.py --results results/quality_gpu_full.json --checkpoint-dir results/checkpoints --out results/pairing_evidence_gpu_full.json --merge-into results/quality_gpu_full.json` | 0 | `verified=True` (both claims) | 2.16 s | `docs/milestone-3/logs/gate_G4a.txt` |
| G4b | `tools/summarise_quality.py --file results/quality_gpu_full.json --eval-windows results/eval_windows.json --out results/quality_summary.json` | 0 | `written: results/quality_summary.json` | 0.17 s | `docs/milestone-3/logs/gate_G4b.txt` |
| G6 | `tools/verify/check_systems_results.py --file results/systems.json` | 0 | `SYSTEMS_OK` | 0.07 s | `docs/milestone-3/logs/gate_G6.txt` |
| G7 | `tools/verify/check_systems_results.py --file results/systems.json --require-paired-quality` | 0 | `PAIRED_QUALITY_OK` + `SYSTEMS_OK` | 0.07 s | `docs/milestone-3/logs/gate_G7.txt` |
| G8c | `tools/verify/check_doc_numbers.py --doc docs/milestone-2/QUALITY.md --results results/quality_gpu_full.json --eval-windows results/eval_windows.json --manifest data/manifest.json --summary results/quality_summary.json` | 0 | `DOC_NUMBERS_OK` (12 constants) | 0.06 s | `docs/milestone-3/logs/gate_G8c.txt` |
| — (reviewer) | `tools/review_recompute.py --file results/quality_gpu_full.json --systems results/systems.json --eval-windows results/eval_windows.json --summary results/quality_summary.json --out results/review_recompute_gpu_full.json` | 0 | `REVIEW_RECOMPUTE_OK: 162 checks` | — | `docs/milestone-3/logs/review-recompute.txt` |

G8 and G8b remain PENDING until SCOPE.md §4 carries a decision; G9 remains PENDING until this commit is pushed. Both are recorded as waiting on this review, not as having passed without one.

## What this review does not establish

* A claim about the paper's books, corpus size, or absolute numbers (criterion 3).
* Generalisation beyond the three unseen books (D-B; book-level variance unmeasured).
* A data-order variance estimate (L-01; `s_d` is a lower bound; the MDD derived from it is optimistic).
* Whether the arm ordering on the three unseen books agrees with the ordering on the run's own metric (the book-aligned sensitivity analysis was not run, D-27; the protocol's rule for it could not be exercised).
* Anything about the tokenizer's generalisation beyond this corpus (D-C).
* That the wall-clock differences are a property of the HE mechanism (implementation-maturity caveat, SYSTEMS.md §3).

## Reviewer's conclusion

The quality comparison is **resolved at n = 3** under the pre-registered rule: the 95% CI for the paired difference excludes 0, and on this corpus at this budget the HE Extractor's held-out loss is the lower one (d̄ = −0.103862, perplexity ratio 0.901350, MDD 0.013315 at 80% power, 3 seeds needed at 80% power for the observed effect). The secondary endpoint agrees in sign and resolution. The systems differences exceed the drift floor where the table says resolvable, and every wall-clock row carries its conditions.

That resolution is **within this protocol**: substitute corpus, 3 unseen books, 3.333% of the paper's token budget, no data-order variance, tokenizer disclosed as a mild leak, book-aligned sensitivity not run. The round establishes a real result, bounded by its disclosures. No number in this review is compared with the paper's, no pilot number appears in any comparison, and no absolute loss or perplexity is quoted against the paper.
