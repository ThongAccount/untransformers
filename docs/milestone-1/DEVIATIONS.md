# Milestone 2 deviation ledger — open hyperparameters and their defaults

Carried from the Milestone-0 spec (§10) plus the ones the Milestone-2
harness fixes by default. Nothing here is a result; these are the pinned
choices the training protocol runs under until evidence says otherwise.

| ID | Item | Disposition | Default in harness |
|---|---|---|---|
| D-01 | Corpus book set (paper's "top 100 children's books" has no published list/ranking/release) | **user-approved** (closed 2026-10-23) | pinned Gutenberg "Children's Literature" bookshelf 20, first 100 by ebook ID ascending, SHA-256 manifest committed — see `docs/milestone-2/CORPUS.md` |
| D-02 | Held-out split (paper evaluates training loss only) | deviation by design | held-out stream added, reporting-only, never for training/selection |
| D-06 | LayerNorm init (paper's generic init sentence) | §7 | gain 1 / bias 0 |
| D-11 | Dropout placement exactly per Eqs. 7/18/20 | spec | two drops per sublayer path — implemented in `model.py` (Eq. 18 residual + Eq. 20 output residual) |
| D-12 | AdamW weight-decay group assignment (paper gives gamma only) | default: standard practice | decay applies to weight matrices; LayerNorm gains/biases and all biases excluded (weight_decay=0 group) |
| D-13 | Gradient clipping (paper silent) | default: OFF | `clip_grad=None`; no clipping |
| D-14 | Arm init seeds (paper: "setting the random seed a couple of times") | paired by construction | seed *s* drives BOTH arms in the same multi-seed sweep; shared parameters (embeddings, FFN, head, norms) are drawn first and are therefore bit-identical across arms, so an arm difference is not an init difference |
| D-19 | Corpus token count 9,514,186 vs the paper's 8.4M | accepted (book-set consequence of D-01) | budgets compared in **tokens processed**, not epochs (REPRODUCIBILITY.md §6) |
| D-20 | `<|endoftext|>` appended after each book | accepted | document boundary in the frozen token array; prevents windows straddling a book boundary silently |
| D-21 | Corpus texts + token array committed to the repo | accepted | public-domain texts; makes T14's checksum assertions verifiable offline |
| D-22 | The tokenizer was fitted on the **whole corpus, including the held-out region** | accepted (mild leak; not fixable without re-tokenizing and re-running) | `tools/build_tokenizer.py` iterates every book in the manifest and the manifest records `tokenizer.trained_on = "full corpus"`, so the held-out text influenced the BPE merges. The held-out metric is therefore mildly optimistic for *both* arms equally; the fix (fit on the train region only) would require re-tokenizing and re-running the sweep, which would break the provenance of the run in flight. Disclosed as F12 of `docs/milestone-3/PROTOCOL.md` |
| D-23 | The quality runner's `--device` never reached the training loop (found 2026-10-24) | fixed **before** any measured number existed | `harness.train_one_seed` built the model on the CPU and never moved it, while `run_quality.py --device cuda` wrote `device=cuda` into the environment manifest: the first "GPU" sweep trained on the CPU (14.3 s/step against 1.2 s/step for the same work in the same session, measured in-session). Fixed in `5b296ca` — the model is moved after the seeded init, the optimizer is built after the move, the batches follow — and locked by the GPU regression test `tests/test_t15_gpu_smoke.py::test_quality_training_loop_really_trains_on_the_gpu`. The only artefacts produced before the fix were 20-step probes, which were discarded |
| D-15 | Optimizer settings | paper-pinned | AdamW lr=0.001, betas (0.9, 0.999), weight decay (gamma) 0.001, eps 1e-8 |
| D-16 | Batch/context | paper-pinned | batch 64, l=128, d=128, 18 layers, FFN 512, vocab 5000 |
| D-17 | Data provider | corpus-backed since T14 | `hextractor.corpus.CorpusBundle` — pinned Gutenberg snapshot, SHA-256-verified, deterministic SHA-256-seeded stream; `synthetic_stream` retained for the mechanics dry-run only |
| D-18 | Step budget | deferred with compute plan | paper's 60,000 batches; the corpus+compute decision sets the actual budget — the harness takes `steps` as a parameter |

Evaluation mechanics: held-out loss = mean next-token CE over
`eval_batches` batches in eval mode (dropout off), reported at
`eval_every` intervals and at the final step. Multi-seed dispersion =
per-eval-point mean/std across seeds (`harness.dispersion`). The harness
returns curves only — no quality verdicts (gate 0).

Two labels the write-up must always carry:

* **the epoch figures are nominal.** Windows are sampled *with replacement*
  (`corpus.py`), so "epochs of our corpus" means "tokens processed ÷ train-region
  size", repeats included — it is not a pass over the data. The sweep's
  2,000 steps × 8,192 tokens = 16,384,000 tokens = **1.813 nominal epochs** of our
  9,038,476-token train region, i.e. **3.333%** of the paper's 491,520,000-token
  budget. The CPU pilot (50 steps = 409,600 tokens = 0.045 nominal epochs) is a
  **protocol smoke test**, not a quality result.
* **`harness.dispersion`'s std is a population SD (ddof=0) and is per-arm, not
  paired.** It is a descriptive summary only. The comparison is reported from the
  paired per-seed differences with the sample SD (ddof=1), the 95% CI and the MDD,
  pre-registered in `docs/milestone-3/PROTOCOL.md` §3.2.
