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
| D-24 | The one-sided 80% Student-t quantile table in `tools/summarise_quality.py` was wrong for 29 of its 30 rows (found 2026-10-24 by the reviewer's independent re-computation) | fixed **before** any measured number existed | the table listed the 80% normal quantile and one wrong entry (1.282 at df = 1 where the exact quantile is 1.376; 0.466 at df = 30 where it is 0.838), and the "seeds needed for 80% power at the observed effect" loop used it — understating the requirement by 10–24% (12 seeds instead of 15 at an effect of 0.8·s_d). The builder now inverts the Student-t CDF numerically instead of reading a table, and `verify_protocol_constants()` refuses to print if the computed quantiles stop reproducing the protocol's stated constants (t₀.₉₇₅,₂ = 4.3027, t₀.₈₀,₂ = 1.0607, CI 2.484·s_d, MDD 3.097·s_d). The protocol fixes the statistics, not a table, so the pre-registration is unchanged |
| D-25 | The systems runner's A/A null compared an arm's first and last slot across the whole block, not at the same context (found 2026-10-24 by the reviewer's independent re-computation) | fixed **before** the systems leg ran | the prefill and incremental-decode blocks carry one row per context length per slot, so the arm-level comparison was dominated by the growth of the latency with the context rather than by run-order drift: on the CPU rehearsal artefact the prefill "floor" was 0.052005 s (a context gap) against 0.003183 s as a same-context null. The null is now computed per (arm, context) and the artefact carries `aa_null_by_context` and `floor_by_context_seconds`; the renderer judges each difference against the floor of its own context and the G6 checker recomputes the null from the slot rows and fails if the artefact's null is not the same-context one |
| D-26 | The systems runner's sublayer-1 scaling warm-up handed the whole context to the eager attention path at once, and the score matrix that path materialises does not fit on a T4 (found 2026-10-24 in the first GPU systems leg, which died with a CUDA OOM **before it wrote an artefact**) | fixed **before** any systems number existed | at T = 8192 with 32 heads the eager attention path allocates 8.6 GB per score tensor and three of them are alive at once — more than the T4's 14.56 GB. The fix is in the runner's **warm-up**, not in the measured implementation: the cache is filled in pieces of 256 positions. The warm-up is not timed (the measured call is a single new token at position T-1), and the cache it leaves is bit-identical — a token's K and V do not depend on how the warm-up was cut, and HE's rolling state keeps the last l-1 vectors either way (`tests/test_systems_warmup.py`) |
| D-27 | The book-aligned stride-1 sensitivity sweep (protocol §2.3, the three fully unseen books) was **not run** (found 2026-10-24) | **open** — the pre-registered sensitivity analysis is an outstanding item, and the write-up makes no sensitivity claim | the re-scoring leg was queued on a Colab Tesla T4 from 07:10 UTC; the allocation endpoint kept answering `Service Unavailable` and the leg was stopped rather than polled (the user's instruction, on cost grounds). The CPU fallback was measured, not assumed: the book-aligned path costs 18-20 ms/window (HE) and 62-65 ms/window (attention) on this 1-core host — 2.1-2.2 h and 7.0-7.4 h per checkpoint, ≈ 28 h for the six — and was rejected (two measurements of the same path; `docs/milestone-3/logs/cpu-book-aligned-cost.txt`). The eval-set half of the same re-scoring *was* run, on the CPU: its six recomputed means reproduce the runs' reported final held-out losses to within 3 × 10⁻⁷ (the float32 CPU-vs-GPU difference). `experiments/eval_checkpoints.py --skip-book-aligned` makes the artefact record the book-aligned block as `not run` with the reason, so a missing sensitivity analysis can never be read as a measured one. `docs/milestone-2/QUALITY.md` §6 states the gap and §7 moves the book-aligned claim into "does not establish" |
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
