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
| D-15 | Optimizer settings | paper-pinned | AdamW lr=0.001, betas (0.9, 0.999), weight decay (gamma) 0.001, eps 1e-8 |
| D-16 | Batch/context | paper-pinned | batch 64, l=128, d=128, 18 layers, FFN 512, vocab 5000 |
| D-17 | Data provider | corpus-backed since T14 | `hextractor.corpus.CorpusBundle` — pinned Gutenberg snapshot, SHA-256-verified, deterministic SHA-256-seeded stream; `synthetic_stream` retained for the mechanics dry-run only |
| D-18 | Step budget | deferred with compute plan | paper's 60,000 batches; the corpus+compute decision sets the actual budget — the harness takes `steps` as a parameter |

Evaluation mechanics: held-out loss = mean next-token CE over
`eval_batches` batches in eval mode (dropout off), reported at
`eval_every` intervals and at the final step. Multi-seed dispersion =
per-eval-point mean/std across seeds (`harness.dispersion`). The harness
returns curves only — no quality verdicts (gate 0).
