# Milestone 2 deviation ledger — open hyperparameters and their defaults

Carried from the Milestone-0 spec (§10) plus the ones the Milestone-2
harness fixes by default. Nothing here is a result; these are the pinned
choices the training protocol runs under until evidence says otherwise.

| ID | Item | Disposition | Default in harness |
|---|---|---|---|
| D-02 | Held-out split (paper evaluates training loss only) | deviation by design | held-out stream added, reporting-only, never for training/selection |
| D-06 | LayerNorm init (paper's generic init sentence) | §7 | gain 1 / bias 0 |
| D-11 | Dropout placement exactly per Eqs. 7/18/20 | spec | two drops per sublayer path — implemented in `model.py` (Eq. 18 residual + Eq. 20 output residual) |
| D-12 | AdamW weight-decay group assignment (paper gives gamma only) | default: standard practice | decay applies to weight matrices; LayerNorm gains/biases and all biases excluded (weight_decay=0 group) |
| D-13 | Gradient clipping (paper silent) | default: OFF | `clip_grad=None`; no clipping |
| D-15 | Optimizer settings | paper-pinned | AdamW lr=0.001, betas (0.9, 0.999), weight decay (gamma) 0.001, eps 1e-8 |
| D-16 | Batch/context | paper-pinned | batch 64, l=128, d=128, 18 layers, FFN 512, vocab 5000 |
| D-17 | Data provider | dry-run only | `synthetic_stream` — deterministic, held-out disjoint; replaced by the T14 corpus manifest + checksums when the corpus lands |
| D-18 | Step budget | deferred with compute plan | paper's 60,000 batches; the corpus+compute decision sets the actual budget — the harness takes `steps` as a parameter |

Evaluation mechanics: held-out loss = mean next-token CE over
`eval_batches` batches in eval mode (dropout off), reported at
`eval_every` intervals and at the final step. Multi-seed dispersion =
per-eval-point mean/std across seeds (`harness.dispersion`). The harness
returns curves only — no quality verdicts (gate 0).
