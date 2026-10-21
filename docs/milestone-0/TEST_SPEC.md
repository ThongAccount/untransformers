# Milestone 0 — Deterministic unit-test suite definition

All tests are deterministic (fixed seeds), CPU-runnable by default (a GPU smoke
subset is marked [gpu]), and pure-PyTorch (no kernels). The oracle for HE is a
literal double-loop implementation of the paper's equations — the only accepted
reference. Suite must pass before ANY training or performance number is
produced (gate 0). Implemented at `tests/` (Milestone 1), pinned by CI.

## Suite T1 — Equation-level reference check (the core gate)
- Oracle: float64 double loops — `x_ext = X@W_in_ext`; for each i:
  `x_out_ext[i] = Σ_{j=1..i} x_ext[j] ∘ w_ext[i-j+1]`;
  `X_adj = (X@W_adj) ∘ X_out_ext`; optional `X_out = X_adj @ W_out`.
- Cases: t ∈ {1, 2, 3, l−1, l}, d ∈ {8, 128}, W^out ∈ {ON, OFF}, random weights
  from a seeded generator (not the init distribution — shapes/values independent).
- Pass: max abs diff vs oracle ≤ 1e-5 (fp32) with d and t at full size; exact
  match on the constant-vector probe of T2.

## Suite T2 — Offset convention (catches the axis-flip bug)
- Set w_ext so that exactly one offset k* is active (gate = e-channel basis at
  one channel, all others zero) and input rows to distinct constant vectors.
- Assert: output row i, that channel == (row i−k*+1's projected value) for all
  i ≥ k*, and == 0 for i < k*. Run for k* ∈ {1, 2, t} — this pins
  **k = i−j+1 with k=1 = newest** (HE_SPEC §3).
- Pass: exact equality (fp32 exact on integer-valued constants).

## Suite T3 — Causality (the correctness invariant of the whole project)
- Random input, eval mode (dropout off), fixed weights. Compute output; perturb
  any token row j > i (i = every valid boundary) and recompute.
- Assert: rows ≤ i bit-identical. Repeat for i ∈ {1, t/2, t−1}, t ∈ {2, l}.
- Pass: bit-exact (not tolerance) — this is the test the Extractor's whole claim
  rests on, and the same property makes HE trivially prefix-consistent.

## Suite T4 — Causal boundary
- t=1: output must equal `(x_1@W_adj) ∘ ((x_1@W_in_ext) ∘ w_ext[0])` (×W^out if
  ON) — no reads of nonexistent past, no padding artifacts.
- Pass: matches T1 oracle at t=1 (already covered) AND parameter-gradient test:
  gates w_ext[1..l−1] get zero gradient from a t=1 forward (T8 partial).

## Suite T5 — Shape / dtype / device
- Parametrized d × l × t grid incl. non-power-of-2 d and t < l; assert output
  shape [t,d], dtype fp32, device follows input; parameter shapes exactly
  W_in_ext [d,d], w_ext [l,d], W_adj [d,d], W_out [d,d] or absent.
- Pass: no implicit casts (assert `out.dtype == torch.float32`).

## Suite T6 — Parameter count (the paper's headline parity claim)
- At d=l=128: HE (W^out ON) params == 65,536 == attention-32h params == 4d².
  WE == 49,152; ME == 128. Also assert HE == 3d² + l·d as a formula over
  several (d,l).
- Pass: exact counts (this test being green is what licenses the phrase
  "matched parameter count" anywhere in the report).

## Suite T7 — Initialization policy
- Seed, build, assert: every weight tensor sampled ~ N(0, 0.01) (Kolmogorov–
  Smirnov-ish check: mean within 3σ/√n, std within tolerance); all biases zero;
  embedding matrices equal raw N(0,0.01) × √d; LayerNorm gain 1 / bias 0.
- Pass: both arms built under the same seeded stream order produce identical
  init values (cross-arm reproducibility), and the statistical gate holds.

## Suite T8 — Gradient checks
- `torch.autograd.gradcheck` (float64, eps 1e-6) on a small HE (d=4, l=4, t=3),
  W^out ON and OFF.
- Analytic cross-check on the offset gates: with x_ext fixed,
  `dL/dw_ext[i-j+1] = dL/dx_out_ext[i] ⊙ x_ext[j]` — verify against autograd.
- Gate-sparsity: t=1 forward ⇒ grads zero for w_ext[1:].
- Pass: gradcheck exact; gate grads match analytic within 1e-7 (fp64).

## Suite T9 — Adjustment is per-position (not last-token)
- Input where row i's vector is distinguishable; assert row i's output depends on
  row i's own input through W^adj (scale row i's input pre-W^adj and watch the
  change stay at row i), and does NOT depend on the last row's input. This pins
  Eq. 24's matrix reading over the prose (HE_SPEC §4).
- Pass: output at row i invariant to changing rows > i except through their own
  positions (combine with T3's machinery).

## Suite T10 — Attention baseline reference
- Same treatment for the paper's Eqs. 10–17: oracle multi-head attention with
  the Eq. 14 causal mask; mask boundary (row 1 attends only to itself); scaling
  1/√h with h = d/n; concat + W^out_att, no biases. Compare vs a straightforward
  `F.scaled_dot_product_attention(is_causal=True)` implementation.
- Param count n-independence (65,536 for n ∈ {1, 32}).
- Pass: tolerance 1e-5 vs oracle; exact mask boundary.

## Suite T11 — Optional-projection equivalence
- With W^out forced to identity, ON ≡ OFF (same forward, same grads modulo the
  unused matrix's zero grads). Assert both code paths exist and agree.
- Pass: bit-identical forwards.

## Suite T12 — End-to-end determinism
- Full model (both arms) in eval mode, same input twice: bit-identical logits.
- Train-mode dropout: with the same seed, two forwards identical; different
  seeds differ (proves dropout is live, not silently off).

## Suite T13 — Op-count counter vs Table 7
- Our analytic counter (HE_SPEC §9) on HE and WE must reproduce the paper's
  Table-7 numbers exactly; attention counters may differ (paper's attention
  counting is internally loose — documented) — assert the HE/WE equality, and
  report (not force) the attention delta. Pass criterion: HE/WE exact match,
  attention delta recorded in the test log.

## Suite T14 — Data pipeline determinism
- Same seed ⇒ byte-identical batch stream across two processes; tokenizer
  round-trip stable; manifest checksums match the committed corpus files.
- Pass: identical SHA of the first 100 batches' token tensors.

## Suite T15 — GPU smoke [gpu]
- All of T1/T3/T6/T8 at d=128, l=128, t=128 on one GPU, fp32; record env
  manifest and timing (timing is NOT a pass criterion — correctness only).
- Pass: same assertions as CPU.

## Coverage / gate mapping
Every gate-0 exit criterion in HE_SPEC.md §11 and REPRODUCIBILITY.md §8 maps to
at least one suite: spec→T1–T5,T9–T11; params→T6; init→T7; grads→T8; data→T14;
gpu→T15; op accounting→T13; determinism→T12. A test that cannot fail is not a
test: each suite includes at least one negative control (mutated offset axis,
broken mask, last-token adjustment, identity-W^out shortcut) that must redden
the suite before the fix lands.
