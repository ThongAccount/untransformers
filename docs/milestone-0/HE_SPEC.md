# Milestone 0 — HE Extractor implementation specification

Source: arXiv:2308.07661v2, "Attention Is Not All You Need Anymore" (Zhe Chen), §3–§6.
Full equations read from the paper's HTML (v2), not from summaries. Evidence labels:
**[P]** paper says · **[D]** derived from the paper's own numbers/equations ·
**[U]** unknown / not established by the paper.

Scope: perf/quality replication of the **HE (higher-performance Extractor)** sublayer
as a drop-in replacement for the multi-head self-attention sublayer, trained from
scratch. This is architecture research, not a runtime optimization. Gate 0 forbids
interpreting any model results until this spec, the reproducibility protocol, and
the test suite are fixed and committed.

## 1. Placement and surrounding architecture (shared by both arms)

Per-layer dataflow (paper Eqs. 4–21), identical in both arms except sublayer-1:

```
one-hot s [t] → tok-embed [t,d] (W^emb_tok [u,d], scaled by √d)          Eq. 4
              + pos-embed [t,d] (W^emb_pos [l,d], learned, scaled by √d) Eq. 5
→ dropout(p=0.1)                                                    Eq. 7  → X^in_lay1
per layer m (paper: m=18):
  X^in_sub1  = LayerNorm(X^in_lay1)                                  Eq. 9
  sublayer-1 = ATTENTION arm or HE arm (this spec)                   Eqs. 10–17 / 23–25,27–28
  X^in_sub2  = LayerNorm(dropout(X^out_sub1) + X^in_lay1)            Eq. 18
  X^out_sub2 = ReLU(X^in_sub2·W^sub2_1 + b^sub2_1)·W^sub2_2 + b^sub2_2  Eq. 19
  X^out_lay1 = dropout(X^out_sub2) + dropout(X^out_sub1) + X^in_lay1  Eq. 20
head: Ŷ = softmax(LayerNorm(X^out_laym)·W^soft + b^soft)              Eq. 21
```

Fixed choices from the paper **[P]**: learned positional embeddings ("nearly
identical to sinusoidal"), pre-layer norm ("can be trained without the warm-up
stage", citing Xiong 2020), row-wise LayerNorm with learned gain/bias (Eq. 8),
ReLU FFN (Eq. 19) with hidden width c=512 at d=128, dropout p=0.1, untied soft
head with bias (Eq. 21), embedding matrices multiplied by √d (after init —
interpretation, see ledger D-07), loss = next-token cross-entropy averaged over
positions (Eq. 3).

Shapes: t ≤ l = 128 (context window), d = 128, u = 5000, batch 64, m = 18,
60,000 batches (see REPRODUCIBILITY.md §6 for the epoch-accounting caveat).

## 2. HE equations (verbatim from the paper, with shapes)

Let X^in_sub1 ∈ R^{t×d}, positions i = 1..t, and let j index past positions.

1. **(Eq. 27, shared projection)** `X^ext = X^in_sub1 · W^in_ext`,
   W^in_ext ∈ R^{d×d}. **No bias.** (The paper's linear transformations in the
   Extractor equations carry no bias terms; only FFN and head show biases. **[P]**)
2. **(Eq. 28, extraction)** `x_i^out_ext = Σ_{j=1}^{i} x_j^ext ∘ w^ext_{i−j+1}`,
   with gate table w^ext ∈ R^{l×d} — one d-vector per offset k ∈ {1..l}.
   ∘ is the Hadamard product. No softmax, no division, no exponentiation.
3. **(Eq. 24, adjustment)** `X^out_adj = (X^in_sub1 · W^adj) ∘ X^out_ext`,
   W^adj ∈ R^{d×d}, **no bias**.
4. **(Eq. 25, output projection — OPTIONAL)** `X^out_sub1 = X^out_adj · W^out`,
   W^out ∈ R^{d×d}, no bias.

Then the shared wrapper: `X^in_sub2 = LayerNorm(dropout(X^out_sub1) + X^in_lay1)`
(Eq. 18, dropout p=0.1), identical to the attention arm.

## 3. Offset convention (the load-bearing detail)

- Offset index **k = i − j + 1**. Therefore **k = 1 ↔ j = i (the current/newest
  token)** and **k = i ↔ j = 1 (the oldest visible token)**. "The weights are
  assigned in accordance with the reverse order of their input sequence" **[P]**.
- Storage: gate table `w^ext` of shape `[l, d]`, offset axis FIRST; `w^ext[0]` is
  the newest-token gate. Implementation MUST NOT silently flip this axis (see
  test T2).
- Sum bounds j = 1..i: strictly causal, **no padding, no future access, no
  boundary state** **[P]**. At t < l only offsets k = 1..t are exercised;
  w^ext[t..l−1] receive no gradient in that sequence.
- Causality contract: the output row i is a pure function of input rows 1..i
  (bit-exact under perturbation of rows > i in eval mode).

## 4. Adjustment is per-position, not last-token

Eq. 24 is a matrix equation: **each row i is adjusted by its own input vector**
`x_i^in_sub1 · W^adj`. The paper's prose motivates the design with "the latest
input vector x_t contains positional information" **[P]**, but the equation —
which is what we implement — applies W^adj to every position. The prose is
rationale for the mechanism, not a t-only special case. (Ledger D-09; test T9
pins the equation's reading.)

## 5. Output projection decision: ON (**W^out enabled**)

The paper calls W^out "optional, as it is not a critical component" **[P]**.
**Decision: enabled by default** for the headline comparison. Evidence **[D]**
from the paper's own Table 7 (d = l = 128):

- Parameters, HE = 65,536 = W^in_ext (16,384) + gates l·d (16,384) + W^adj
  (16,384) + **W^out (16,384)**. Without W^out, HE has 49,152 = 0.75× the
  32-head attention arm's 65,536 (= 4d²: Q,K,V,O), and the paper's "same
  parameter count" claim would be false.
- Cross-checks: WE's 49,152 = gates + adj + out; ME's 128 = gates only (ME is
  extraction-only per §5.4); SHE's 2,129,920 = l·d² + adj + out. Every
  Table-7 parameter count that includes an "adjustment" also includes W^out.
- Operations: HE's 7,364,608 multiplications reconcile exactly as
  in_ext-t·d·d + extraction t(t+1)/2·d + adjustment Hadamard t·d + adj-t·d·d +
  **out-t·d·d** — i.e. the Table-7 op count also includes Eq. 25. Enabling W^out
  does not break the "fewer ops" claim: 14,647,296 vs 21,710,848 total ops.

So the parameter-matched comparison the paper advertises embeds W^out = ON. We
implement `use_output_projection: bool` (default **True**) and record a separate
OFF arm as an optional diagnostic (paper-literal reading; expect it to be a
strictly weaker comparison for HE).

## 6. Parameter inventory (sublayer-1)

| Arm | Params at d=l=128 | Formula |
|---|---|---|
| HE (W^out ON) | **65,536** | 3d² + l·d |
| 32-head attention | **65,536** | 4d² (Q,K,V,O; no biases in Eqs. 10–17) |
| 1-head attention | 65,536 | same (head count does not change parameter count) |
| WE | 49,152 | l·d + 2d² |
| ME | 128 | l (extraction only) |
| SHE | 2,129,920 | l·d² + 2d² |

Everything outside sublayer-1 is byte-identical between arms (TokEmbed 5000×128,
PosEmbed 128×128, 37 LayerNorms, 18× FFN(128→512→128), head 128×5000 + bias).
Note the Table-7 parameter column counts sublayer-1 only; LayerNorm params are
equal in both arms and excluded from the comparison.

## 7. Initialization policy

- **All weights ~ N(0, 0.01²)**, every bias zero **[P]** ("The biases of the
  models are initialized with zeros, while the weights are initialized randomly
  following a normal distribution with a mean of zero and a standard deviation
  of 0.01"). This covers W^in_ext, the gate table, W^adj, W^out, FFN weights,
  W^soft, and the embedding matrices.
- Embedding matrices are additionally **multiplied by √d** **[P]** (stated for
  W^emb_tok and W^emb_pos; interpretation: applied after sampling — ledger D-07).
- Interpretation **[U→decision]**: LayerNorm gain = 1, bias = 0 (standard; the
  paper's generic sentence does not explicitly cover LayerNorm — ledger D-06).
- All sampling from a single seeded generator per run; seed recorded per arm.

## 8. Reference (oracle) semantics

The unit-test oracle implements Eqs. 27→28→24→25 literally as double loops over
(i, j) in float64/float32 on CPU. Any vectorized/tiled implementation must match
the oracle to tight tolerance (fp32: max abs diff ≤ 1e-5 relative to the oracle
scale; fp64 gradcheck: ≤ 1e-6). No fused-kernel shortcut may be introduced before
the vectorized reference passes.

## 9. Analytic op-count convention (ours, for Milestone 3 reconciliation)

Count per sublayer at a fixed t: matmul = t·d·d multiplications and t·d·(d−1)
additions per [t,d]×[d,d]; extraction = t(t+1)/2·d multiplications and
t(t−1)/2·d additions (the offset-1 term has no adds; sum over j=1..i per row);
Hadamard products count as t·d multiplications. This convention reproduces the
paper's HE and WE rows of Table 7 **exactly** (7,364,608 / 14,647,296 and
5,267,456 / 10,469,376). It does not reproduce the paper's attention rows under
any obvious mapping ([U] — the paper's attention counting is internally loose;
our suite reports both and the delta, per test T14, rather than forcing a match).

## 10. Known ambiguities carried into the deviation ledger

| ID | Item | Status |
|---|---|---|
| D-01 | Corpus set (paper's "top 100" undefined) | see REPRODUCIBILITY.md |
| D-02 | Held-out split (paper evaluates training loss only) | deviation by design |
| D-03 | W^out ON (paper calls it optional) | decided §5, recorded |
| D-05 | "0.457 epochs" vs its own batch/corpus arithmetic | see REPRODUCIBILITY.md §6 |
| D-06 | LayerNorm init interpretation | §7 |
| D-07 | Embedding ×√d applied post-init (interpretation) | §7 |
| D-08 | Hardware/software (RTX 4050 6 GB / PyTorch 2.0 vs ours) | REPRODUCIBILITY.md §2 |
| D-09 | Adjustment per-position (Eq. 24) vs last-token prose (§6 rationale) | §4 |
| D-10 | Loss averaging granularity (per-position mean within batch) | assumption, logged |
| D-11 | Dropout placement exactly per Eqs. 7/18/20 (2 drops per sublayer path) | assumption, logged |

## 11. Gate-0 exit criteria (all must hold before Milestone 1 starts)

1. This spec, REPRODUCIBILITY.md, and TEST_SPEC.md committed.
2. Test suite (T1–T15) implemented and passing deterministically on CPU for HE,
   attention-1h and attention-32h arms; GPU smoke pass recorded.
3. Tokenizer + corpus manifest committed with checksums; corpus decision
   recorded (user sign-off required — see REPRODUCIBILITY.md §3).
4. Deviation ledger contains every entry of §10 with its disposition.
5. No training run has been executed and no result has been interpreted
   (gate 0's explicit prohibition).
