# PROTOCOL REVIEW — Milestone 2/3 measurement protocol (Job A)

**Author:** Researcher (reviewer, not builder) · **Date:** 2026-10-23 UTC
**Basis:** branch `research/milestone-1-extractor` @ `1693df5` · **Paper:** [arXiv:2308.07661v2 — *Attention Is Not All You Need Anymore*](https://arxiv.org/abs/2308.07661)

**Status: pre-registration.** This memo is written *before* any quality or systems
number exists (no `results/` artifact was in the repo at `1693df5`). It fixes what
may be measured, how it must be summarised, and what may not be claimed. Job B
(review of the results) will check the write-up against this document.

**Gate discipline:** no result interpretation here. This memo does not say which
arm is better; it says what "better" would have to mean, and what the current
harness would and would not be able to establish.

---

## 1. What I checked (audit trail)

Repo facts below were read from the source at `1693df5`; every number carries the
command or the line it came from.

| area | files read in full |
|---|---|
| gates / plan | `GATES.md`, `PLAN.md`, `docs/milestone-0/REPRODUCIBILITY.md`, `docs/milestone-1/DEVIATIONS.md`, `docs/milestone-2/CORPUS.md`, `docs/milestone-3/EVIDENCE.md` |
| harness | `hextractor/harness.py`, `hextractor/init.py`, `hextractor/model.py`, `hextractor/corpus.py`, `hextractor/layers.py`, `hextractor/incremental.py` |
| runners / checkers | `experiments/run_quality.py`, `experiments/run_systems.py`, `tools/verify/check_quality_results.py`, `tools/verify/check_systems_results.py`, `tools/build_tokenizer.py` |
| data | `data/manifest.json`, `data/tokens.npy` (9,514,186 tokens, uint16) |
| tests | `tests/test_t07_init.py`, `tests/test_t14_data_pipeline.py`, `tests/test_t15_gpu_smoke.py`, `tests/test_m3_incremental.py`, `tests/test_m3_sdpa_baseline.py` |

Probes I ran myself (local host, `/home/node/.venv/bin/python`, CPU):

* **RNG pairing across arms.** Built both arms under the same seed, then reset the
  global seed immediately before each forward: `torch.rand(6)` after the HE forward
  and after the attention forward are **bit-identical** → the dropout RNG stream is
  identical across arms. (My first probe said "different"; that probe was wrong —
  it drew `torch.randint` between the two forwards. The corrected probe is the one
  reported here.) Dropout calls per forward: **7 in both arms**.
* **Post-init global RNG state** is identical across arms (sublayer parameters are
  allocated with `torch.empty`, so construction consumes no RNG).
* **Split composition** recomputed from `data/manifest.json` + `data/tokens.npy`
  using the 100 EOS markers (`<|endoftext|>` = id 0) as book boundaries.
* **Paper facts** (Table 1 hyperparameters, 8.4M-token corpus, evaluation metric)
  read from the paper's own HTML this session: <https://arxiv.org/html/2308.07661v2>.

Existing suite state at `f207b7a`: `91 passed, 1 skipped` (the skip is the CUDA-only
T15). The new SDPA-baseline suite landed in `1693df5`.

---

## 2. Q1 — What makes the held-out split defensible here

The paper has **no held-out split**: it evaluates with "training cost (i.e. the
average training loss over a batch) … since training cost equals perplexity in this
task" (paper §4, HTML v2). Our held-out loss is therefore a *substitute metric*, not
a reproduction of the paper's metric. Two consequences, both binding on the write-up:

1. **Our absolute loss/perplexity numbers are not comparable to the paper's.**
   Different books, different tokenizer fit, different token budget, different
   metric. Only *within-protocol* arm-vs-arm comparisons are meaningful.
2. The paper's metric being train loss means our held-out number is a *stricter*
   proxy than anything the paper reports. That is an improvement, not a discrepancy
   to be explained away — but it must be stated, not used to imply we "beat" the
   paper.

### 2.1 What the split actually is (measured, not assumed)

Cut = `int(total * 0.95)` on the **frozen token array** (`hextractor/corpus.py:128-140`):

| quantity | value |
|---|---|
| total tokens | 9,514,186 |
| train region (first 95%) | 9,038,476 tokens |
| held-out region (last 5%) | 475,710 tokens |
| books fully in train | **96** |
| book straddling the cut | **ebook 3499** (index 96): 87,783 tokens in train, 68,987 in held-out |
| books fully held out | **3**: ebook 3536 (110,927 tok), 3795 (129,118 tok), 3796 (166,678 tok) |
| share of held-out tokens from fully unseen books | **85.5%** (406,723 / 475,710) |
| share from the tail of a *seen* book | 14.5% (68,987) |
| possible window start positions in the held-out region | 475,581 (`region.size - l - 1`) |
| windows drawn per eval point | 8 batches × 64 = **512** (511 distinct starts) |
| target tokens per eval point | 8 × 64 × 128 = **65,536** (13.8% of the held-out region) |

### 2.2 Verdict: defensible, with three disclosures and one addition

**Defensible as a reporting-only split for a within-corpus, same-protocol arm
comparison**, because: the held-out tokens are never trained on and never used for
selection; the stream is seeded by `SHA-256(train_seed | split | corpus_id |
tokenizer_sha256)` (`corpus.py:143-145`), so the *same* 512 windows are used at every
eval point, for every arm and every seed; and windows are drawn strictly inside a
region, so no training window can contain held-out tokens and vice versa.

It is **not** defensible as a generalization estimate, for three reasons that must be
disclosed rather than smoothed over:

* **D-A — the cut is mid-book.** The split boundary lands inside book 3499, so the
  held-out metric mixes *unseen-book* behaviour (85.5% of tokens) with
  *seen-book/unseen-text* behaviour (14.5%). A model can score well on the tail of a
  book it saw the beginning of. The write-up must report the two components
  separately (see §2.3), never as one "held-out loss".
* **D-B — n_books = 3.** The unseen-book evidence rests on exactly three books.
  Book-level variance is therefore unmeasured, and no statement of the form "HE
  generalizes better to new books" is supported at this n. Say "on three unseen
  books", not "on unseen books".
* **D-C — the tokenizer saw the held-out text.** `tools/build_tokenizer.py` iterates
  *all* manifest books to fit the BPE merges (manifest: `tokenizer.trained_on =
  "pinned corpus snapshot (manifest books, manifest order)"`), i.e. the held-out
  region influenced the vocabulary. This is a mild, unavoidable leak under the
  "one frozen token array" design. It is **not in the deviation ledger** — it needs
  an entry (proposed **D-22**) and a sentence in the write-up. It cannot be fixed
  without re-tokenizing and re-running, i.e. it is a declared limitation
  (§8.D).

### 2.3 The addition: a book-aligned sensitivity evaluation (evaluation only)

Cheapest high-value fix, and it does **not** touch the runs in flight: the sweep
saves checkpoints (`SAVE_CHECKPOINTS=1`), so every checkpoint can be re-scored on a
**book-aligned held-out set** without retraining.

* **Book-aligned set:** the three fully unseen books only (ebook 3536, 3795, 3796;
  406,723 tokens), evaluated *exhaustively* (stride-1 sweep, ~3,177 windows of 128
  target tokens per book) rather than by sampled windows — this removes window-sampling
  noise from the sensitivity metric entirely.
* **Pre-registration:** the **primary endpoint stays the run's own metric** (§3).
  The book-aligned number is a *sensitivity analysis*, reported alongside, and is
  explicitly labelled post-hoc. It may confirm or contradict the primary; it may not
  replace it.
* Interpretation rule fixed in advance: if the arm ordering on the primary metric
  and on the book-aligned metric disagree, the write-up reports the disagreement as
  the finding (it would mean the result is driven by the mid-book tail), and does
  not pick the favourable one.

---

## 3. Q2 — Which dispersion / uncertainty statistics to report (3 seeds)

### 3.1 The harness's current dispersion is not the statistic the comparison needs

`hextractor/harness.py:160-176` (`dispersion()`) reports, per arm and per eval
point: `n`, `mean`, and `std` computed with denominator **n** (population SD,
`ddof=0`). Two problems:

* For n = 3 the population SD understates the sample SD by a factor √(3/2) = **1.225**
  — it is the wrong estimator for a t-based interval.
* It is a **per-arm** dispersion across seeds. The comparison we need is a **paired**
  statistic: the arms share the initialisation seed, the batch order, the dropout
  stream and the evaluation windows, so the paired difference is far better
  determined than either arm's marginal spread.

The write-up must therefore recompute the statistics from the per-seed values; the
harness's `dispersion` block is a descriptive summary, not the inference.

### 3.2 The pre-registered report (exactly what to print)

Let `L_arm(s)` be the held-out loss of arm ∈ {HE, attention-32h} at seed `s ∈ {1,2,3}`.

**Primary endpoint (one per arm/seed):** held-out loss at the **final step** (2000).
**Secondary endpoint:** mean of the **last four eval points** (1250, 1500, 1750, 2000).
**Curves:** the full per-seed eval curve for both arms (all 8 eval points), so the
reader can see whether the final point is a plateau or a spike.

For each endpoint, print:

| statistic | definition |
|---|---|
| per-seed values | `L_HE(1..3)`, `L_attn(1..3)` — no hiding individual seeds |
| paired differences | `d_s = L_HE(s) − L_attn(s)` (sign convention stated explicitly) |
| mean difference | `d̄ = (d_1+d_2+d_3)/3` |
| sample SD | `s_d` with **`ddof=1`** |
| standard error | `s_d / √3` |
| **95% CI** | `d̄ ± 2.484 · s_d` (t₀.₉₇₅,₂ = 4.3027, √3 = 1.7321) |
| **MDD at 80% power** | `3.097 · s_d` ((t₀.₉₇₅,₂ + t₀.₈₀,₂)/√3, t₀.₈₀,₂ = 1.0607) |
| per-arm dispersion | mean and sample SD (ddof=1) per arm at the endpoint |
| perplexity | `ppl = exp(loss)` per arm; arm ratio `ppl_HE/ppl_attn = exp(d̄)`, with the CI endpoints mapped through `exp()` (monotone, so the interval stays valid) |
| train loss | final train loss and train curve per seed, so underfitting/divergence is visible |

**Decision rule, fixed now:**

* The comparison is **resolved** iff the 95% CI for `d̄` excludes 0.
* If it does not, the write-up says **"not resolved at n = 3"**, prints the MDD
  (`3.097 · s_d`) as the smallest effect this protocol could have detected, and
  states how many seeds would have been needed. "Not resolved" is a legitimate
  result and must be reported as one, not converted into a favourable trend claim.
* No p-value in the headline. If a p-value is printed at all, it is the paired
  t-test on `d_s` with df = 2, shown *next to* the CI, never instead of it.

### 3.3 Eval-set noise: separate it from training noise

The held-out stream is seeded independently of the model seed, so **all arms, seeds
and eval points are scored on the identical 512 windows**. Two consequences:

* **Eval-set noise cancels in the paired difference** to a large extent (same
  windows, same targets). The CI in §3.2 therefore describes *training-run* noise
  (initialisation + dropout), not the uncertainty of "the model quality on this
  corpus".
* It is **not zero** for the absolute per-arm loss: 65,536 target tokens drawn from
  475,581 possible starts is a *sample* of the held-out region. Report the
  **eval-set bootstrap SE** separately: resample the 512 windows with replacement
  (1,000 resamples), recompute each arm's mean loss and the paired difference, and
  print the bootstrap SE of each. State explicitly that this SE is **not** included
  in `2.484 · s_d`, and that the two are not simply additive.

### 3.4 The unavoidable caveat: all three seeds share one batch sequence

`corpus_stream_factory` (`harness.py:195-205`) seeds the stream from `train_seed`
only; `run_multi_seed_corpus` (`harness.py:210-229`) creates the stream **once** and
reuses it for every seed, and `train_one_seed` restarts it (`train_it = train_stream()`).
So **every seed sees the identical sequence of batches** — only initialisation and
dropout vary.

Consequence: `s_d` is a **lower bound** on full run-to-run variance (it omits
data-order variance entirely), and the MDD derived from it is optimistic. This must
be stated in the write-up's limitations. It cannot be fixed inside the current
sweep without breaking provenance (changing the data order changes the run), so it
goes into the scope decision as a declared limitation, not a silent omission.

---

## 4. Q3 — What the systems comparison must measure to be meaningful

### 4.1 What is being compared, and the one thing that must change

`PLAN.md` (Milestone 3) requires "both an uncomplicated reference and a
responsibly optimized attention baseline; note kernel/fusion differences", and the
gate is: *"Claim a speed or memory improvement only when repeated measurements
establish it against the optimized baseline without hiding a quality regression."*

**The optimized baseline is not timed.** `experiments/run_systems.py` defaults
`--arms` and `--systems-arms` to `HE attention-32h` (lines 355-357), and
`measure_sublayer_scaling` hardcodes `["HE", "attention-32h"]` (line 426). The
`attention-32h-sdpa` arm exists in `hextractor/model.py:16-18` (fused
`scaled_dot_product_attention`) and — as of `1693df5` — is proven to be the *same
function* as the reference arm (`tests/test_m3_sdpa_baseline.py`: bit-identical init,
matching logits, causality). It is currently only present in the **analytic**
`weight_traffic` block.

Any speed or memory claim from the systems leg as configured would therefore be a
claim against an *unfused* baseline and would fail the Milestone-3 gate. This is a
runner-configuration change on the **systems leg only**; it does not touch the
quality sweep, so it costs no provenance.

### 4.2 Kernel and implementation asymmetry (the biggest interpretive trap)

* HE is an **eager Python offset loop**: `for k in range(kmax)` with slice
  arithmetic (`hextractor/layers.py:51-58`), i.e. 128 iterations of elementwise
  work per layer per forward — the module docstring says so explicitly ("No fused
  kernels here … the vectorized reference form itself").
* The attention baseline can dispatch a **fused kernel** (`attention-32h-sdpa`).
* The incremental path repeats the same loop per token (`hextractor/incremental.py:105`),
  and the attention KV cache grows by `torch.cat` per token
  (`incremental.py:141-142`) → an O(T) copy per token that a preallocated cache would
  not pay.

Therefore **a wall-clock "HE vs attention" number from this harness measures eager
Python vs fused kernel, not the paper's algorithmic claim.** Two honest ways forward:

* **(a)** implement HE at comparable engineering quality (batched/fused). Out of
  budget now; declared as future work (§8.D).
* **(b)** frame every systems statement as *"this implementation, under these
  conditions"*; put the *algorithmic* claim on the measurements where both sides
  are treated identically — the analytic op counts (T13) and the sublayer-1 scaling
  sweep — and attach the kernel caveat to every wall-clock row.

Whatever is done, the write-up must not present "HE is faster" as a property of the
HE mechanism on the strength of this harness alone.

### 4.3 The measurement list the systems leg must satisfy

| # | requirement | current state | action |
|---|---|---|---|
| S1 | Time the optimized baseline (`attention-32h-sdpa`) alongside the reference | not timed | add to `--arms`, `--systems-arms`, the scaling sweep and `check_systems_results.py` |
| S2 | Quantify run-order effects | arms timed **sequentially** (HE first, attention second), no null | add an **A/A null** (same arm twice) and interleave (ABBA) the order; report the null's spread |
| S3 | Enough trials | `--trials 5`, `--warmup 2`; decode uses `max(1, trials//2) = 2` trials × 16 tokens = **32 samples** | `--trials ≥ 10`; decode trials ≥ 5; publish median, p10/p90, min and **n** with every row |
| S4 | Separate first-token from steady-state decode | the first generated token (right after the prompt) is pooled with the rest | report the first token separately (it carries prefill-adjacent effects) and steady-state separately |
| S5 | Record the numerics settings | `env_manifest()` records torch/CUDA/cuDNN/GPU but **not** TF32 / `float32_matmul_precision` / cuDNN flags | set them explicitly (e.g. `allow_tf32=False`, `set_float32_matmul_precision("highest")`) and record them; otherwise a T4 number is not reproducible across runs |
| S6 | Per-arm memory | CUDA path resets peak counters per arm, but the **CPU** path returns `ru_maxrss`, a process-lifetime high-water mark → the second arm's row inherits the first arm's peak | label CPU memory rows as **process-level**, or measure each arm in its own process |
| S7 | Traffic numbers must say what they are | `sublayer_scaling.state_bytes_per_layer` is a **formula** (`(l-1)·d·4` HE, `2·T·d·4` attention), not a measurement; `weight_traffic`'s note says "weights are streamed once per step" while `training_step_parameter_bytes_read = 3·param_bytes` — the note contradicts the number | label the rows **analytic**, print the formula beside each, fix the note (training reads 3× param bytes: fwd + dgrad + wgrad; decode reads 1×), and move the per-token state/KV traffic into the traffic block |
| S8 | Tie performance to a stated quality level | the timed models are **freshly initialised** (`apply_init_policy` only, untrained); `paired_quality` merely copies `steps`/`final_held_out_loss` out of the checkpoint metadata — the weights are never loaded | load the saved checkpoints into the timed models and print the checkpoint's held-out loss next to each row, **or** state explicitly that the systems numbers describe the architecture at initialisation and do not pair them with quality numbers |
| S9 | Keep the two context regimes apart | the model's positional table caps context at `l = 128`; beyond that only the sublayer-1 sweep (128…8192) exists | keep model-level rows (≤128) and sublayer rows separate and labelled; note that HE's state is **flat** past `l-1` while attention's grows linearly — that is the fair algorithmic comparison, provided both are measured the same way |
| S10 | Units and denominators with every row | training throughput = `batch·l` tokens/step; decode = `1/median per-token latency` | print units, warmup/trial counts and the exact context length next to each number |

### 4.4 Misleading-if-reported-as-is (the short list MLForge asked for)

1. HE-vs-attention wall clock from this harness (eager loop vs fused kernel).
2. Attention decode latency with a `torch.cat`-growing KV cache.
3. "Weight traffic saved" read off the analytic block (and its contradictory note).
4. CPU "peak memory" per arm (process-level RSS).
5. A speed number attached to a quality level it was never measured at (untrained models).
6. A single-order timing difference with no A/A null on a shared T4 (clock/thermal drift
   is inside the number).
7. A T4 number without the TF32/precision settings (not reproducible).
8. A decode latency computed from 2 trials × 16 tokens with the first token pooled in.

---

## 5. Q4 — Where the harness could quietly produce a misleading number

Ordered by severity. "Fix class": **A** = disclose now (text only), **B** = post-run,
evaluation-only (checkpoints are saved), **C** = must fix before the systems numbers
are quoted (systems leg only; no effect on quality provenance), **D** = out of budget →
declared limitation.

| # | issue | evidence | why it misleads | fix |
|---|---|---|---|---|
| F1 | **G3/G4 pairing gates cannot fail.** `run_quality.py:136-137` writes `same_batches_same_order: True` and `paired_init_seeds: True` as **hardcoded literals**; `check_quality_results.py:63-66` only asserts those literals are `True` | source | the gate "certifies" a claim that was never measured. The pairing is real (I verified it), but the gate cannot detect its absence | **C**: derive the flags from the run — hash the first N batch windows per arm and compare, and compare the shared parameters bitwise across arms — then write the evidence into the results JSON |
| F2 | **The optimized attention baseline is never timed** | `run_systems.py:355-357, 426` | any speed/memory claim fails the Milestone-3 gate ("against the optimized baseline") | **C** (S1) |
| F3 | **Eager HE vs fused attention** | `layers.py:51-58`; `model.py:16-18` | wall-clock measures implementation maturity, not the mechanism | **C** (S2) / **D** (fused HE) |
| F4 | **Systems numbers come from untrained models**, while `paired_quality` implies a quality tie | `run_systems.py:124-127, 431-447` | performance and quality appear linked but are measured on different models | **C** (S8) |
| F5 | **Attention KV cache grows by `torch.cat` per token** | `incremental.py:141-142` | O(T) copy per token inflates attention decode latency vs a preallocated cache | **C** (disclose) / **D** (preallocate) |
| F6 | **CPU peak memory is process-level** | `run_systems.py:70-84` | the second arm inherits the first arm's high-water mark | **C** (S6) |
| F7 | **Analytic traffic presented next to measurements, with a contradictory note** | `run_systems.py:277-279, 309-324` | "weights are streamed once per step" vs `3 × param_bytes`; state bytes are formulas | **C** (S7) |
| F8 | **Too few trials, no order null** | `run_systems.py:367-368, 418-422` | clock drift on a shared T4 is indistinguishable from an arm effect | **C** (S2, S3) |
| F9 | **TF32 / matmul precision unrecorded** | `run_systems.py:330-347` | the number is not reproducible across runs | **C** (S5) |
| F10 | **Dispersion uses population SD (ddof=0) and is per-arm, not paired** | `harness.py:160-176` | understates spread by 18% at n=3, and never produces the paired statistic the comparison needs | **A/B**: recompute in the write-up (§3); the harness field can stay but must be labelled |
| F11 | **All seeds share one batch sequence** | `harness.py:195-229`, `corpus.py:143-166` | across-seed variance = init + dropout only → a **lower bound** on run-to-run variance; MDD optimistic | **A** (declare) / **D** (data-order sweep) |
| F12 | **Tokenizer trained on the full corpus, including the held-out region** | `tools/build_tokenizer.py`; manifest `tokenizer.trained_on` | held-out text influenced the BPE merges → mild leak into the held-out metric; not in the ledger | **A** (D-22 + disclosure) / **D** (re-tokenize) |
| F13 | **Doc/code drift on weight decay** | `docs/milestone-0/REPRODUCIBILITY.md` §5 says 0.01; `harness.py:40` uses `1e-3` (= the paper's γ = 0.001, D-15) | same value in both arms → the comparison is unaffected, but the doc misstates the config | **A**: correct the doc |
| F14 | **"epochs" are nominal** | windows are sampled **with replacement** (`corpus.py:162-165`); `run_quality.py` emits `epochs_of_our_corpus` | "1.81 epochs" reads like 1.81 passes over the data; it is 1.81 × the region size in tokens, with repeats | **A**: rename/label *nominal epochs* everywhere |
| F15 | **The CPU pilot is not a result** | 50 steps = 409,600 tokens = **0.045 nominal epochs** | a reduced-budget pilot read as a quality outcome | **A**: label it a protocol smoke test, everywhere it appears |
| F16 | **Budget framing** | sweep = 2,000 × 8,192 = **16,384,000 tokens** = 1.813 nominal epochs of our train region; the paper's 60,000 batches × 8,192 = **491,520,000 tokens** on an 8.4M-token corpus = **58.5 epochs** (the paper's own "0.457 epochs" label is not reproducible under any standard reading — D-19) | our sweep is **3.333%** of the paper's token budget | **A**: the token budget goes in the headline of every result, and the ratio to the paper is stated explicitly |
| F17 | **Paper metric mismatch** | paper: "training cost (average training loss over a batch)"; ours: held-out loss | an absolute loss comparison to the paper is meaningless | **A**: state it in methods; never compare absolute numbers |

---

## 6. Q5 — Corpus decision: **blessed, with conditions**

**Verdict: bless the pinned Gutenberg "Children's Literature" bookshelf-20 snapshot
(first 100 by ascending ebook ID) for this round.** Reasons: the paper's own corpus
is unrecoverable ("top 100 children's books" has no list, ranking or release date),
so *any* substitute is a deviation; this one is deterministic, permissively licensed,
pinned by SHA-256 at both the listing and the file level, in the same domain
(free English children's books) and the same language as the paper, and it is
reproducible from the manifest alone. Nothing in the paper's protocol depends on the
particular books, only on the domain and on having 100 books tokenized with a
5000-vocab BPE — all of which this preserves.

**Conditions (all binding on the write-up):**

* **C-1 — the selection is not random and is age-skewed.** "First 100 by ascending
  ebook ID" is a deterministic but arbitrary rule; low IDs are older Gutenberg
  records, so the corpus skews toward 19th-century children's books. The paper's
  books are unknown and probably more modern. This is a representativeness
  limitation of the substitute, not of the protocol — but it must be stated, not
  implied away.
* **C-2 — disclose D-22 (tokenizer leak), §5 F12.**
* **C-3 — the unseen-book evidence is n = 3 books** (§2.2 D-B).
* **C-4 — never compare absolute loss/perplexity to the paper's numbers** (§2, F17).

**Would a different substitute be better-justified?** On the axis that matters for a
*fixed-token-budget* comparison, yes, and it is worth recording for the next round:
taking the **full 317-book shelf** would give more unique books and fewer repeated
passes at the same step count, and a **book-aligned held-out set** (e.g. every 10th
book held out) would give ~30 unseen books instead of 3 — which is the single change
that would most improve the quality claim. Both require re-running the sweep, i.e.
breaking the provenance of the run in flight. So: **recommended for the next round,
declined for this one** — and the decision belongs in `docs/milestone-3/SCOPE.md`
(Job B), not silently here.

---

## 7. Pre-registered write-up rules (locked before the numbers land)

1. Every headline number carries: arm, seed(s), step count, **tokens processed**,
   **nominal epochs** of our train region, corpus_id + tokenizer SHA-256, held-out
   definition, eval window count and target tokens, device/torch, trials and warmup.
2. The quality comparison is reported as **paired per-seed differences** with the
   CI and the MDD (§3.2) — never as two marginal means compared by eye.
3. Forbidden claims, in writing:
   * "HE is better" / "HE generalizes better" without a named endpoint, a CI, and
     the n=3 / n_books=3 caveat;
   * any comparison of absolute loss or perplexity to the paper's numbers;
   * "significant" without the CI;
   * "faster" / "less memory" without the optimized SDPA baseline, the A/A null,
     the trial counts, the precision settings and the kernel caveat;
   * any use of the CPU pilot as a quality result;
   * the harness's `std` (ddof=0) quoted as the dispersion;
   * the systems numbers presented as a property of the HE *mechanism*.
4. Negative results are results: "not resolved at n = 3" with the MDD is a complete
   and acceptable outcome, and must be reported as one.
5. The write-up cites the paper's own numbers from the paper
   (<https://arxiv.org/html/2308.07661v2>, Table 1) and labels every difference
   between the paper's protocol and ours.

---

## 8. Action list for MLForge

**A — disclose now (text only; no rerun, no protocol change)**
1. Add **D-22** to `docs/milestone-1/DEVIATIONS.md`: tokenizer fitted on the full
   corpus including the held-out region (mild leak; not fixable without re-running).
2. Correct `docs/milestone-0/REPRODUCIBILITY.md` §5: weight decay **0.001** (paper
   γ), not 0.01 (code: `harness.py:40`; D-15).
3. Label every epoch figure **nominal**, and print the token budget (16,384,000
   tokens; 3.333% of the paper's 491,520,000) in the results header.
4. Label the CPU pilot (50 steps, 0.045 nominal epochs) a **protocol smoke test** in
   every place it appears.
5. In the write-up's methods: state the paper's metric (training cost) vs ours
   (held-out loss), the split's mid-book cut, the 3-unseen-books limit, the
   age-skewed selection rule, and the data-order caveat (F11).

**B — post-run, evaluation-only (checkpoints are saved; no retraining)**
6. **Book-aligned sensitivity eval** over ebook 3536/3795/3796, exhaustive stride-1
   sweep, for all six checkpoints; report alongside the primary endpoint (§2.3).
7. **Eval-set bootstrap** (1,000 resamples over the 512 windows) → bootstrap SE of
   each arm's mean loss and of the paired difference; reported separately from the
   training-noise CI.
8. Recompute the paired statistics with `ddof=1`: `d̄`, `s_d`, CI `±2.484·s_d`,
   MDD `3.097·s_d`, per-seed values, ppl via `exp(·)`.

**C — must fix before any systems number is quoted (systems leg only; the quality
sweep's provenance is untouched)**
9. Time `attention-32h-sdpa` in all systems blocks; extend
   `check_systems_results.py` to require its rows (S1).
10. A/A null + interleaved arm order (S2); `--trials ≥ 10`, decode trials ≥ 5,
    publish median/p10/p90/min/n (S3); separate first-token from steady-state
    decode (S4).
11. Set and record TF32 / `float32_matmul_precision` / cuDNN flags; add the git
    commit and the arm list to the environment manifest (S5).
12. Label CPU memory as process-level or isolate arms per process (S6).
13. Traffic rows: label **analytic**, print the formula, fix the "streamed once"
    note, and add per-token state/KV traffic to the traffic block (S7).
14. Load the saved checkpoints into the timed models (or declare "architecture at
    initialisation") so performance numbers carry a stated quality level (S8).
15. Make G3/G4 certify the pairing claims they currently assert: derive
    `same_batches_same_order` / `paired_init_seeds` from the run (batch-window hash
    per arm; bitwise shared-parameter comparison) and record the evidence (F1).

**D — out of budget for this round → declare as limitations in `SCOPE.md`**
16. Data-order seed sweep (needs retraining) — F11.
17. Fused/optimised HE implementation — F3.
18. Preallocated attention KV cache — F5.
19. Re-tokenizer trained on the train region only — F12.
20. Full 317-book corpus and/or a book-aligned held-out set (n ≈ 30 unseen books) —
    §6.

---

## 9. What I could not verify (open gaps for Job B)

* **The paper's exact corpus** (book list, which release, ranking) — not
  reconstructable; §6 records the consequence.
* **The paper's "0.457 epochs"** for 60,000 batches over an 8.4M-token corpus. The
  arithmetic under any standard reading gives 58.5 epochs (491.52M / 8.4M). I could
  not reconstruct the paper's intended meaning; our budget comparisons are therefore
  made in **tokens only** (D-19).
* **Whether the paper reports the last-batch training cost or an average over
  training** — the HTML states the metric is "the average training loss over a
  batch" but not the averaging window. If the write-up compares against the paper's
  absolute values at all, this needs resolving from the PDF first (and §5 F17 says
  it should not).
* **The quality and systems numbers themselves** — none existed at `1693df5`; that
  is Job B.
* **The GPU sweep's in-session conditions beyond what the driver records** (clock
  throttling, exact GPU utilisation) — to be checked against the session log when
  the artefacts land.
