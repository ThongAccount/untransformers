# Milestone 3 — the systems comparison: HE Extractor vs 32-head self-attention

**Status:** every table below is rendered straight from `results/systems.json`
by `tools/render_results_tables.py`, so a number in this document is by
construction the number in the artefact. The measurement protocol was fixed in
advance in `docs/milestone-3/PROTOCOL.md` §4.3 (requirements S1–S10).

**What this comparison is.** A measurement of *this implementation*, on one
accelerator, under one pinned numerics configuration, with a repeated A/A null
that says how large a difference the clock can resolve.

**What it is not.** It is not a measurement of the HE *mechanism*. HE here is
an eager Python loop over the 128 context offsets (`hextractor/layers.py` — the
module docstring says there are no fused kernels), while the attention arms
may dispatch a fused SDPA kernel. A wall-clock difference between them
therefore measures **implementation maturity under these conditions**, and
every wall-clock row below carries that caveat. The algorithmic comparison —
the one in which both sides are treated identically — lives in the analytic
blocks (op counts, per-token state) and in the sublayer-1 scaling sweep, not
in the wall-clock rows.

---

## 1. The environment, the configuration and the numerics

<!-- TABLES:systems -->

## 2. How to read these rows

* **Every row carries its units, its denominators, its trial count and its
  quality level.** A row whose quality level says "architecture at
  initialisation" is not tied to any quality number and must never be paired
  with one; a row whose quality level names a checkpoint carries that
  checkpoint's held-out loss.
* **The arms are measured twice, in a palindromic (ABBA) order.** The A/A
  null — the same arm at the start and at the end of the order — is the
  run-order drift floor. An arm-to-arm difference smaller than the floor is
  not resolvable against clock/thermal drift, and is reported as such rather
  than as a result.
* **The floor is printed next to every arm-to-arm difference.** The rule was
  fixed before the systems numbers existed: the floor for a block and a
  context is the **max−min spread across the A/A null's repetitions**, taken
  as the **largest** such spread over the arms at that context; a difference
  is **resolvable** iff its absolute value exceeds that floor, and below the
  floor it is **not resolvable, whatever its sign**. Exceeding the floor is
  necessary, not sufficient, for a claim — the kernel caveat still travels
  with the row.
* **Context lengths are kept apart.** The model-level rows live at or below
  the model's context cap (`l = 128`); the sublayer-1 sweep goes far past it
  (128 … 8192) and measures **one sublayer**, not the model.
* **The training step and the incremental decode are different measurements.**
  The training step is a full forward-and-backward at the paper's shape (batch
  64, `l = 128`, 18 layers); the incremental decoding rows time the same model
  one token at a time with a KV cache, and the HE arm carries a bounded rolling
  state where the attention arms carry a cache that grows with the context. A
  training-step number and an incremental-decode number are never divided by,
  or subtracted from, each other.
* **The traffic blocks are analytic.** They are formulas with their
  substitutions printed, not measurements, and they cannot show a difference
  between the arms where the arms carry the same parameter count by
  construction.
* **The decode rows separate the first generated token from the steady
  state.** The first token carries prefill-adjacent effects (the first cache
  append) and is pooled with the steady state only in the row that says so.

## 3. The claims this comparison does and does not support

**Supported, with the conditions attached:**

* the training-step, prefill and decode latencies of these three
  implementations on this accelerator, at this shape, with these numerics
  settings, with this trial count, and with the run-order drift floor that
  bounds their resolution;
* the per-token state each arm carries, as an analytic formula (HE flat in the
  context length; attention's KV cache linear in it);
* the scaling of the sublayer-1 cost with the context length for all three
  arms, measured the same way.

**Not supported by this harness:**

* "HE is faster" or "HE uses less memory" as a statement about the HE
  mechanism. The wall-clock rows compare an eager Python loop against a
  kernel that may be fused; the kernel caveat travels with every such row.
* any speed claim made against an unoptimised baseline: the optimized
  `attention-32h-sdpa` arm is timed in every block, and it is the arm the
  comparison is made against. **Gate G6 requires the SDPA rows in every
  block; if any block lacks them, this sentence is false and the gate fails.**
* any decode-latency claim that ignores the `torch.cat`-growing KV cache of
  the attention arms (an O(T) copy per token; a preallocated cache would not
  pay it — declared limitation L-03 in `docs/milestone-3/SCOPE.md`).
* any per-arm CPU memory comparison: the CPU rows are **process-level**
  high-water marks. On CUDA the counters are reset per arm, so those rows are
  per-arm.
* any pairing of a performance number with a quality number that the row does
  not itself name.

## 4. The measurements

The tables in §1 are the measurements. Read them with §2.

## 5. Reproducing this

```bash
python experiments/run_systems.py \
    --arms HE attention-32h attention-32h-sdpa \
    --systems-arms HE attention-32h attention-32h-sdpa \
    --device cuda --out results/systems.json \
    --d 128 --l 128 --layers 18 --ffn 512 --vocab 5000 --batch 64 --heads 32 \
    --trials 10 --warmup 2 --gen-tokens 16 --decode-trials 5 \
    --checkpoint-dir results/checkpoints
```

The gates are `G6` (the systems artefact is structurally complete) and `G7`
(the paired quality checkpoints are carried into it); their evidence is in
`docs/milestone-3/EVIDENCE.md`.
