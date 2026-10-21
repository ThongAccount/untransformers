# Plan: Test attention-free Extractors before adopting them

## First decision

Do **not** replace attention in the product runtime as the first implementation. First independently test the most consequential, cheapest claim in “Attention Is Not All You Need Anymore” (arXiv:2308.07661v2): that the HE (higher-performance Extractor) sublayer can beat 32-head self-attention at the same parameter count and fewer arithmetic operations.

This is a research prototype, trained from scratch with an Extractor architecture. It is not a drop-in optimization for existing Transformer checkpoints: replacing attention changes the model architecture and weights, so it conflicts with the current UnTransformers premise that the model remains unchanged. Keep the experiment isolated and make adoption a separate, explicit project-scope decision based on results.

## Why this first

- The paper defines four causal, softmax-free Extractors: SHE, HE, WE, and ME. HE is the best first test: its shared projection plus per-offset channel gates is intended to retain expressiveness while matching 32-head attention’s parameter count.
- At the paper’s `d=128`, `l=128` setting, its analytic accounting reports HE at 14,647,296 operations versus 21,710,848 for 32-head attention, with 65,536 parameters for each. These are analytic counts, not measured speedups.
- The evidence is weak: one small model, one 8.4M-token children’s-books corpus, training-loss comparisons, one run per configuration, no confidence intervals, and no wall-clock benchmarks. Replication—not adoption—is the immediate goal.
- The vault analysis is the source for this plan: `~/Agent Memories/journal/2026-10-21-attention-is-not-all-you-need-anymore.md`.

## Principles

1. **Separate architecture research from execution optimization.** No results from this experiment count as proof that an unchanged checkpoint runs faster.
2. **Make comparisons controlled and reproducible.** Hold data order, tokenization, optimizer, initialization policy, training budget, batch shapes, and evaluation procedure constant between attention and Extractor runs. Record unavoidable deviations from the paper.
3. **Measure quality and systems behavior independently.** Report held-out loss/perplexity, training loss, parameter count, analytic operation count, and measured training/inference latency and memory. Do not use operation counts as a performance proxy.
4. **Use multiple seeds and uncertainty.** Report per-seed results and dispersion; do not promote a single-run win.
5. **Preserve causal correctness.** Test prefix-by-prefix behavior and verify no output at position `t` depends on future tokens.

## Milestones

### 0. Reproducibility and specification

- Translate the paper’s HE equations into an implementation specification, including tensor shapes, offset convention, causal boundary behavior, adjustment/output projections, initialization, and whether the optional output projection is enabled.
- Identify and pin the tokenizer and data source. The paper reports a self-built Gutenberg corpus but no verified release or official implementation; do not claim an exact reproduction if the corpus or preprocessing cannot be reconstructed.
- Document hardware/software, seeds, batch ordering, training schedule, and all deviations before comparing results.
- Define a small, deterministic unit-test suite: equation-level reference checks, shape/dtype/device checks, causality, and gradient checks.

**Gate:** Do not interpret model results until the implementation and data pipeline are reproducible and the comparison protocol is fixed.

### 1. Minimal HE and attention baselines

- Implement HE as a standalone causal sublayer, with a simple reference implementation first; optimize only after correctness is established.
- Implement the paper’s standard decoder baseline and its best reported self-attention setting (32 heads) under the same model/training configuration.
- Start at the paper’s small configuration (`d=128`, `l=128`, 18 layers, FFN width 512) if feasible. Track the exact batch and update budget; the paper’s 60,000 batches correspond to only 0.457 epochs.
- Add WE and ME only as low-cost diagnostic controls if the HE/attention path is working. Defer SHE: the paper’s accounting makes it about 12.8× the operations and 32.5× the parameters of 32-head attention at this setting.

**Gate:** Both paths pass correctness tests and can train end-to-end without silent shape, masking, or evaluation differences.

### 2. Independent quality replication

- Train each architecture with identical batches in identical order and matched optimizer/training budgets; repeat with multiple independent seeds.
- Keep a held-out split that is never used for training or model selection. Report held-out and training loss separately; include perplexity only where its calculation is well-defined and consistent.
- Compare HE against 32-head attention, and report WE/ME only as secondary controls. Publish per-seed curves, aggregate statistics, parameter counts, and analytically derived operation counts.
- Reproduce the paper’s reported setup as closely as available data permits, while separately labeling the held-out evaluation and any methodology improvements as deviations.

**Gate:** HE must show a consistent held-out quality result across seeds, not merely lower training loss in one run, before a systems comparison is warranted.

### 3. Measured systems comparison

- Benchmark forward training and incremental decoding separately, with warmup, synchronization, repeated trials, and identical hardware/software.
- Measure latency/throughput, peak memory, and parameter/weight traffic at multiple context lengths. Include both an uncomplicated reference and a responsibly optimized attention baseline; note kernel/fusion differences.
- Profile before optimizing HE. Its offset-specific gates and shared projection may shift cost to memory traffic or kernel-launch overhead despite the paper’s lower analytic operation count.
- Keep quality checkpoints paired with every performance result; compare only at an explicitly stated quality level and workload.

**Gate:** Claim a speed or memory improvement only when repeated measurements establish it against the optimized baseline without hiding a quality regression.

### 4. Scope decision

Review evidence and choose explicitly among:

- **Keep UnTransformers execution-only:** retain the replication as research, but do not add architecture replacement to the runtime.
- **Broaden the project:** revise its premise and define support for training/serving alternative architectures as a distinct workstream.
- **Stop:** archive the experiment if results fail to replicate or offer no measured advantage.

No option is assumed in advance. Any change to the project’s model-invariance premise requires a new decision; this plan does not silently redefine it.

## Out of scope for the first pass

- Replacing attention in pretrained Hugging Face checkpoints or claiming Extractors are drop-in for existing weights.
- Reproducing SHE first, scaling to modern LLM sizes, or building production kernels/serving integration before correctness and quality are established.
- Treating the paper’s critical-path argument or operation tables as measured performance evidence.
- Claiming a successful replication if the original corpus, configuration, or evaluation cannot be reconstructed.

## Deliverables

- A versioned experiment specification and provenance record.
- Minimal reference HE and attention implementations with correctness tests.
- Reproducible multi-seed quality results, followed only if the quality gate passes by measured systems benchmarks.
- A written scope decision based on those results.
