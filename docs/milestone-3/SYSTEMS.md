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

### Conditions

| field | value |
|---|---|
| device | cuda |
| hardware | Tesla T4 |
| torch | 2.11.0+cu130 |
| python | 3.13.16 |
| threads | 1 |
| commit | `b2a3221d006f4a41365e82531f0beb02e7b0a55e` (branch `research/milestone-1-extractor`, dirty: False) |
| arms | ['HE', 'attention-32h', 'attention-32h-sdpa'] |
| shape | d=128 l=128 layers=18 ffn=512 vocab=5000 heads=32 |
| batch | 64 |
| trials / warmup / decode trials | 10 / 2 / 5 |
| generated tokens | 16 |
| model contexts | [1, 8, 32, 64, 96, 112] |
| sublayer-1 contexts | [128, 256, 512, 1024, 2048, 4096, 8192] |
| measurement order | ['HE', 'attention-32h', 'attention-32h-sdpa', 'attention-32h-sdpa', 'attention-32h', 'HE'] (palindromic (ABBA): arms + reversed(arms)) |
| dtype | float32 |
| allow_tf32 (matmul / cuDNN) | False / False |
| float32_matmul_precision | highest |
| cuDNN (benchmark / deterministic) | False / True |
| deterministic_algorithms | False |

### Run-order null (A/A) and the drift floor

Every arm is measured twice, in a palindromic (ABBA) order, so the same arm appears at the start and at the end of the order **at the same context**. The spread of that null is the run-order drift floor: an arm-to-arm difference smaller than the floor of its context is not resolvable against clock/thermal drift. The floor of a context is the largest such spread over the arms.

| block | context | arm | first slot (index) | last slot (index) | first median s | last median s | difference s | relative | floor at this context (s) |
|---|---|---|---|---|---|---|---|---|---|
| training_step | model shape | HE | 0 | 5 | 1.139125 | 1.142458 | 0.003334 | 0.0029 | 0.003334 |
| training_step | model shape | attention-32h | 1 | 4 | 0.372666 | 0.374404 | 0.001738 | 0.0047 | 0.003334 |
| training_step | model shape | attention-32h-sdpa | 2 | 3 | 0.195442 | 0.196672 | 0.001230 | 0.0063 | 0.003334 |
| prefill | 1 | HE | 0 | 5 | 0.007112 | 0.007315 | 0.000203 | 0.0286 | 0.000203 |
| prefill | 1 | attention-32h | 1 | 4 | 0.009994 | 0.009959 | -0.000035 | 0.0035 | 0.000203 |
| prefill | 1 | attention-32h-sdpa | 2 | 3 | 0.007228 | 0.007283 | 0.000055 | 0.0076 | 0.000203 |
| prefill | 8 | HE | 0 | 5 | 0.012175 | 0.012600 | 0.000425 | 0.0349 | 0.004309 |
| prefill | 8 | attention-32h | 1 | 4 | 0.010423 | 0.010484 | 0.000061 | 0.0059 | 0.004309 |
| prefill | 8 | attention-32h-sdpa | 2 | 3 | 0.011561 | 0.007252 | -0.004309 | 0.3727 | 0.004309 |
| prefill | 32 | HE | 0 | 5 | 0.031145 | 0.028795 | -0.002350 | 0.0755 | 0.002350 |
| prefill | 32 | attention-32h | 1 | 4 | 0.010641 | 0.010843 | 0.000202 | 0.0190 | 0.002350 |
| prefill | 32 | attention-32h-sdpa | 2 | 3 | 0.007512 | 0.007426 | -0.000086 | 0.0114 | 0.002350 |
| prefill | 64 | HE | 0 | 5 | 0.054721 | 0.050340 | -0.004381 | 0.0801 | 0.004381 |
| prefill | 64 | attention-32h | 1 | 4 | 0.010734 | 0.010838 | 0.000104 | 0.0097 | 0.004381 |
| prefill | 64 | attention-32h-sdpa | 2 | 3 | 0.007445 | 0.007661 | 0.000216 | 0.0290 | 0.004381 |
| prefill | 96 | HE | 0 | 5 | 0.079631 | 0.070962 | -0.008669 | 0.1089 | 0.008669 |
| prefill | 96 | attention-32h | 1 | 4 | 0.010787 | 0.010981 | 0.000194 | 0.0180 | 0.008669 |
| prefill | 96 | attention-32h-sdpa | 2 | 3 | 0.007403 | 0.007561 | 0.000158 | 0.0213 | 0.008669 |
| prefill | 112 | HE | 0 | 5 | 0.092025 | 0.083914 | -0.008111 | 0.0881 | 0.008111 |
| prefill | 112 | attention-32h | 1 | 4 | 0.010798 | 0.011006 | 0.000208 | 0.0193 | 0.008111 |
| prefill | 112 | attention-32h-sdpa | 2 | 3 | 0.007369 | 0.007545 | 0.000176 | 0.0239 | 0.008111 |
| incremental_decode | 1 | HE | 0 | 5 | 0.014754 | 0.014545 | -0.000208 | 0.0141 | 0.000208 |
| incremental_decode | 1 | attention-32h | 1 | 4 | 0.011778 | 0.011733 | -0.000045 | 0.0038 | 0.000208 |
| incremental_decode | 1 | attention-32h-sdpa | 2 | 3 | 0.008010 | 0.007970 | -0.000040 | 0.0050 | 0.000208 |
| incremental_decode | 8 | HE | 0 | 5 | 0.019054 | 0.019173 | 0.000119 | 0.0063 | 0.000487 |
| incremental_decode | 8 | attention-32h | 1 | 4 | 0.011864 | 0.011811 | -0.000052 | 0.0044 | 0.000487 |
| incremental_decode | 8 | attention-32h-sdpa | 2 | 3 | 0.008467 | 0.007980 | -0.000487 | 0.0575 | 0.000487 |
| incremental_decode | 32 | HE | 0 | 5 | 0.035094 | 0.035001 | -0.000093 | 0.0026 | 0.000431 |
| incremental_decode | 32 | attention-32h | 1 | 4 | 0.011866 | 0.011434 | -0.000431 | 0.0363 | 0.000431 |
| incremental_decode | 32 | attention-32h-sdpa | 2 | 3 | 0.008308 | 0.007946 | -0.000362 | 0.0436 | 0.000431 |
| incremental_decode | 64 | HE | 0 | 5 | 0.056125 | 0.056662 | 0.000537 | 0.0096 | 0.000537 |
| incremental_decode | 64 | attention-32h | 1 | 4 | 0.012013 | 0.011692 | -0.000322 | 0.0268 | 0.000537 |
| incremental_decode | 64 | attention-32h-sdpa | 2 | 3 | 0.008307 | 0.008016 | -0.000291 | 0.0350 | 0.000537 |
| incremental_decode | 96 | HE | 0 | 5 | 0.076962 | 0.076703 | -0.000259 | 0.0034 | 0.000471 |
| incremental_decode | 96 | attention-32h | 1 | 4 | 0.011995 | 0.011523 | -0.000471 | 0.0393 | 0.000471 |
| incremental_decode | 96 | attention-32h-sdpa | 2 | 3 | 0.008231 | 0.007898 | -0.000333 | 0.0404 | 0.000471 |
| incremental_decode | 112 | HE | 0 | 5 | 0.088966 | 0.088278 | -0.000688 | 0.0077 | 0.000688 |
| incremental_decode | 112 | attention-32h | 1 | 4 | 0.011890 | 0.011490 | -0.000400 | 0.0336 | 0.000688 |
| incremental_decode | 112 | attention-32h-sdpa | 2 | 3 | 0.008089 | 0.008008 | -0.000081 | 0.0100 | 0.000688 |

| block | the artefact's block-level A/A difference (s) | relative |
|---|---|---|
| training_step | 0.003334 | 0.0063 |
| prefill | 0.008669 | 0.3727 |
| incremental_decode | 0.000688 | 0.0575 |

The artefact's block-level difference is the **largest** of the per-context floors above — the same-context null, never a cross-context one. The verdicts in the next table use the floor of the context they are printed in.

### Arm-to-arm differences against the drift floor

The floor for a block and a context is the **max−min spread across the A/A null's repetitions** (the same arm at the start and at the end of the palindromic order), taken as the **largest** such spread over the arms at that context. A difference is **resolvable** iff its absolute value exceeds the floor; below the floor it is **not resolvable, whatever its sign**. The rule was fixed before the systems numbers existed (`docs/milestone-3/SCOPE.md` §3.2). Exceeding the floor is necessary, not sufficient, for a claim.

| block | context | arm A | arm B | median A (s) | median B (s) | difference (s) | relative | floor (s) | resolvable? |
|---|---|---|---|---|---|---|---|---|---|
| training_step | model shape | HE | attention-32h | 1.140627 | 0.373457 | 0.767170 | 2.0542 | 0.003334 | **resolvable** |
| training_step | model shape | HE | attention-32h-sdpa | 1.140627 | 0.196479 | 0.944149 | 4.8053 | 0.003334 | **resolvable** |
| training_step | model shape | attention-32h | attention-32h-sdpa | 0.373457 | 0.196479 | 0.176978 | 0.9007 | 0.003334 | **resolvable** |
| prefill | 1 | HE | attention-32h | 0.007174 | 0.009994 | -0.002820 | 0.2822 | 0.000203 | **resolvable** |
| prefill | 1 | HE | attention-32h-sdpa | 0.007174 | 0.007231 | -0.000057 | 0.0079 | 0.000203 | not resolvable |
| prefill | 1 | attention-32h | attention-32h-sdpa | 0.009994 | 0.007231 | 0.002763 | 0.3822 | 0.000203 | **resolvable** |
| prefill | 8 | HE | attention-32h | 0.012319 | 0.010437 | 0.001882 | 0.1803 | 0.004309 | not resolvable |
| prefill | 8 | HE | attention-32h-sdpa | 0.012319 | 0.007360 | 0.004959 | 0.6738 | 0.004309 | **resolvable** |
| prefill | 8 | attention-32h | attention-32h-sdpa | 0.010437 | 0.007360 | 0.003077 | 0.4181 | 0.004309 | not resolvable |
| prefill | 32 | HE | attention-32h | 0.030139 | 0.010700 | 0.019439 | 1.8167 | 0.002350 | **resolvable** |
| prefill | 32 | HE | attention-32h-sdpa | 0.030139 | 0.007462 | 0.022677 | 3.0389 | 0.002350 | **resolvable** |
| prefill | 32 | attention-32h | attention-32h-sdpa | 0.010700 | 0.007462 | 0.003238 | 0.4339 | 0.002350 | **resolvable** |
| prefill | 64 | HE | attention-32h | 0.052592 | 0.010763 | 0.041829 | 3.8863 | 0.004381 | **resolvable** |
| prefill | 64 | HE | attention-32h-sdpa | 0.052592 | 0.007576 | 0.045016 | 5.9421 | 0.004381 | **resolvable** |
| prefill | 64 | attention-32h | attention-32h-sdpa | 0.010763 | 0.007576 | 0.003187 | 0.4207 | 0.004381 | not resolvable |
| prefill | 96 | HE | attention-32h | 0.077613 | 0.010874 | 0.066740 | 6.1377 | 0.008669 | **resolvable** |
| prefill | 96 | HE | attention-32h-sdpa | 0.077613 | 0.007482 | 0.070132 | 9.3737 | 0.008669 | **resolvable** |
| prefill | 96 | attention-32h | attention-32h-sdpa | 0.010874 | 0.007482 | 0.003392 | 0.4534 | 0.008669 | not resolvable |
| prefill | 112 | HE | attention-32h | 0.089404 | 0.010879 | 0.078525 | 7.2177 | 0.008111 | **resolvable** |
| prefill | 112 | HE | attention-32h-sdpa | 0.089404 | 0.007513 | 0.081891 | 10.8996 | 0.008111 | **resolvable** |
| prefill | 112 | attention-32h | attention-32h-sdpa | 0.010879 | 0.007513 | 0.003366 | 0.4480 | 0.008111 | not resolvable |
| incremental_decode | 1 | HE | attention-32h | 0.014712 | 0.011773 | 0.002939 | 0.2496 | 0.000208 | **resolvable** |
| incremental_decode | 1 | HE | attention-32h-sdpa | 0.014712 | 0.008002 | 0.006709 | 0.8384 | 0.000208 | **resolvable** |
| incremental_decode | 1 | attention-32h | attention-32h-sdpa | 0.011773 | 0.008002 | 0.003771 | 0.4712 | 0.000208 | **resolvable** |
| incremental_decode | 8 | HE | attention-32h | 0.019114 | 0.011840 | 0.007273 | 0.6143 | 0.000487 | **resolvable** |
| incremental_decode | 8 | HE | attention-32h-sdpa | 0.019114 | 0.008189 | 0.010924 | 1.3340 | 0.000487 | **resolvable** |
| incremental_decode | 8 | attention-32h | attention-32h-sdpa | 0.011840 | 0.008189 | 0.003651 | 0.4458 | 0.000487 | **resolvable** |
| incremental_decode | 32 | HE | attention-32h | 0.035076 | 0.011689 | 0.023387 | 2.0009 | 0.000431 | **resolvable** |
| incremental_decode | 32 | HE | attention-32h-sdpa | 0.035076 | 0.008175 | 0.026901 | 3.2907 | 0.000431 | **resolvable** |
| incremental_decode | 32 | attention-32h | attention-32h-sdpa | 0.011689 | 0.008175 | 0.003514 | 0.4298 | 0.000431 | **resolvable** |
| incremental_decode | 64 | HE | attention-32h | 0.056517 | 0.011872 | 0.044644 | 3.7604 | 0.000537 | **resolvable** |
| incremental_decode | 64 | HE | attention-32h-sdpa | 0.056517 | 0.008149 | 0.048368 | 5.9356 | 0.000537 | **resolvable** |
| incremental_decode | 64 | attention-32h | attention-32h-sdpa | 0.011872 | 0.008149 | 0.003724 | 0.4570 | 0.000537 | **resolvable** |
| incremental_decode | 96 | HE | attention-32h | 0.076844 | 0.011812 | 0.065033 | 5.5058 | 0.000471 | **resolvable** |
| incremental_decode | 96 | HE | attention-32h-sdpa | 0.076844 | 0.008063 | 0.068781 | 8.5307 | 0.000471 | **resolvable** |
| incremental_decode | 96 | attention-32h | attention-32h-sdpa | 0.011812 | 0.008063 | 0.003749 | 0.4649 | 0.000471 | **resolvable** |
| incremental_decode | 112 | HE | attention-32h | 0.088416 | 0.011701 | 0.076715 | 6.5562 | 0.000688 | **resolvable** |
| incremental_decode | 112 | HE | attention-32h-sdpa | 0.088416 | 0.008035 | 0.080381 | 10.0035 | 0.000688 | **resolvable** |
| incremental_decode | 112 | attention-32h | attention-32h-sdpa | 0.011701 | 0.008035 | 0.003666 | 0.4562 | 0.000688 | **resolvable** |

### Training step (forward + backward + optimizer step)

| arm | median s/step | mean | p10 | p90 | min | n | tok/s | params | peak memory bytes (GPU: allocated / reserved; CPU: process RSS) | memory scope | quality level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HE | 1.1406 | 1.1491 | 1.1358 | 1.1596 | 1.1296 | 20 | 7170.5 | 4861320 | 1707080704 / 1920991232 | per-arm: max over this arm's timed slots of torch.cuda.max_memory_allocated/max_memory_reserved (counters reset before every slot) (S6) | trained checkpoint (seed 1, held-out loss 4.138239) |
| attention-32h | 0.3735 | 0.3735 | 0.3725 | 0.3747 | 0.3719 | 20 | 21880.1 | 4861320 | 4123294720 / 4351590400 | per-arm: max over this arm's timed slots of torch.cuda.max_memory_allocated/max_memory_reserved (counters reset before every slot) (S6) | trained checkpoint (seed 1, held-out loss 4.243823) |
| attention-32h-sdpa | 0.1965 | 0.1959 | 0.1941 | 0.1972 | 0.1939 | 20 | 41653.2 | 4861320 | 1725955072 / 1939865600 | per-arm: max over this arm's timed slots of torch.cuda.max_memory_allocated/max_memory_reserved (counters reset before every slot) (S6) | architecture at initialisation (no quality level) |

- HE here is an eager Python offset loop (hextractor/layers.py: the module docstring states there are no fused kernels); the attention arms may dispatch a fused SDPA kernel. A wall-clock difference between them therefore measures implementation maturity under these conditions, not the HE mechanism as such.

### Prefill latency (one full forward over the prompt)

| arm | context | median s | p10 | p90 | min | n | tok/s | peak memory bytes (GPU: allocated / reserved; CPU: process RSS) | quality level |
|---|---|---|---|---|---|---|---|---|---|
| HE | 1 | 0.00717 | 0.00691 | 0.00755 | 0.00677 | 20 | 136.71 | 37556736 / 54525952 | trained checkpoint (seed 1, held-out loss 4.138239) |
| HE | 8 | 0.01232 | 0.01192 | 0.01405 | 0.01177 | 20 | 634.91 | 37710848 / 54525952 | trained checkpoint (seed 1, held-out loss 4.138239) |
| HE | 32 | 0.03014 | 0.02862 | 0.03182 | 0.02825 | 20 | 1111.32 | 38239744 / 54525952 | trained checkpoint (seed 1, held-out loss 4.138239) |
| HE | 64 | 0.05259 | 0.04991 | 0.05854 | 0.04957 | 20 | 1271.35 | 39168512 / 54525952 | trained checkpoint (seed 1, held-out loss 4.138239) |
| HE | 96 | 0.07761 | 0.07057 | 0.08530 | 0.07029 | 20 | 1352.84 | 39651328 / 54525952 | trained checkpoint (seed 1, held-out loss 4.138239) |
| HE | 112 | 0.08940 | 0.08206 | 0.09753 | 0.08138 | 20 | 1334.70 | 40004096 / 54525952 | trained checkpoint (seed 1, held-out loss 4.138239) |
| attention-32h | 1 | 0.00999 | 0.00972 | 0.01034 | 0.00950 | 20 | 100.41 | 37556736 / 54525952 | trained checkpoint (seed 1, held-out loss 4.243823) |
| attention-32h | 8 | 0.01044 | 0.01015 | 0.01087 | 0.00998 | 20 | 763.05 | 37710848 / 54525952 | trained checkpoint (seed 1, held-out loss 4.243823) |
| attention-32h | 32 | 0.01070 | 0.01049 | 0.01115 | 0.01044 | 20 | 2951.08 | 38239744 / 54525952 | trained checkpoint (seed 1, held-out loss 4.243823) |
| attention-32h | 64 | 0.01076 | 0.01063 | 0.01109 | 0.01059 | 20 | 5905.10 | 39168512 / 56623104 | trained checkpoint (seed 1, held-out loss 4.243823) |
| attention-32h | 96 | 0.01087 | 0.01060 | 0.01116 | 0.01049 | 20 | 8742.15 | 41123840 / 56623104 | trained checkpoint (seed 1, held-out loss 4.243823) |
| attention-32h | 112 | 0.01088 | 0.01065 | 0.01227 | 0.01043 | 20 | 10176.29 | 41303552 / 56623104 | trained checkpoint (seed 1, held-out loss 4.243823) |
| attention-32h-sdpa | 1 | 0.00723 | 0.00710 | 0.00766 | 0.00690 | 20 | 137.30 | 37556736 / 54525952 | architecture at initialisation (no quality level) |
| attention-32h-sdpa | 8 | 0.00736 | 0.00707 | 0.01387 | 0.00702 | 20 | 1103.17 | 37710848 / 54525952 | architecture at initialisation (no quality level) |
| attention-32h-sdpa | 32 | 0.00746 | 0.00717 | 0.00814 | 0.00711 | 20 | 4308.96 | 38239744 / 54525952 | architecture at initialisation (no quality level) |
| attention-32h-sdpa | 64 | 0.00758 | 0.00728 | 0.00793 | 0.00725 | 20 | 8354.43 | 39168512 / 54525952 | architecture at initialisation (no quality level) |
| attention-32h-sdpa | 96 | 0.00748 | 0.00723 | 0.00766 | 0.00715 | 20 | 12696.40 | 39651328 / 54525952 | architecture at initialisation (no quality level) |
| attention-32h-sdpa | 112 | 0.00751 | 0.00733 | 0.00798 | 0.00710 | 20 | 14844.06 | 40004096 / 54525952 | architecture at initialisation (no quality level) |

### Incremental decoding (per generated token)

The first generated token is reported separately from the steady state; the pooled row keeps every sample.

| arm | context | first-token median s | steady-state median s | pooled median s | pooled n | steady tok/s | tok/s incl. first | cache bytes at prompt | per layer | quality level |
|---|---|---|---|---|---|---|---|---|---|---|
| HE | 1 | 0.008717 | 0.014712 | 0.014353 | 160 | 67.97 | 71.06 | 9216 | 512 | trained checkpoint (seed 1, held-out loss 4.138239) |
| HE | 8 | 0.013606 | 0.019114 | 0.018718 | 160 | 52.32 | 53.99 | 73728 | 4096 | trained checkpoint (seed 1, held-out loss 4.138239) |
| HE | 32 | 0.029389 | 0.035076 | 0.034888 | 160 | 28.51 | 28.72 | 294912 | 16384 | trained checkpoint (seed 1, held-out loss 4.138239) |
| HE | 64 | 0.050314 | 0.056517 | 0.056012 | 160 | 17.69 | 17.76 | 589824 | 32768 | trained checkpoint (seed 1, held-out loss 4.138239) |
| HE | 96 | 0.070851 | 0.076844 | 0.076605 | 160 | 13.01 | 13.10 | 884736 | 49152 | trained checkpoint (seed 1, held-out loss 4.138239) |
| HE | 112 | 0.082909 | 0.088416 | 0.088322 | 160 | 11.31 | 11.38 | 1032192 | 57344 | trained checkpoint (seed 1, held-out loss 4.138239) |
| attention-32h | 1 | 0.011876 | 0.011773 | 0.011780 | 160 | 84.94 | 85.23 | 18432 | 1024 | trained checkpoint (seed 1, held-out loss 4.243823) |
| attention-32h | 8 | 0.011936 | 0.011840 | 0.011856 | 160 | 84.46 | 84.57 | 147456 | 8192 | trained checkpoint (seed 1, held-out loss 4.243823) |
| attention-32h | 32 | 0.011775 | 0.011689 | 0.011699 | 160 | 85.55 | 87.34 | 589824 | 32768 | trained checkpoint (seed 1, held-out loss 4.243823) |
| attention-32h | 64 | 0.011968 | 0.011872 | 0.011877 | 160 | 84.23 | 85.22 | 1179648 | 65536 | trained checkpoint (seed 1, held-out loss 4.243823) |
| attention-32h | 96 | 0.011867 | 0.011812 | 0.011815 | 160 | 84.66 | 86.74 | 1769472 | 98304 | trained checkpoint (seed 1, held-out loss 4.243823) |
| attention-32h | 112 | 0.011573 | 0.011701 | 0.011699 | 160 | 85.46 | 87.01 | 2064384 | 114688 | trained checkpoint (seed 1, held-out loss 4.243823) |
| attention-32h-sdpa | 1 | 0.007921 | 0.008002 | 0.007990 | 160 | 124.96 | 125.53 | 18432 | 1024 | architecture at initialisation (no quality level) |
| attention-32h-sdpa | 8 | 0.008145 | 0.008189 | 0.008189 | 160 | 122.11 | 125.22 | 147456 | 8192 | architecture at initialisation (no quality level) |
| attention-32h-sdpa | 32 | 0.008072 | 0.008175 | 0.008169 | 160 | 122.33 | 126.09 | 589824 | 32768 | architecture at initialisation (no quality level) |
| attention-32h-sdpa | 64 | 0.008185 | 0.008149 | 0.008161 | 160 | 122.72 | 124.61 | 1179648 | 65536 | architecture at initialisation (no quality level) |
| attention-32h-sdpa | 96 | 0.008273 | 0.008063 | 0.008078 | 160 | 124.03 | 126.47 | 1769472 | 98304 | architecture at initialisation (no quality level) |
| attention-32h-sdpa | 112 | 0.008242 | 0.008035 | 0.008056 | 160 | 124.45 | 124.68 | 2064384 | 114688 | architecture at initialisation (no quality level) |

### Sublayer-1 scaling (one incremental token, one sublayer)

The state bytes are **analytic** (the formula is printed), not measured; the latencies are measured.

| arm | context | median s/token | p10 | p90 | n | state bytes/layer | state bytes all layers | formula |
|---|---|---|---|---|---|---|---|---|
| HE | 128 | 0.004696 | 0.004616 | 0.004780 | 10 | 65024 | 1170432 | `(l - 1) * d * 4 bytes = (128 - 1) * 128 * 4` |
| HE | 256 | 0.005200 | 0.004981 | 0.005446 | 10 | 65024 | 1170432 | `(l - 1) * d * 4 bytes = (128 - 1) * 128 * 4` |
| HE | 512 | 0.004720 | 0.004597 | 0.004864 | 10 | 65024 | 1170432 | `(l - 1) * d * 4 bytes = (128 - 1) * 128 * 4` |
| HE | 1024 | 0.004678 | 0.004580 | 0.004754 | 10 | 65024 | 1170432 | `(l - 1) * d * 4 bytes = (128 - 1) * 128 * 4` |
| HE | 2048 | 0.004861 | 0.004755 | 0.004977 | 10 | 65024 | 1170432 | `(l - 1) * d * 4 bytes = (128 - 1) * 128 * 4` |
| HE | 4096 | 0.004754 | 0.004690 | 0.004834 | 10 | 65024 | 1170432 | `(l - 1) * d * 4 bytes = (128 - 1) * 128 * 4` |
| HE | 8192 | 0.005623 | 0.004909 | 0.008494 | 10 | 65024 | 1170432 | `(l - 1) * d * 4 bytes = (128 - 1) * 128 * 4` |
| attention-32h | 128 | 0.000443 | 0.000427 | 0.000475 | 10 | 131072 | 2359296 | `2 * T * d * 4 bytes = 2 * 128 * 128 * 4` |
| attention-32h | 256 | 0.000506 | 0.000438 | 0.000530 | 10 | 262144 | 4718592 | `2 * T * d * 4 bytes = 2 * 256 * 128 * 4` |
| attention-32h | 512 | 0.000474 | 0.000436 | 0.000528 | 10 | 524288 | 9437184 | `2 * T * d * 4 bytes = 2 * 512 * 128 * 4` |
| attention-32h | 1024 | 0.000443 | 0.000422 | 0.000481 | 10 | 1048576 | 18874368 | `2 * T * d * 4 bytes = 2 * 1024 * 128 * 4` |
| attention-32h | 2048 | 0.000452 | 0.000421 | 0.000480 | 10 | 2097152 | 37748736 | `2 * T * d * 4 bytes = 2 * 2048 * 128 * 4` |
| attention-32h | 4096 | 0.000448 | 0.000430 | 0.000484 | 10 | 4194304 | 75497472 | `2 * T * d * 4 bytes = 2 * 4096 * 128 * 4` |
| attention-32h | 8192 | 0.000465 | 0.000429 | 0.000493 | 10 | 8388608 | 150994944 | `2 * T * d * 4 bytes = 2 * 8192 * 128 * 4` |
| attention-32h-sdpa | 128 | 0.000216 | 0.000206 | 0.000236 | 10 | 131072 | 2359296 | `2 * T * d * 4 bytes = 2 * 128 * 128 * 4` |
| attention-32h-sdpa | 256 | 0.000212 | 0.000204 | 0.000229 | 10 | 262144 | 4718592 | `2 * T * d * 4 bytes = 2 * 256 * 128 * 4` |
| attention-32h-sdpa | 512 | 0.000260 | 0.000248 | 0.000279 | 10 | 524288 | 9437184 | `2 * T * d * 4 bytes = 2 * 512 * 128 * 4` |
| attention-32h-sdpa | 1024 | 0.000319 | 0.000308 | 0.000321 | 10 | 1048576 | 18874368 | `2 * T * d * 4 bytes = 2 * 1024 * 128 * 4` |
| attention-32h-sdpa | 2048 | 0.000459 | 0.000455 | 0.000490 | 10 | 2097152 | 37748736 | `2 * T * d * 4 bytes = 2 * 2048 * 128 * 4` |
| attention-32h-sdpa | 4096 | 0.000797 | 0.000794 | 0.000806 | 10 | 4194304 | 75497472 | `2 * T * d * 4 bytes = 2 * 4096 * 128 * 4` |
| attention-32h-sdpa | 8192 | 0.001469 | 0.001456 | 0.001500 | 10 | 8388608 | 150994944 | `2 * T * d * 4 bytes = 2 * 8192 * 128 * 4` |

### Per-token state traffic (analytic)

| arm | state bytes/layer/token | all layers/token | formula | grows with context |
|---|---|---|---|---|
| HE | 65024 | 1170432 | `(l - 1) * d * 4 = (128 - 1) * 128 * 4 bytes` | False |
| attention-32h | context-dependent (see the sublayer-1 table) | context-dependent (see the sublayer-1 table) | `2 * T * d * 4 bytes (K and V caches)` | True |
| attention-32h-sdpa | context-dependent (see the sublayer-1 table) | context-dependent (see the sublayer-1 table) | `2 * T * d * 4 bytes (K and V caches)` | True |

- *HE*: HE's per-token state is flat in the context length: the offset gates only look back l-1 positions
- *attention-32h*: attention's KV cache holds 2 * T * d floats per layer and grows linearly with the context; the implementation here reallocates it with torch.cat on every token (declared limitation, PROTOCOL.md §8.D)
- *attention-32h-sdpa*: attention's KV cache holds 2 * T * d floats per layer and grows linearly with the context; the implementation here reallocates it with torch.cat on every token (declared limitation, PROTOCOL.md §8.D)

### Weight traffic (analytic)

| arm | params | parameter bytes (fp32) | training-step bytes read | decode-step bytes read |
|---|---|---|---|---|
| HE | 4861320 | 19445280 | 58335840 | 19445280 |
| attention-32h | 4861320 | 19445280 | 58335840 | 19445280 |
| attention-32h-sdpa | 4861320 | 19445280 | 58335840 | 19445280 |

- *HE* training read: `3 * n_params * 4 bytes (forward + dgrad + wgrad)`; decode read: `1 * n_params * 4 bytes (one forward pass)`
- *HE* note: analytic: a training step reads the parameter bytes three times (forward, dgrad, wgrad); a decode step reads them once. Both arms carry the same parameter count by construction, so this block cannot show a traffic difference between the arms.
- *attention-32h* training read: `3 * n_params * 4 bytes (forward + dgrad + wgrad)`; decode read: `1 * n_params * 4 bytes (one forward pass)`
- *attention-32h* note: analytic: a training step reads the parameter bytes three times (forward, dgrad, wgrad); a decode step reads them once. Both arms carry the same parameter count by construction, so this block cannot show a traffic difference between the arms.
- *attention-32h-sdpa* training read: `3 * n_params * 4 bytes (forward + dgrad + wgrad)`; decode read: `1 * n_params * 4 bytes (one forward pass)`
- *attention-32h-sdpa* note: analytic: a training step reads the parameter bytes three times (forward, dgrad, wgrad); a decode step reads them once. Both arms carry the same parameter count by construction, so this block cannot show a traffic difference between the arms.

### Paired quality attached to the systems rows

| arm | seed | steps | final held-out loss |
|---|---|---|---|
| HE | 1 | 2000 | 4.138239 |
| HE | 2 | 2000 | 4.149547 |
| HE | 3 | 2000 | 4.151253 |
| attention-32h | 1 | 2000 | 4.243823 |
| attention-32h | 2 | 2000 | 4.256581 |
| attention-32h | 3 | 2000 | 4.250221 |

### Caveats carried by the artefact

- HE here is an eager Python offset loop (hextractor/layers.py: the module docstring states there are no fused kernels); the attention arms may dispatch a fused SDPA kernel. A wall-clock difference between them therefore measures implementation maturity under these conditions, not the HE mechanism as such.
- the attention KV cache is reallocated with torch.cat on every decoded token (O(T) copy per token), which inflates the attention decode latency relative to a preallocated cache (PROTOCOL.md §8.D, out of budget for this round)
- CPU memory rows are process-level; the CUDA rows are per-arm (see the 'scope' field of every memory block)
- traffic blocks are analytic formulas, not measurements
- rows whose quality block says 'architecture at initialisation' carry no quality level and must not be paired with a quality number

## 2. How to read these rows

* **Every row carries its units, its denominators, its trial count and its
  quality level.** A row whose quality level says "architecture at
  initialisation" is not tied to any quality number and must never be paired
  with one; a row whose quality level names a checkpoint carries that
  checkpoint's held-out loss.
* **The arms are measured twice, in a palindromic (ABBA) order.** The A/A
  null — the same arm at the start and at the end of the order, **at the same
  context** — is the run-order drift floor. An arm-to-arm difference smaller
  than the floor of its context is not resolvable against clock/thermal drift,
  and is reported as such rather than as a result.
* **The floor is printed next to every arm-to-arm difference.** The rule was
  fixed before the systems numbers existed: the floor for a block and a
  context is the **max−min spread across the A/A null's repetitions** (the
  same arm, first slot and last slot, at that context), taken as the
  **largest** such spread over the arms at that context; a difference
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
