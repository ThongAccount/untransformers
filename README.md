# UnTransformers

*Transformers is popular. But it consumes countless GPUs and dedicated chips just to keep up with the pace. This repo goes the other way around.*

**Hugging Face Transformers, with the arrow reversed.**

---

## The other way around

Every trick the field reached for to keep transformers affordable points the same direction:

> **Shrink the model until it fits the machine.**

Quantize to 4 bits. Prune the experts. Distill 70B into 7B. Truncate the context to 4K. Serve a different
model than the one that was trained, and call it "good enough."

UnTransformers reverses the axis:

> **Bend the model until it fits the machine.**

The checkpoint stays untouched — full precision, full context, full architecture, every weight the author
trained. Nothing is quantized, nothing is pruned, nothing is distilled. Everything gained comes from the
side the industry left behind: *how the model is actually executed*.

|                        | The usual way                          | The UnTransformers way                       |
| ---------------------- | -------------------------------------- | -------------------------------------------- |
| Unit of work           | the model                              | the machine                                  |
| Allowed moves          | quantize, prune, distill, truncate     | kernels, memory, scheduling, parallelism      |
| Output fidelity        | "close enough"                         | the checkpoint, unchanged                    |
| Headline metric        | tokens/sec on a 4-bit model            | tokens/sec at full fidelity, per dollar      |

## The invariant

> **The model is never modified.**
> No quantization. No pruning. No distillation. No shortened context. No swapped architecture.
> If the logits move, it is a bug — not a feature.

Every optimization in this repo must be *execution-side*: it changes **how** the model runs, never **what**
the model is. This is checkable, which means it can be a hard gate rather than a vibe.

## Why this is worth a repo

The pace of model scaling keeps outrunning the pace of hardware. The industry's answer has been to throw
away capability to buy back throughput. The alternative — which nobody optimizes for, because it does not
make for a good benchmark headline — is to make a *single* machine stretch further.

That alternative has a compounding property: it stacks. A better kernel helps every model. A better KV
layout helps every long-context model. Better scheduling helps every concurrent workload. None of it
requires a retraining run, and none of it costs you a point of quality.

## What we're building

| Phase | Focus | Ship criterion |
| ----- | ----- | -------------- |
| **0** | Faithful runtime + parity harness | logits match HF `transformers` on the same weights, fp32 tolerance `1e-3` |
| **1** | Memory | paged KV cache, layer-by-layer weight streaming, CPU offload with prefetch — full context on a small card |
| **2** | Kernels | fused attention paths, graph capture, decode-optimized GEMM dispatch |
| **3** | Scheduling | continuous batching, chunked prefill, prefill/decode priority |
| **4** | Heterogeneous | CPU ⇄ GPU ⇄ NPU split, one model across mixed silicon |
| **5** | The stretch benchmark | tokens/sec/$ at full fidelity, published and reproducible |

Each phase is an ablation: one variable changed, everything else held fixed.

## Non-goals

- **Not a model zoo.** Checkpoints come from Hugging Face; this repo runs them.
- **Not a quantization library.** See the invariant.
- **Not a training framework.** Inference and serving, execution-side only.
- **Not a fork of `transformers`.** It is a drop-in runtime *around* it.

## Repo layout

```
untransformers/
├── README.md
├── pyproject.toml          # package + tool config
├── src/untransformers/     # runtime, kernels, schedulers
├── tests/                  # parity, correctness, benchmarks
└── bench/                  # reproducible benchmark harness
```

## Status

**Disclaimer:** This project is currently in development.

Day 0 — repository initialized. Nothing implemented yet.

## Contributing

Read `CONTRIBUTING.md` (coming with Phase 0). Every PR needs a parity check and a benchmark delta.

## License

TBD — pick one before the first release.
