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
