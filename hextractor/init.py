"""Initialization policy (HE_SPEC §7).

All weights ~ N(0, 0.01^2), biases zero, embedding matrices additionally
multiplied by sqrt(d), LayerNorm gain 1 / bias 0. A single seeded generator
per run; shared (non-sublayer-1) parameters are drawn FIRST so both arms
produce identical shared init under the same seed (T7 cross-arm check).
"""

import torch


@torch.no_grad()
def apply_init_policy(model, seed=0, std=0.01):
    g = torch.Generator().manual_seed(seed)

    def fill(name, p):
        if "norm" in name:                      # LayerNorm gain/bias
            p.copy_(torch.ones_like(p) if p.ndim == 1 and "bias" not in name
                    else torch.zeros_like(p))
        elif "tok_embed" in name or "pos_embed" in name:
            p.copy_(torch.randn(p.shape, generator=g) * std * (model.d ** 0.5))
        elif p.ndim == 1:                       # biases (FFN, head)
            p.copy_(torch.zeros_like(p))
        else:                                   # weight matrices
            p.copy_(torch.randn(p.shape, generator=g) * std)

    # pass 1: shared params (everything outside sublayer-1)
    for name, p in model.named_parameters():
        if not name.startswith("sub1."):
            fill(name, p)
    # pass 2: sublayer-1 params
    for name, p in model.named_parameters():
        if name.startswith("sub1."):
            fill(name, p)
    return model
