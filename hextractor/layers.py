"""Vectorized sublayer-1 implementations.

Correctness contract (HE_SPEC §8): these are the vectorized forms that must
match oracle.py's double loop to <= 1e-5 (fp32) before any fused shortcut is
allowed. No fused kernels here — the offset loop over k is the vectorized
reference form itself (O(t) iterations of [t, d] elementwise work).
"""

import torch
import torch.nn as nn


class HESublayer(nn.Module):
    """Causal, softmax-free Extractor sublayer (paper Eqs. 27-25).

    variants (parameter inventory, HE_SPEC §6):
        HE  — shared projection + per-offset vector gates + adjustment + out
        WE  — gates + adjustment + out (no shared projection)
        ME  — scalar gates, extraction only
        SHE — per-offset matrix gates + adjustment + out
    """

    def __init__(self, d, l, variant="HE"):
        super().__init__()
        assert variant in ("HE", "WE", "ME", "SHE"), variant
        self.d, self.l, self.variant = d, l, variant
        self.use_shared_projection = variant == "HE"
        self.use_adjustment = variant in ("HE", "WE", "SHE")
        self.use_output_projection = variant in ("HE", "WE", "SHE")

        if self.use_shared_projection:
            self.W_in_ext = nn.Parameter(torch.empty(d, d))
        if variant == "ME":
            self.w_ext = nn.Parameter(torch.empty(l))            # scalar gates
        elif variant == "SHE":
            self.w_ext = nn.Parameter(torch.empty(l, d, d))      # per-offset [d,d]
        else:
            self.w_ext = nn.Parameter(torch.empty(l, d))         # offset axis FIRST
        if self.use_adjustment:
            self.W_adj = nn.Parameter(torch.empty(d, d))
        if self.use_output_projection:
            self.W_out = nn.Parameter(torch.empty(d, d))

    def forward(self, X):
        """X: [..., t, d] -> [..., t, d]. Strictly causal."""
        t = X.shape[-2]
        src = X @ self.W_in_ext if self.use_shared_projection else X

        out = torch.zeros_like(src)
        kmax = min(t, self.l)
        for k in range(kmax):
            # offset k (0-based): w_ext[k] gates src[..., i-k, :] -> out[..., i, :], i >= k
            if self.variant == "ME":
                out[..., k:, :] += src[..., : t - k, :] * self.w_ext[k]
            elif self.variant == "SHE":
                out[..., k:, :] += src[..., : t - k, :] @ self.w_ext[k]
            else:
                out[..., k:, :] += src[..., : t - k, :] * self.w_ext[k]

        if self.use_adjustment:
            out = (X @ self.W_adj) * out                         # Eq. 24: per-position
        if self.use_output_projection:
            out = out @ self.W_out                               # Eq. 25
        return out

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class AttentionSublayer(nn.Module):
    """Multi-head self-attention, paper Eqs. 10-17. No biases."""

    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0
        self.d, self.n_heads, self.h = d, n_heads, d // n_heads
        self.Wq = nn.Parameter(torch.empty(d, d))
        self.Wk = nn.Parameter(torch.empty(d, d))
        self.Wv = nn.Parameter(torch.empty(d, d))
        self.Wo = nn.Parameter(torch.empty(d, d))

    def forward(self, X):
        t = X.shape[-2]
        q = (X @ self.Wq).unflatten(-1, (self.n_heads, self.h)).transpose(-2, -3)
        k = (X @ self.Wk).unflatten(-1, (self.n_heads, self.h)).transpose(-2, -3)
        v = (X @ self.Wv).unflatten(-1, (self.n_heads, self.h)).transpose(-2, -3)
        scores = q @ k.transpose(-1, -2) / (self.h ** 0.5)       # Eq. 15-16 scaling
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=X.device), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))       # Eq. 14 mask
        ctx = scores.softmax(dim=-1) @ v
        ctx = ctx.transpose(-2, -3).flatten(-2)
        return ctx @ self.Wo                                     # Eq. 17

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class SDPAAttentionSublayer(nn.Module):
    """Optimized attention baseline (Milestone 3): same math, same parameter
    names and shapes as `AttentionSublayer`, but the attention core is
    `torch.nn.functional.scaled_dot_product_attention` (flash / mem-efficient /
    math kernels chosen by the runtime).

    This is the "responsibly optimized attention baseline" the plan asks for:
    the reference implementation above materialises the t x t score matrix,
    this one asks the runtime for a fused kernel. Kernel choice depends on
    the device and the shapes, so every measurement records which kernel the
    runtime picked.

    It is NOT part of the quality protocol — the quality arms stay exactly
    `HE` and `attention-32h`.
    """

    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0
        self.d, self.n_heads, self.h = d, n_heads, d // n_heads
        self.Wq = nn.Parameter(torch.empty(d, d))
        self.Wk = nn.Parameter(torch.empty(d, d))
        self.Wv = nn.Parameter(torch.empty(d, d))
        self.Wo = nn.Parameter(torch.empty(d, d))

    def forward(self, X):
        t = X.shape[-2]
        q = (X @ self.Wq).unflatten(-1, (self.n_heads, self.h)).transpose(-2, -3)
        k = (X @ self.Wk).unflatten(-1, (self.n_heads, self.h)).transpose(-2, -3)
        v = (X @ self.Wv).unflatten(-1, (self.n_heads, self.h)).transpose(-2, -3)
        ctx = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.h ** -0.5)
        ctx = ctx.transpose(-2, -3).flatten(-2)
        return ctx @ self.Wo

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
