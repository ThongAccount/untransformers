"""Shared transformer shell (HE_SPEC §1) — identical in both arms except
sublayer-1. Used by T12 (determinism) and later milestones; no training here.
"""

import torch
import torch.nn as nn

from .layers import HESublayer, AttentionSublayer, SDPAAttentionSublayer


def make_sublayer1(arm, d, l, n_heads=32):
    if arm in ("HE", "WE", "ME", "SHE"):
        return HESublayer(d, l, variant=arm)
    if arm == "attention-32h":
        return AttentionSublayer(d, n_heads=32)
    if arm == "attention-32h-sdpa":
        # Milestone 3 systems baseline only — same math, fused kernel.
        return SDPAAttentionSublayer(d, n_heads=32)
    if arm == "attention-1h":
        return AttentionSublayer(d, n_heads=1)
    raise ValueError(f"unknown arm {arm!r}")


class ExtractorTransformer(nn.Module):
    """Decoder-only shell per paper Eqs. 4-21, pre-LayerNorm, ReLU FFN.

    Args:
        arm: "HE" | "WE" | "ME" | "SHE" | "attention-32h" | "attention-1h"
        d, l, n_layers, ffn_width, vocab, dropout: per REPRODUCIBILITY.md §2.
    """

    def __init__(self, arm="HE", d=128, l=128, n_layers=18, ffn_width=512,
                 vocab=5000, n_heads=32, dropout=0.1):
        super().__init__()
        self.arm, self.d, self.l = arm, d, l
        self.tok_embed = nn.Parameter(torch.empty(vocab, d))
        self.pos_embed = nn.Parameter(torch.empty(l, d))
        self.drop = nn.Dropout(dropout)

        self.norms1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.sub1 = nn.ModuleList([make_sublayer1(arm, d, l, n_heads) for _ in range(n_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(d, ffn_width), nn.ReLU(), nn.Linear(ffn_width, d))
            for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)                          # Eq. 21, untied + bias

    def forward(self, ids):
        B, T = ids.shape
        assert T <= self.l, "context exceeds l"
        x = self.tok_embed[ids] * (self.d ** 0.5)                # Eq. 4
        x = x + self.pos_embed[:T] * (self.d ** 0.5)             # Eq. 5
        x = self.drop(x)                                         # Eq. 7
        for n1, s1, n2, f in zip(self.norms1, self.sub1, self.norms2, self.ffn):
            sub1_out = s1(n1(x))                                 # Eqs. 9 + 10-17/23-28
            x_in_sub2 = n2(self.drop(sub1_out) + x)              # Eq. 18
            sub2_out = f(x_in_sub2)                              # Eq. 19
            x = self.drop(sub2_out) + self.drop(sub1_out) + x    # Eq. 20 (two drops)
        x = self.final_norm(x)
        return self.head(x)                                      # Eq. 21
