"""T4 — Causal boundary: t=1 has no past reads, no padding artifacts."""

import torch

from hextractor.layers import HESublayer
from hextractor.oracle import he_oracle

from conftest import make_case


def test_t1_matches_oracle_formula():
    d = 16
    X, Win, w, Wadj, Wout = make_case(1, d, l=d, seed=7)
    oracle = he_oracle(X, Win, w, Wadj, Wout)
    # literal t=1 formula: (x_1 @ W_adj) ∘ ((x_1 @ W_in_ext) ∘ w_ext[0]) @ W_out
    literal = ((X @ Wadj) * ((X @ Win) * w[0])) @ Wout
    assert torch.allclose(oracle, literal, atol=1e-12)
    layer = HESublayer(d, d)
    with torch.no_grad():
        layer.W_in_ext.copy_(Win.float())
        layer.w_ext.copy_(w.float())
        layer.W_adj.copy_(Wadj.float())
        layer.W_out.copy_(Wout.float())
    assert torch.allclose(layer(X.float()), oracle.float(), atol=1e-6)


def test_t1_gate_grad_sparsity():
    d = 8
    X, Win, w, Wadj, Wout = make_case(1, d, l=d, seed=8)
    X, Win, w, Wadj, Wout = (t.float().requires_grad_(False) for t in (X, Win, w, Wadj, Wout))
    layer = HESublayer(d, d)
    with torch.no_grad():
        layer.W_in_ext.copy_(Win)
        layer.w_ext.copy_(w)
        layer.W_adj.copy_(Wadj)
        layer.W_out.copy_(Wout)
    layer.w_ext.requires_grad_(True)
    out = layer(X)
    out.sum().backward()
    # only offset k=1 (row 0) may receive gradient at t=1
    assert layer.w_ext.grad[1:].abs().sum().item() == 0.0
    assert layer.w_ext.grad[0].abs().sum().item() > 0.0
