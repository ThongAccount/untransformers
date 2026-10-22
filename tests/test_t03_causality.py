"""T3 — Causality: perturbing token rows j > i must leave rows <= i bit-identical."""

import torch

from hextractor.layers import HESublayer

from conftest import make_case


def eval_layer(t, d, seed=3):
    X, Win, w, Wadj, Wout = make_case(t, d, l=d, seed=seed)
    layer = HESublayer(d, d).eval()
    with torch.no_grad():
        layer.W_in_ext.copy_(Win.float())
        layer.w_ext.copy_(w.float())
        layer.W_adj.copy_(Wadj.float())
        layer.W_out.copy_(Wout.float())
    return layer, X


def test_bit_exact_under_future_perturbation():
    layer, X = eval_layer(64, 32)
    with torch.no_grad():
        base = layer(X.float())
        for i in (0, 31, 62):
            Xp = X.clone()
            Xp[i + 1:] += 100.0                      # perturb everything after row i
            out = layer(Xp.float())
            assert torch.equal(out[: i + 1], base[: i + 1]), f"row {i} changed"


def test_negative_control_non_causal_differs():
    # A non-causal (full-j sum) computation MUST redden this suite.
    layer, X = eval_layer(16, 8)
    with torch.no_grad():
        base = layer(X.float())
        Xp = X.clone()
        Xp[1:] += 100.0
        # emulate non-causal: perturb then compare a LATER row, which must change
        assert not torch.equal(layer(Xp.float())[1], base[1])
