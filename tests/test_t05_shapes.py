"""T5 — Shape / dtype / device over a parametrized grid."""

import pytest
import torch

from hextractor.layers import HESublayer, AttentionSublayer

from conftest import make_case


@pytest.mark.parametrize("d,l,t", [(8, 8, 1), (8, 16, 5), (33, 64, 17), (128, 128, 127), (64, 128, 128)])
def test_shape_dtype(d, l, t):
    X, Win, w, Wadj, Wout = make_case(t, d, l=l, seed=d + t)
    layer = HESublayer(d, l).eval()
    with torch.no_grad():
        layer.W_in_ext.copy_(Win.float())
        layer.w_ext.copy_(w.float())
        layer.W_adj.copy_(Wadj.float())
        layer.W_out.copy_(Wout.float())
    out = layer(X.float())
    assert out.shape == (t, d)
    assert out.dtype == torch.float32


def test_parameter_shapes_and_batched_input():
    d, l, t = 16, 32, 7
    layer = HESublayer(d, l)
    assert layer.W_in_ext.shape == (d, d)
    assert layer.w_ext.shape == (l, d)                    # offset axis FIRST
    assert layer.W_adj.shape == (d, d)
    assert layer.W_out.shape == (d, d)
    X = torch.randn(3, t, d)
    out = layer(X)
    assert out.shape == (3, t, d)

    att = AttentionSublayer(d, n_heads=4)
    assert att.Wq.shape == (d, d) and att.Wo.shape == (d, d)
    assert att(torch.randn(2, t, d)).shape == (2, t, d)


def test_negative_control_dtype_not_silently_upcast():
    # fp64 input must NOT silently produce fp32 via an internal cast.
    d, l, t = 8, 8, 4
    X, Win, w, Wadj, Wout = make_case(t, d, l=l, seed=1)
    layer = HESublayer(d, l).double()
    with torch.no_grad():
        layer.W_in_ext.copy_(Win)
        layer.w_ext.copy_(w)
        layer.W_adj.copy_(Wadj)
        layer.W_out.copy_(Wout)
    out = layer(X)
    assert out.dtype == torch.float64
