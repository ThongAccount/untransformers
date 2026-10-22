"""T8 — Gradient checks: gradcheck, analytic gate gradients, gate sparsity."""

import torch

from hextractor.layers import HESublayer

from conftest import make_case


def test_gradcheck_fp64_small():
    d, l, t = 4, 4, 3
    X, Win, w, Wadj, Wout = make_case(t, d, l=l, seed=9)
    layer = HESublayer(d, l).double()
    for p in layer.parameters():
        p.requires_grad_(True)
    with torch.no_grad():
        layer.W_in_ext.copy_(Win)
        layer.w_ext.copy_(w)
        layer.W_adj.copy_(Wadj)
        layer.W_out.copy_(Wout)
    def f(x):
        return layer(x)
    Xc = X.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(f, (Xc,), eps=1e-6, atol=1e-6)


def test_analytic_gate_gradient():
    # with x_ext fixed: dL/dw_ext[i-j+1] = dL/dx_out_ext[i] (.) x_ext[j]
    d, l, t = 8, 8, 5
    X, Win, w, Wadj, Wout = make_case(t, d, l=l, seed=10)
    layer = HESublayer(d, l)
    with torch.no_grad():
        layer.W_in_ext.copy_(Win.float())
        layer.w_ext.copy_(w.float())
        layer.W_adj.copy_(Wadj.float())
        layer.W_out.copy_(Wout.float())
    Xf = X.float().requires_grad_(False)

    x_ext = Xf @ layer.W_in_ext
    out = layer(Xf)
    g = torch.randn_like(out)
    # out = ((X@W_adj) * out_ext) @ W_out  ->  dL/d out_ext = (g @ W_out^T) * (X@W_adj)
    dL_d_out_ext = (g @ layer.W_out.T) * (Xf @ layer.W_adj)

    layer.zero_grad()
    (out * g).sum().backward()
    auto = layer.w_ext.grad.clone()

    analytic = torch.zeros_like(layer.w_ext)
    for i in range(t):
        for j in range(i + 1):
            analytic[i - j] += dL_d_out_ext[i] * x_ext[j]
    assert torch.allclose(auto, analytic, atol=1e-6, rtol=0), (auto - analytic).abs().max()


def test_negative_control_flipped_gate_gradient_differs():
    d, l, t = 8, 8, 5
    X, Win, w, Wadj, Wout = make_case(t, d, l=l, seed=10)
    layer = HESublayer(d, l)
    with torch.no_grad():
        layer.W_in_ext.copy_(Win.float())
        layer.w_ext.copy_(w.flip(0).float())
        layer.W_adj.copy_(Wadj.float())
        layer.W_out.copy_(Wout.float())
    out = layer(X.float())
    out.sum().backward()
    # with the axis flipped, the analytic relation must NOT hold
    x_ext = X.float() @ layer.W_in_ext
    analytic = torch.zeros_like(layer.w_ext)
    for i in range(t):
        for j in range(i + 1):
            analytic[i - j] += (X.float() @ layer.W_adj)[i] * x_ext[j]
    assert not torch.allclose(layer.w_ext.grad, analytic, atol=1e-6, rtol=0)
