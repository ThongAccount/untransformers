"""WE and ME diagnostic controls (plan Milestone 1: added once the HE/attention
path is working). Same discipline as the HE suites: oracle double-loop first,
vectorized matching to 1e-5 (relative), causality bit-exact, negative controls
red before the fix. SHE stays deferred per the plan.

Per HE_SPEC §6: WE = gates + adjustment + out (no shared projection);
ME = scalar gates, extraction only (no adjustment, no output projection).
"""

import pytest
import torch

from hextractor.layers import HESublayer
from hextractor.oracle import he_oracle

from conftest import make_case


def build(variant, t, d, seed):
    X, _, w, Wadj, Wout = make_case(t, d, l=d, seed=seed)
    if variant == "ME":
        w_scalar = w[:, 0].contiguous()
        oracle = he_oracle(X, None, w_scalar, None, None, gate="scalar")
        layer = HESublayer(d, d, variant="ME")
        with torch.no_grad():
            layer.w_ext.copy_(w_scalar.float())
        return oracle, layer, X, w_scalar, Wadj, Wout
    oracle = he_oracle(X, None, w, Wadj, Wout)
    layer = HESublayer(d, d, variant="WE")
    with torch.no_grad():
        layer.w_ext.copy_(w.float())
        layer.W_adj.copy_(Wadj.float())
        layer.W_out.copy_(Wout.float())
    return oracle, layer, X, w, Wadj, Wout


def rel(impl, oracle):
    scale = oracle.float().abs().max().item()
    return (impl - oracle.float()).abs().max().item() / scale


@pytest.mark.parametrize("variant", ["WE", "ME"])
@pytest.mark.parametrize("t,d", [(1, 8), (8, 8), (16, 33), (128, 128)])
def test_matches_oracle(variant, t, d):
    oracle, layer, X, *_ = build(variant, t, d, seed=t * 7 + d)
    assert rel(layer(X.float()), oracle) <= 1e-5


def test_we_parameter_shapes_and_flags():
    d, l = 16, 32
    layer = HESublayer(d, l, variant="WE")
    assert getattr(layer, "W_in_ext", None) is None      # no shared projection
    assert layer.w_ext.shape == (l, d)
    assert layer.W_adj.shape == (d, d)
    assert layer.W_out.shape == (d, d)
    assert layer.n_params() == l * d + 2 * d * d


def test_me_parameter_shapes_and_flags():
    d, l = 16, 32
    layer = HESublayer(d, l, variant="ME")
    assert layer.w_ext.shape == (l,)                     # scalar gates
    assert (getattr(layer, "W_in_ext", None) is None
            and getattr(layer, "W_adj", None) is None
            and getattr(layer, "W_out", None) is None)
    assert layer.n_params() == l


@pytest.mark.parametrize("variant", ["WE", "ME"])
def test_causality_bit_exact(variant):
    _, layer, X, *_ = build(variant, 32, 32, seed=3)
    layer.eval()
    with torch.no_grad():
        base = layer(X.float())
        Xp = X.clone()
        Xp[10:] += 25.0                       # perturb rows > 9
        out = layer(Xp.float())
    assert torch.equal(out[:10], base[:10])   # rows <= 9 bit-identical
    assert not torch.allclose(out[10], base[10], atol=1e-6)


def test_we_negative_control_axis_flip_reddens():
    oracle, layer, X, w, Wadj, Wout = build("WE", 16, 16, seed=5)
    with torch.no_grad():
        layer.w_ext.copy_(w.flip(0).float())
    assert rel(layer(X.float()), oracle) > 1e-5


def test_me_negative_control_gate_sign_flip_reddens():
    oracle, layer, X, w_scalar, _, _ = build("ME", 16, 16, seed=5)
    with torch.no_grad():
        layer.w_ext.copy_((-w_scalar).float())
    assert rel(layer(X.float()), oracle) > 1e-5


def test_me_gate_gradient_analytic_no_hadamard():
    # ME has no adjustment/output projection, so dL/dw_ext[k-1] = dL/dout_ext[i] (.) x_ext[j]
    t, d = 8, 8
    X, _, w, _, _ = make_case(t, d, l=d, seed=6)
    w_scalar = w[:, 0].contiguous()
    layer = HESublayer(d, d, variant="ME")
    with torch.no_grad():
        layer.w_ext.copy_(w_scalar.float())
    Xf = X.float()
    layer.zero_grad()
    out = layer(Xf)
    g = torch.randn_like(out)
    (out * g).sum().backward()
    analytic = torch.zeros_like(layer.w_ext)
    for i in range(t):
        for j in range(i + 1):
            analytic[i - j] += (g[i] * Xf[j]).sum()   # scalar gate -> channel dot
    assert torch.allclose(layer.w_ext.grad, analytic, atol=1e-5, rtol=0)
