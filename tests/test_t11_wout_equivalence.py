"""T11 — Optional-projection equivalence: identity W^out => ON == OFF."""

import torch

from hextractor.layers import HESublayer

from conftest import make_case


def test_identity_wout_paths_agree_bit_exact():
    d, l, t = 16, 16, 9
    X, Win, w, Wadj, Wout = make_case(t, d, l=l, seed=31)
    layer_on = HESublayer(d, l).eval()
    with torch.no_grad():
        layer_on.W_in_ext.copy_(Win.float())
        layer_on.w_ext.copy_(w.float())
        layer_on.W_adj.copy_(Wadj.float())
        layer_on.W_out.copy_(torch.eye(d))               # identity
    layer_off = HESublayer(d, l).eval()
    layer_off.use_output_projection = False
    layer_off.W_out = None
    with torch.no_grad():
        layer_off.W_in_ext.copy_(Win.float())
        layer_off.w_ext.copy_(w.float())
        layer_off.W_adj.copy_(Wadj.float())
    out_on = layer_on(X.float())
    out_off = layer_off(X.float())
    assert torch.equal(out_on, out_off)                  # bit-identical


def test_negative_control_nonidentity_differs():
    d, l, t = 16, 16, 9
    X, Win, w, Wadj, Wout = make_case(t, d, l=l, seed=31)
    layer_on = HESublayer(d, l).eval()
    with torch.no_grad():
        layer_on.W_in_ext.copy_(Win.float())
        layer_on.w_ext.copy_(w.float())
        layer_on.W_adj.copy_(Wadj.float())
        layer_on.W_out.copy_(Wout.float())               # non-identity
    layer_off = HESublayer(d, l).eval()
    layer_off.use_output_projection = False
    layer_off.W_out = None
    with torch.no_grad():
        layer_off.W_in_ext.copy_(Win.float())
        layer_off.w_ext.copy_(w.float())
        layer_off.W_adj.copy_(Wadj.float())
    assert not torch.allclose(layer_on(X.float()), layer_off(X.float()), atol=1e-5)
