"""T9 — Adjustment is per-position (Eq. 24's matrix reading), not last-token."""

import torch

from hextractor.layers import HESublayer

from conftest import make_case


def build(t, d, seed=13):
    X, Win, w, Wadj, Wout = make_case(t, d, l=d, seed=seed)
    layer = HESublayer(d, d).eval()
    with torch.no_grad():
        layer.W_in_ext.copy_(Win.float())
        layer.w_ext.copy_(w.float())
        layer.W_adj.copy_(Wadj.float())
        layer.W_out.copy_(Wout.float())
    return layer, X


def test_row_i_output_tracks_row_i_input():
    t, d = 12, 8
    layer, X = build(t, d)
    with torch.no_grad():
        base = layer(X.float())
        Xp = X.clone()
        Xp[5] *= 3.0                                     # scale one row's input pre-W_adj
        out = layer(Xp.float())
    # the change must be visible AT row 5 (own position) — not only at the last row
    assert not torch.allclose(out[5], base[5], atol=1e-6)
    # and rows <= 4 unaffected (causality) — except through their own positions
    assert torch.equal(out[:5], base[:5])


def test_negative_control_last_token_reading_differs():
    # A last-token-only adjustment (prose reading) would change ALL rows when
    # the last row's input changes; the equation reading changes only the last.
    t, d = 10, 8
    layer, X = build(t, d)
    with torch.no_grad():
        base = layer(X.float())
        Xp = X.clone()
        Xp[-1] *= 3.0
        out = layer(Xp.float())
    assert torch.equal(out[:-1], base[:-1])              # earlier rows untouched
    assert not torch.allclose(out[-1], base[-1], atol=1e-6)
