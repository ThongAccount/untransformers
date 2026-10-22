"""T1 — Equation-level reference check: vectorized HE vs the double-loop oracle."""

import torch

from hextractor.oracle import he_oracle
from hextractor.layers import HESublayer

from conftest import make_case


def build(layer, W_in_ext, w_ext, W_adj, W_out):
    with torch.no_grad():
        if layer.use_shared_projection:
            layer.W_in_ext.copy_(W_in_ext.float())
        layer.w_ext.copy_(w_ext.float())
        if layer.use_adjustment:
            layer.W_adj.copy_(W_adj.float())
        if layer.use_output_projection:
            layer.W_out.copy_(W_out.float())
    return layer


def run_case(t, d, use_out, seed):
    X, W_in_ext, w_ext, W_adj, W_out = make_case(t, d, l=d, seed=seed)
    oracle = he_oracle(X, W_in_ext, w_ext, W_adj, W_out if use_out else None)
    layer = build(HESublayer(d, d, variant="HE"), W_in_ext, w_ext, W_adj, W_out)
    if not use_out:
        layer.W_out = None
        layer.use_output_projection = False
    impl = layer(X.float())
    return oracle, impl


def rel_close(impl, oracle, rel=1e-5):
    """HE_SPEC §8: max abs diff <= 1e-5 RELATIVE to the oracle scale (fp32)."""
    scale = oracle.float().abs().max().item()
    return (impl - oracle.float()).abs().max().item() <= rel * scale


def test_matches_oracle_at_full_size():
    oracle, impl = run_case(t=128, d=128, use_out=True, seed=1)
    assert rel_close(impl, oracle)


def test_matches_oracle_small_shapes_and_wout_off():
    # t values are relative to l (spec T1): {1, 2, 3, l-1, l}; l = d here.
    for d in (8, 128):
        for t in (1, 2, 3, d - 1, d):
            for use_out in (True, False):
                oracle, impl = run_case(t, d, use_out, seed=t * 100 + d + int(use_out))
                assert rel_close(impl, oracle), (t, d, use_out)


def test_negative_control_axis_flip_reddens():
    # The offset-axis flip is THE bug class: flipping the gate axis must fail.
    X, W_in_ext, w_ext, W_adj, W_out = make_case(16, 16, l=16, seed=5)
    oracle = he_oracle(X, W_in_ext, w_ext, W_adj, W_out)
    layer = build(HESublayer(16, 16), W_in_ext, w_ext, W_adj, W_out)
    with torch.no_grad():
        layer.w_ext.copy_(w_ext.flip(0).float())
    flipped = layer(X.float())
    assert not rel_close(flipped, oracle)
