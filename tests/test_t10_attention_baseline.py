"""T10 — Attention baseline reference: Eqs. 10-17 vs oracle and vs SDPA."""

import torch
import torch.nn.functional as F

from hextractor.layers import AttentionSublayer
from hextractor.oracle import attention_oracle

from conftest import make_case


def build(t, d, n_heads, seed=21):
    X, Win, _, Wadj, _ = make_case(t, d, l=d, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    Wq = torch.randn(d, d, generator=g) * 0.5
    Wk = torch.randn(d, d, generator=g) * 0.5
    Wv = torch.randn(d, d, generator=g) * 0.5
    Wo = torch.randn(d, d, generator=g) * 0.5
    layer = AttentionSublayer(d, n_heads).eval()
    with torch.no_grad():
        layer.Wq.copy_(Wq.float())
        layer.Wk.copy_(Wk.float())
        layer.Wv.copy_(Wv.float())
        layer.Wo.copy_(Wo.float())
    return layer, X, (Wq, Wk, Wv, Wo)


def test_matches_oracle_and_sdpa():
    t, d = 16, 32
    layer, X, (Wq, Wk, Wv, Wo) = build(t, d, n_heads=4)
    oracle = attention_oracle(X, Wq, Wk, Wv, Wo, n_heads=4)
    impl = layer(X.float())
    scale = oracle.float().abs().max().item()
    assert (impl - oracle.float()).abs().max().item() <= 1e-5 * scale

    # cross-check against PyTorch's SDPA (is_causal=True)
    h = d // 4
    q = (X.float() @ Wq.float()).view(t, 4, h).transpose(0, 1)
    k = (X.float() @ Wk.float()).view(t, 4, h).transpose(0, 1)
    v = (X.float() @ Wv.float()).view(t, 4, h).transpose(0, 1)
    sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    sdpa = sdpa.transpose(0, 1).flatten(1) @ Wo.float()
    assert torch.allclose(impl, sdpa, atol=1e-4, rtol=0)


def test_mask_boundary_row1_attends_to_itself():
    t, d = 8, 16
    layer, X, (Wq, Wk, Wv, Wo) = build(t, d, n_heads=2)
    with torch.no_grad():
        base = layer(X.float())
        Xp = X.clone()
        Xp[1:] += 50.0
        out = layer(Xp.float())
    assert torch.equal(out[0], base[0])                  # row 1 sees only itself


def test_head_count_param_independence():
    d = 128
    from hextractor.layers import AttentionSublayer as A
    assert A(d, 1).n_params() == A(d, 32).n_params() == 65536


def test_negative_control_broken_mask_reddens():
    t, d = 8, 16
    layer, X, _ = build(t, d, n_heads=2)
    with torch.no_grad():
        base = layer(X.float())
    # a non-causal implementation would change row 0 when later rows change
    Xp = X.clone()
    Xp[3] += 7.0
    with torch.no_grad():
        out = layer(Xp.float())
    assert torch.equal(out[0], base[0])                  # still causal
    assert not torch.allclose(out[3], base[3], atol=1e-6)
