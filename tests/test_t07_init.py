"""T7 — Initialization policy (HE_SPEC §7) and cross-arm reproducibility."""

import math

import torch

from hextractor.init import apply_init_policy
from hextractor.model import ExtractorTransformer


def shared_tensors(model):
    return {n: p for n, p in model.named_parameters() if not n.startswith("sub1.")}


def test_init_statistics_and_structure():
    torch.manual_seed(0)
    model = ExtractorTransformer(arm="HE", d=32, l=32, n_layers=2, ffn_width=64, vocab=100)
    apply_init_policy(model, seed=123)
    with torch.no_grad():
        # weight matrices ~ N(0, 0.01)
        w = model.sub1[0].W_in_ext
        assert abs(w.mean().item()) < 3 * 0.01 / math.sqrt(w.numel())
        assert abs(w.std().item() - 0.01) < 0.01 * 0.5 + 1e-9
        # biases zero
        assert model.head.bias.abs().sum().item() == 0.0
        for m in model.ffn:
            assert m[0].bias.abs().sum().item() == 0.0
        # LayerNorm gain 1 / bias 0
        assert model.norms1[0].weight.abs().sub(1).sum().item() == 0.0
        assert model.norms1[0].bias.abs().sum().item() == 0.0
        # embeddings == raw N(0,0.01) * sqrt(d)
        std_expected = 0.01 * math.sqrt(model.d)
        assert abs(model.tok_embed.std().item() - std_expected) < std_expected * 0.5 + 1e-9


def test_cross_arm_shared_init_identical():
    m_he = ExtractorTransformer(arm="HE", d=32, l=32, n_layers=2, ffn_width=64, vocab=100)
    m_at = ExtractorTransformer(arm="attention-32h", d=32, l=32, n_layers=2, ffn_width=64, vocab=100)
    apply_init_policy(m_he, seed=42)
    apply_init_policy(m_at, seed=42)
    sh_he, sh_at = shared_tensors(m_he), shared_tensors(m_at)
    assert sh_he.keys() == sh_at.keys()
    for k in sh_he:
        assert torch.equal(sh_he[k], sh_at[k]), k


def test_negative_control_seed_changes_init():
    m1 = ExtractorTransformer(arm="HE", d=32, l=32, n_layers=2, ffn_width=64, vocab=100)
    m2 = ExtractorTransformer(arm="HE", d=32, l=32, n_layers=2, ffn_width=64, vocab=100)
    apply_init_policy(m1, seed=1)
    apply_init_policy(m2, seed=2)
    assert not torch.equal(m1.sub1[0].W_in_ext, m2.sub1[0].W_in_ext)
