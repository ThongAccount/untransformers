"""T12 — End-to-end determinism (eval bit-identical; train dropout live)."""

import torch

from hextractor.init import apply_init_policy
from hextractor.model import ExtractorTransformer


def small(arm):
    m = ExtractorTransformer(arm=arm, d=32, l=32, n_layers=2, ffn_width=64, vocab=100, dropout=0.1)
    return apply_init_policy(m, seed=5)


def test_eval_bit_identical():
    for arm in ("HE", "attention-32h"):
        m = small(arm).eval()
        ids = torch.randint(0, 100, (2, 16))
        with torch.no_grad():
            a, b = m(ids), m(ids)
        assert torch.equal(a, b), arm


def test_train_dropout_seed_behavior():
    m = small("HE").train()
    ids = torch.randint(0, 100, (2, 16))
    torch.manual_seed(11)
    a = m(ids)
    torch.manual_seed(11)
    b = m(ids)
    assert torch.equal(a, b)                             # same seed -> identical
    torch.manual_seed(12)
    c = m(ids)
    assert not torch.equal(a, c)                         # different seed -> differs (dropout live)


def test_negative_control_dropout_off_would_be_silent():
    m = small("HE").train()
    m.drop.p = 0.0
    for ffn in m.ffn:
        pass
    ids = torch.randint(0, 100, (2, 16))
    torch.manual_seed(1)
    a = m(ids)
    torch.manual_seed(2)
    b = m(ids)
    assert torch.equal(a, b)                             # with dropout off, seeds agree
