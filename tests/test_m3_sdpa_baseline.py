"""The SDPA systems baseline must be the same function as the reference arm.

`attention-32h-sdpa` exists only as the Milestone-3 "responsibly optimized
attention baseline": if it were not equivalent to `attention-32h`, every
systems number that includes it would be comparing two different models
instead of two implementations of one model.

These checks are the pre-condition for quoting SDPA numbers at all:

  1. identical init — the shared parameters are drawn first and the
     sublayer-1 parameters have the same names, shapes and draw order, so
     the two arms start bit-identical under the same seed;
  2. identical forward — same logits within fp32 tolerance.
"""

import torch

from hextractor.harness import TrainConfig, build_model
from hextractor.init import apply_init_policy


def _cfg():
    return TrainConfig(d=32, l=32, n_layers=2, ffn_width=64, vocab=128,
                       batch_size=4, n_heads=32)


def test_sdpa_arm_starts_bit_identical_to_reference():
    cfg = _cfg()
    ref = build_model("attention-32h", cfg)
    sdpa = build_model("attention-32h-sdpa", cfg)
    apply_init_policy(ref, seed=11)
    apply_init_policy(sdpa, seed=11)
    r = dict(ref.named_parameters())
    s = dict(sdpa.named_parameters())
    assert set(r) == set(s), "parameter names differ between the two attention arms"
    for name in r:
        assert r[name].shape == s[name].shape, f"{name}: shape differs"
        assert torch.equal(r[name], s[name]), f"{name}: initial values differ"
    assert sum(p.numel() for p in ref.parameters()) == sum(p.numel() for p in sdpa.parameters())


def test_sdpa_arm_forward_matches_reference():
    cfg = _cfg()
    ref = build_model("attention-32h", cfg).eval()
    sdpa = build_model("attention-32h-sdpa", cfg).eval()
    apply_init_policy(ref, seed=11)
    apply_init_policy(sdpa, seed=11)
    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.l))
    with torch.no_grad():
        a = ref(ids)
        b = sdpa(ids)
    assert a.shape == b.shape
    diff = (a - b).abs().max().item()
    scale = max(1.0, a.abs().max().item())
    assert diff <= 1e-5 * scale, f"max abs diff {diff} (scale {scale})"


def test_sdpa_arm_is_causal():
    cfg = _cfg()
    model = build_model("attention-32h-sdpa", cfg).eval()
    apply_init_policy(model, seed=11)
    torch.manual_seed(5)
    ids = torch.randint(0, cfg.vocab, (1, 16))
    with torch.no_grad():
        base = model(ids)
        perturbed = ids.clone()
        perturbed[:, 8:] = (perturbed[:, 8:] + 1) % cfg.vocab
        out = model(perturbed)
    assert torch.equal(out[:, :8], base[:, :8])
    assert not torch.equal(out[:, 8:], base[:, 8:])
