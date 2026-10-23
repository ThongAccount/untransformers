"""Milestone 3 — incremental decoding exactness (T-m3).

The systems comparison is only meaningful if both arms decode the *same
function* as their full forward pass. These suites assert that contract
(fp32, eval mode, dropout disabled), plus the state-size property the
comparison rests on: attention's cache grows with context, HE's rolling state
does not.

No performance verdicts here — this file is correctness only.
"""

import pytest
import torch

from hextractor.init import apply_init_policy
from hextractor.incremental import (
    AttentionState, HEState, ModelCache, attention_forward_incremental,
    he_forward_incremental, model_forward_incremental,
)
from hextractor.layers import AttentionSublayer, HESublayer
from hextractor.model import ExtractorTransformer

SMALL = dict(d=16, l=16, n_layers=2, ffn_width=32, vocab=64, n_heads=4, dropout=0.1)


def _model(arm, seed=3, **kw):
    cfg = dict(SMALL)
    cfg.update(kw)
    torch.manual_seed(seed)
    m = ExtractorTransformer(arm=arm, **cfg)
    apply_init_policy(m, seed=seed)
    m.eval()                     # dropout off -> exactness is meaningful
    return m


def _ids(batch=2, t=16, vocab=64, seed=5):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, t), generator=g)


@pytest.mark.parametrize("arm", ["HE", "WE", "ME", "attention-1h"])
def test_incremental_matches_full_forward(arm):
    """Prefill + single-token steps reproduce the full forward exactly."""
    m = _model(arm)
    ids = _ids()
    with torch.no_grad():
        full = m(ids)

    cache = ModelCache.for_model(m)
    prefill = 7
    with torch.no_grad():
        model_forward_incremental(m, ids[:, :prefill], cache)
        for t in range(prefill, ids.shape[1]):
            step = model_forward_incremental(m, ids[:, t:t + 1], cache)
            diff = (step[:, 0] - full[:, t]).abs().max().item()
            assert diff <= 1e-5, f"{arm}: step {t} max abs diff {diff}"


def test_incremental_matches_full_forward_32h_at_paper_width():
    """The actual comparison arm (32 heads) at the paper's d=128."""
    m = _model("attention-32h", d=128, l=128, ffn_width=512, vocab=5000)
    ids = _ids(batch=1, t=32, vocab=5000)
    with torch.no_grad():
        full = m(ids)
    cache = ModelCache.for_model(m)
    with torch.no_grad():
        model_forward_incremental(m, ids[:, :16], cache)
        for t in range(16, ids.shape[1]):
            step = model_forward_incremental(m, ids[:, t:t + 1], cache)
            assert (step[:, 0] - full[:, t]).abs().max().item() <= 1e-5


def test_chunked_prefill_is_exact():
    """A multi-token chunk after a prefill stays exact (causal mask inside chunk)."""
    m = _model("attention-1h")
    ids = _ids(t=16)
    with torch.no_grad():
        full = m(ids)
    cache = ModelCache.for_model(m)
    with torch.no_grad():
        model_forward_incremental(m, ids[:, :6], cache)
        chunk = model_forward_incremental(m, ids[:, 6:10], cache)
        for j in range(chunk.shape[1]):
            assert (chunk[:, j] - full[:, 6 + j]).abs().max().item() <= 1e-5


def test_negative_control_state_is_actually_used():
    """If the rolling state were ignored, the exactness tests would be vacuous.
    Corrupting it must change the output."""
    torch.manual_seed(7)
    sub = HESublayer(d=8, l=4, variant="HE")
    with torch.no_grad():
        for p in sub.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
    X = torch.randn(1, 1, 8)
    state = HEState()
    # fill the buffer with the last l-1 src vectors of a prefix
    prefix = torch.randn(1, 3, 8)
    src = prefix @ sub.W_in_ext
    state.buf = src
    state.m = 3
    clean = he_forward_incremental(sub, X, state, start=3)
    state.buf = state.buf + 1.0
    dirty = he_forward_incremental(sub, X, state, start=3)
    assert not torch.allclose(clean, dirty)


def test_attention_cache_is_used():
    """Same control for the KV cache."""
    torch.manual_seed(7)
    sub = AttentionSublayer(d=8, n_heads=2)
    with torch.no_grad():
        for p in sub.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
    X = torch.randn(1, 1, 8)
    state = AttentionState()
    state.k = torch.randn(1, 2, 3, 4)
    state.v = torch.randn(1, 2, 3, 4)
    clean = attention_forward_incremental(sub, X, state, start=3)
    state.v = state.v + 1.0
    dirty = attention_forward_incremental(sub, X, state, start=3)
    assert not torch.allclose(clean, dirty)


def test_state_size_grows_for_attention_saturates_for_he():
    """Attention's cache grows linearly with context; HE's rolling state
    saturates at l-1 vectors per layer and never grows past it."""
    m_he = _model("HE")
    m_at = _model("attention-1h")
    sizes_he, sizes_at = [], []
    for t in (4, 12, 15, 16):
        ids = _ids(batch=1, t=t)
        for m, bucket in ((m_he, sizes_he), (m_at, sizes_at)):
            cache = ModelCache.for_model(m)
            with torch.no_grad():
                model_forward_incremental(m, ids, cache)
            bucket.append(cache.state_bytes())
    # HE: saturates once the context reaches l-1 = 15 vectors per layer
    assert sizes_he[-1] == sizes_he[-2], "HE state must be flat past l-1"
    assert sizes_he[-1] == (16 - 1) * 16 * 4 * 2      # (l-1) * d * 4B * layers
    # attention: strictly growing with context
    assert all(b > a for a, b in zip(sizes_at, sizes_at[1:]))
    assert sizes_at[-1] > sizes_he[-1]


def test_incremental_is_causal():
    """Future tokens must not change earlier decoded logits."""
    m = _model("HE")
    ids = _ids(t=16)
    other = ids.clone()
    other[:, 10:] = (other[:, 10:] + 7) % 64
    cache_a, cache_b = ModelCache.for_model(m), ModelCache.for_model(m)
    with torch.no_grad():
        la = model_forward_incremental(m, ids[:, :12], cache_a)
        lb = model_forward_incremental(m, other[:, :12], cache_b)
    # positions 0..9 are decoded before the divergence and must be identical
    assert torch.equal(la[:, :10], lb[:, :10])


def test_model_context_cap_is_enforced():
    m = _model("HE")
    cache = ModelCache.for_model(m)
    ids = _ids(t=16)
    with torch.no_grad():
        model_forward_incremental(m, ids[:, :16], cache)
    with pytest.raises(AssertionError):
        model_forward_incremental(m, _ids(t=1), cache)
