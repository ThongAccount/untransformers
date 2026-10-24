"""The sublayer cache warm-up must not change what is measured.

The first GPU leg of the systems comparison died with a CUDA OOM in the
sublayer-1 scaling block: the eager attention path materialises the
`[H, m, T]` score matrix for the chunk it is handed, and at T = 8192 with
32 heads that is 8.6 GB per tensor (three alive at once) — more than a T4
has.

The fix is in the runner's **warm-up**, not in the measured implementation:
the cache is filled in pieces. The warm-up is not timed (the measured call is
a single new token at position T-1), and the cache that comes out must be
*identical* — a token's K and V do not depend on how the warm-up was cut, and
the HE state keeps the last l-1 vectors either way.

These tests pin that equivalence, so the fix cannot silently change the
measurement it was made to enable.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "experiments"))

import run_systems as rs  # noqa: E402
from hextractor.harness import TrainConfig  # noqa: E402
from hextractor.incremental import AttentionState, HEState  # noqa: E402


def _cfg():
    return TrainConfig(d=32, l=8, n_layers=1, ffn_width=64, vocab=100, n_heads=4)


def _sub(arm, cfg):
    torch.manual_seed(7)
    sub = rs._make_sublayer(arm, cfg)
    with torch.no_grad():
        for p in sub.parameters():
            p.copy_(torch.randn_like(p) * 0.01)
    sub.eval()
    return sub


def _same_state(a, b):
    if isinstance(a, HEState):
        assert a.m == b.m, "HE holds a different number of vectors"
        if a.buf is None:
            assert b.buf is None
        else:
            assert torch.equal(a.buf, b.buf), "HE's rolling state differs"
    else:
        assert torch.equal(a.k, b.k), "the K cache differs"
        assert torch.equal(a.v, b.v), "the V cache differs"


def test_warming_the_cache_in_pieces_gives_the_same_cache_for_every_arm():
    cfg = _cfg()
    T = 40
    chunk = torch.randn(1, T - 1, cfg.d)
    for arm in ("HE", "attention-32h", "attention-32h-sdpa"):
        sub = _sub(arm, cfg)
        one = HEState() if arm == "HE" else AttentionState()
        rs._sublayer_incremental(arm, sub, chunk, one, 0)

        pieces = HEState() if arm == "HE" else AttentionState()
        rs._warm_cache(arm, sub, chunk, pieces, piece=7)

        _same_state(one, pieces)


def test_the_measured_call_is_identical_after_either_warm_up():
    """The timed call is a single new token: the two warm-ups must give it the
    same state and therefore the same output."""
    cfg = _cfg()
    T = 40
    chunk = torch.randn(1, T - 1, cfg.d)
    x = torch.randn(1, 1, cfg.d)
    for arm in ("HE", "attention-32h", "attention-32h-sdpa"):
        sub = _sub(arm, cfg)
        one = HEState() if arm == "HE" else AttentionState()
        rs._sublayer_incremental(arm, sub, chunk, one, 0)
        out_one = rs._sublayer_incremental(arm, sub, x, one, T - 1)

        pieces = HEState() if arm == "HE" else AttentionState()
        rs._warm_cache(arm, sub, chunk, pieces, piece=7)
        out_pieces = rs._sublayer_incremental(arm, sub, x, pieces, T - 1)

        assert torch.allclose(out_one, out_pieces, atol=0, rtol=0), \
            f"{arm}: the measured call differs after the two warm-ups"


def test_the_warm_up_is_not_timed():
    """The warm-up happens before the timing loop; the runner must not call
    the sublayer inside the timed region."""
    src = (ROOT / "experiments" / "run_systems.py").read_text(encoding="utf-8")
    body = src.split("def measure_sublayer_scaling", 1)[1]
    warm_at = body.index("_warm_cache(")
    timing_at = body.index("t0 = time.perf_counter()")
    assert warm_at < timing_at, "the warm-up must happen before the timing loop"
    timed_region = body[timing_at:timing_at + 400]
    assert "_warm_cache(" not in timed_region, \
        "the cache warm-up must not be inside the timed region"
