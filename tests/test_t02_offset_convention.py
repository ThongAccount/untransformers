"""T2 — Offset convention: k = i-j+1, k=1 = newest (catches the axis flip)."""

import torch

from hextractor.oracle import he_oracle
from hextractor.layers import HESublayer


NEUTRAL = 3   # channel carrying the constant 1 that neutralizes the adjustment


def offset_probe(t, d, k_star, channel):
    """Only offset k_star active: gate is a one-channel basis there, zero elsewhere.

    Channel `channel` carries distinct per-row values (identifies the source
    row); channel NEUTRAL carries the constant 1 and W_adj[NEUTRAL, channel]=1
    so the Eq. 24 Hadamard is neutral at the probed channel.
    """
    X = torch.zeros(t, d)
    X[:, channel] = torch.arange(1, t + 1, dtype=torch.float32)
    X[:, NEUTRAL] = 1.0
    W_in_ext = torch.eye(d)
    W_adj = torch.eye(d)
    W_adj[:, channel] = 0.0
    W_adj[NEUTRAL, channel] = 1.0
    w_ext = torch.zeros(t, d)
    w_ext[k_star - 1, channel] = 1.0                              # 1-based offset k_star
    W_out = torch.eye(d)
    return X, W_in_ext, w_ext, W_adj, W_out, channel


def test_offset_k1_is_newest():
    t, d, k_star, channel = 6, 4, 1, 2
    X, Win, w, Wadj, Wout, ch = offset_probe(t, d, k_star, channel)
    oracle = he_oracle(X, Win, w, Wadj, Wout)
    layer = HESublayer(d, t)
    with torch.no_grad():
        layer.W_in_ext.copy_(Win)
        layer.w_ext.copy_(w)
        layer.W_adj.copy_(Wadj)
        layer.W_out.copy_(Wout)
    impl = layer(X)
    assert torch.equal(impl, oracle.float())
    # k=1: out[i, ch] == projected value of row i itself
    assert torch.equal(impl[:, ch], X[:, ch])          # row i reads itself


def test_offset_k_general_and_zeros_before_k():
    t, d = 8, 4
    for k_star in (1, 2, t):
        channel = 1
        X, Win, w, Wadj, Wout, ch = offset_probe(t, d, k_star, channel)
        oracle = he_oracle(X, Win, w, Wadj, Wout)
        layer = HESublayer(d, t)
        with torch.no_grad():
            layer.W_in_ext.copy_(Win)
            layer.w_ext.copy_(w)
            layer.W_adj.copy_(Wadj)
            layer.W_out.copy_(Wout)
        impl = layer(X)
        assert torch.equal(impl, oracle.float())
        # rows i < k_star (1-based) get zero at that channel
        assert torch.equal(impl[: k_star - 1, ch], torch.zeros(k_star - 1, dtype=impl.dtype))
        # row i (0-based i >= k_star-1): value of row i-k_star+1's channel
        for i in range(k_star - 1, t):
            assert impl[i, ch].item() == X[i - k_star + 1, ch].item()
