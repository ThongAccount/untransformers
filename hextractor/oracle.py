"""Literal double-loop oracle for the paper's equations — the only accepted
reference (HE_SPEC §8).

Deliberately naive: explicit (i, j) loops, no vectorization tricks. fp64 by
default so gradcheck-level comparisons are meaningful.
"""

import math

import torch


def he_oracle(X, W_in_ext, w_ext, W_adj, W_out=None, gate="vector",
              dtype=torch.float64):
    """HE sublayer oracle: Eqs. 27 -> 28 -> 24 -> 25, literal loops.

    Args:
        X:        [t, d] input to sublayer-1 (post-LayerNorm).
        W_in_ext: [d, d] shared projection (None for WE/ME/SHE variants).
        w_ext:    gates — "vector" [l, d] (offset axis FIRST, w_ext[0] is the
                  newest-token gate), "scalar" [l] (ME), "matrix" [l, d, d] (SHE).
        W_adj:    [d, d] adjustment projection (None for ME).
        W_out:    [d, d] output projection (None = Eq. 25 disabled).
    Returns: [t, d].

    Offset convention (HE_SPEC §3): k = i - j + 1, so k=1 <-> j=i (newest).
    Sum bounds j = 1..i: strictly causal, no padding.
    """
    X = X.to(dtype)
    if W_in_ext is not None:
        W_in_ext = W_in_ext.to(dtype)
    w_ext = w_ext.to(dtype)
    if W_adj is not None:
        W_adj = W_adj.to(dtype)
    if W_out is not None:
        W_out = W_out.to(dtype)

    t, d = X.shape
    Xe = X @ W_in_ext if W_in_ext is not None else X          # Eq. 27 (or identity for ME)

    out_ext = torch.zeros(t, d, dtype=dtype)
    for i in range(t):
        acc = torch.zeros(d, dtype=dtype)
        for j in range(i + 1):                                # j = 1..i (1-based) -> 0-based 0..i
            k = i - j + 1                                     # 1-based offset, k=1 newest
            if gate == "vector":
                acc = acc + Xe[j] * w_ext[k - 1]
            elif gate == "scalar":
                acc = acc + Xe[j] * w_ext[k - 1]
            elif gate == "matrix":
                acc = acc + (w_ext[k - 1] @ Xe[j])
            else:
                raise ValueError(f"unknown gate mode {gate!r}")
        out_ext[i] = acc                                      # Eq. 28

    if W_adj is not None:
        out = (X @ W_adj) * out_ext                           # Eq. 24
    else:
        out = out_ext                                         # ME: extraction only

    if W_out is not None:
        out = out @ W_out                                     # Eq. 25
    return out


def attention_oracle(X, Wq, Wk, Wv, Wo, n_heads, dtype=torch.float64):
    """Multi-head self-attention oracle, paper Eqs. 10-17 (double loops).

    Eq. 14 causal mask (row i attends to j <= i), scaling 1/sqrt(h) with
    h = d / n (Eq. 16), concat heads then W^out_att (Eq. 17), no biases.
    """
    X = X.to(dtype)
    Wq, Wk, Wv, Wo = (w.to(dtype) for w in (Wq, Wk, Wv, Wo))
    t, d = X.shape
    h = d // n_heads
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    ctx = torch.zeros(t, d, dtype=dtype)
    for i in range(t):
        for hd in range(n_heads):
            sl = slice(hd * h, (hd + 1) * h)
            q = Q[i, sl]
            scores = [(q @ K[j, sl]) / math.sqrt(h) for j in range(i + 1)]  # Eq. 14+15
            m = max(scores)
            exps = [math.exp(s - m) for s in scores]
            z = sum(exps)
            acc = torch.zeros(h, dtype=dtype)
            for j in range(i + 1):
                acc = acc + (exps[j] / z) * V[j, sl]
            ctx[i, sl] = acc
    return ctx @ Wo                                            # Eq. 17
