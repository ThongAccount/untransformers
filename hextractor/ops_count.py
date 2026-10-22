"""Analytic operation counter (HE_SPEC §9 convention).

Per sublayer at fixed t: matmul [t,d]x[d,d] = t*d*d multiplications and
t*d*(d-1) additions; extraction = t(t+1)/2*d multiplications and t(t-1)/2*d
additions; Hadamard products = t*d multiplications. This convention
reproduces the paper's Table-7 HE/WE rows exactly (7,364,608 / 14,647,296
and 5,267,456 / 10,469,376 at t=d=l=128); the attention rows do not
reconcile under any obvious mapping — reported as a delta, not forced (T13).
"""


def count_ops_fwd(variant, t, d, l):
    """Forward-only (mults, adds) for one sublayer under HE_SPEC §9."""
    mults = adds = 0
    # extraction (all Extractor variants)
    mults += t * (t + 1) // 2 * d
    adds += t * (t - 1) // 2 * d
    if variant == "HE":
        mults += t * d * d            # in_ext projection (Eq. 27)
        adds += t * d * (d - 1)
    if variant in ("HE", "WE", "SHE"):
        mults += t * d                # adjustment Hadamard (Eq. 24)
        mults += t * d * d            # adjustment projection
        adds += t * d * (d - 1)
        mults += t * d * d            # output projection (Eq. 25)
        adds += t * d * (d - 1)
    return mults, adds


def count_ops_attention_fwd(t, d, l, n_heads):
    """Attention forward ops under the same convention (ours, not the paper's)."""
    h = d // n_heads
    mults = 4 * t * d * d                       # Q,K,V,O projections
    adds = 4 * t * d * (d - 1)
    mults += t * (t + 1) // 2 * d               # q.k scores (t(t+1)/2 pairs x d)
    adds += t * (t + 1) // 2 * (d - 1) + t * (t + 1) // 2  # dot + softmax exps
    mults += t * (t + 1) // 2 * d               # attn.v
    adds += t * (t + 1) // 2 * (d - 1)
    return mults, adds


def table7_counts(t=128, d=128, l=128):
    """Paper Table-7 numbers this convention must reproduce."""
    he_m, he_a = count_ops_fwd("HE", t, d, l)
    we_m, we_a = count_ops_fwd("WE", t, d, l)
    att_m, att_a = count_ops_attention_fwd(t, d, l, 32)
    return {
        "HE_mults": he_m,
        "HE_total": he_m + he_a,            # paper's "total ops" = mults + adds
        "WE_mults": we_m,
        "WE_total": we_m + we_a,
        "attention32_mults_ours": att_m,
        "attention32_total_ours": att_m + att_a,
        "paper_HE_mults": 7364608,
        "paper_HE_total": 14647296,
        "paper_WE_mults": 5267456,
        "paper_WE_total": 10469376,
        "paper_attention32_total": 21710848,
    }
