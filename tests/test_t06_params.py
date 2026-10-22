"""T6 — Parameter count: the paper's headline parity claim.

At d=l=128: HE (W^out ON) == 65,536 == attention-32h == 4d^2. Without W^out
HE has 49,152 = 0.75x — the trap this test exists to keep visible.
"""

import torch

from hextractor.layers import HESublayer, AttentionSublayer


def test_headline_counts_at_d_l_128():
    d = l = 128
    he = HESublayer(d, l, variant="HE").n_params()
    att32 = AttentionSublayer(d, n_heads=32).n_params()
    att1 = AttentionSublayer(d, n_heads=1).n_params()
    assert he == 65536
    assert att32 == 65536
    assert att1 == 65536


def test_wout_off_is_the_papers_trap():
    d = l = 128
    he = HESublayer(d, l, variant="HE")
    with torch.no_grad():
        he.W_out = None
        he.use_output_projection = False
    assert he.n_params() == 49152
    assert he.n_params() != 65536                      # "same param count" would be false


def test_variant_formulas():
    for d, l in ((128, 128), (64, 32), (16, 8)):
        assert HESublayer(d, l, "HE").n_params() == 3 * d * d + l * d
        assert HESublayer(d, l, "WE").n_params() == l * d + 2 * d * d
        assert HESublayer(d, l, "ME").n_params() == l
        assert HESublayer(d, l, "SHE").n_params() == l * d * d + 2 * d * d


def test_we_me_counts_match_paper_table7():
    d = l = 128
    assert HESublayer(d, l, "WE").n_params() == 49152
    assert HESublayer(d, l, "ME").n_params() == 128
    assert HESublayer(d, l, "SHE").n_params() == 2129920
