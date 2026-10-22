import pytest
import torch

L = D = 128


def make_case(t, d, l=None, seed=0, std=0.5, dtype=torch.float64):
    """Seeded random case: (X, W_in_ext, w_ext, W_adj, W_out) at fp64."""
    l = l or d
    g = torch.Generator().manual_seed(seed)
    def r(*shape):
        return (torch.randn(*shape, generator=g) * std).to(dtype)
    X = r(t, d)
    W_in_ext = r(d, d)
    w_ext = r(l, d)
    W_adj = r(d, d)
    W_out = r(d, d)
    return X, W_in_ext, w_ext, W_adj, W_out


@pytest.fixture(scope="module")
def full_case():
    return make_case(t=L, d=D, seed=11)
