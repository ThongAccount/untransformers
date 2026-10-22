"""T15 — GPU smoke [gpu]. Correctness only (timing is not a pass criterion).

Runs the T1/T3/T6/T8 assertions at d=l=t=128 on one GPU. Skipped without
CUDA — the CPU suites carry the correctness gate; this records the GPU env
manifest when capacity allows (currently free-tier-blocked, see session log).
"""

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("no GPU available (capacity-blocked; see handoff session 31)", allow_module_level=True)

from hextractor.layers import HESublayer          # noqa: E402
from hextractor.oracle import he_oracle            # noqa: E402

from conftest import make_case                    # noqa: E402


@pytest.mark.parametrize("t,d", [(128, 128)])
def test_gpu_equation_and_causality(t, d):
    dev = "cuda"
    X, Win, w, Wadj, Wout = make_case(t, d, l=d, seed=77)
    oracle = he_oracle(X, Win, w, Wadj, Wout, dtype=torch.float64)
    layer = HESublayer(d, d).to(dev).eval()
    with torch.no_grad():
        layer.W_in_ext.copy_(Win.to(dev))
        layer.w_ext.copy_(w.to(dev))
        layer.W_adj.copy_(Wadj.to(dev))
        layer.W_out.copy_(Wout.to(dev))
    out = layer(X.to(dev))
    assert out.device.type == "cuda"
    assert torch.allclose(out.cpu(), oracle.float(), atol=1e-5, rtol=0)

    with torch.no_grad():
        base = layer(X.to(dev))
        Xp = X.clone()
        Xp[64:] += 10.0
        out_p = layer(Xp.to(dev))
    assert torch.equal(out_p[:65].cpu(), base[:65].cpu())

    info = {"device": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "torch": torch.__version__}
    print(f"\n[gpu manifest] {info}")
