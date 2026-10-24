"""T15 — GPU smoke [gpu]. Correctness only (timing is not a pass criterion).

Runs the T1/T3/T6/T8 assertions at d=l=t=128 on one GPU. Skipped without
CUDA — the CPU suites carry the correctness gate; this records the GPU env
manifest when capacity allows.

First executed on 2026-10-23 when a free-tier T4 slot finally opened. It had
never run before that (every earlier session was CPU-only), and it caught two
defects in itself, both fixed here:
  * the layer was built in fp32 while the input was fp64 (`copy_` casts into
    the destination dtype) -> "expected mat1 and mat2 to have the same dtype";
  * the causality assertion compared rows [:65] after perturbing row 64, but
    row 64 is itself allowed to change -> the assertion was wrong, not the
    model. Rows < 64 are the ones that must stay bit-identical.
"""

import math

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("no GPU available (capacity-blocked; see handoff session 31)", allow_module_level=True)

from hextractor.harness import TrainConfig                        # noqa: E402
from hextractor.init import apply_init_policy                    # noqa: E402
from hextractor.incremental import ModelCache, model_forward_incremental  # noqa: E402
from hextractor.layers import HESublayer                         # noqa: E402
from hextractor.model import ExtractorTransformer                # noqa: E402
from hextractor.oracle import he_oracle                          # noqa: E402

from conftest import make_case                                   # noqa: E402

DEV = "cuda"


def _copy_weights(layer, Win, w, Wadj, Wout, dev):
    with torch.no_grad():
        layer.W_in_ext.copy_(Win.to(dev))
        layer.w_ext.copy_(w.to(dev))
        layer.W_adj.copy_(Wadj.to(dev))
        layer.W_out.copy_(Wout.to(dev))


@pytest.mark.parametrize("t,d", [(128, 128)])
def test_gpu_equation_and_causality(t, d):
    X, Win, w, Wadj, Wout = make_case(t, d, l=d, seed=77)
    oracle = he_oracle(X, Win, w, Wadj, Wout, dtype=torch.float64)

    # fp64 layer — the equation gate, same convention as the CPU suite.
    layer = HESublayer(d, d).to(DEV).double().eval()
    _copy_weights(layer, Win, w, Wadj, Wout, DEV)
    out = layer(X.to(DEV))
    assert out.device.type == "cuda"
    assert out.dtype == torch.float64
    assert torch.allclose(out.cpu(), oracle, atol=1e-5, rtol=0)

    # causality: perturbing rows >= 64 must leave rows < 64 bit-identical,
    # and MUST change row 64 (negative control, so the check cannot pass
    # vacuously).
    with torch.no_grad():
        base = layer(X.to(DEV))
        Xp = X.clone()
        Xp[64:] += 10.0
        out_p = layer(Xp.to(DEV))
    assert torch.equal(out_p[:64].cpu(), base[:64].cpu())
    assert not torch.equal(out_p[64].cpu(), base[64].cpu())

    # fp32 layer — the dtype training actually uses.
    layer32 = HESublayer(d, d).to(DEV).eval()
    _copy_weights(layer32, Win, w, Wadj, Wout, DEV)
    out32 = layer32(X.float().to(DEV))
    assert out32.dtype == torch.float32
    assert torch.allclose(out32.cpu().double(), oracle, atol=2e-3, rtol=5e-3)

    info = {"device": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "torch": torch.__version__}
    print(f"\n[gpu manifest] {info}")


@pytest.mark.parametrize("arm", ["HE", "attention-32h"])
def test_gpu_model_forward_and_incremental_agree(arm):
    """The full model and the cached incremental path agree on the GPU.

    This is the GPU half of the Milestone-3 exactness contract: the systems
    comparison is only meaningful if the incremental path decodes the same
    function on the device where the measurements are taken.
    """
    torch.manual_seed(4)
    model = ExtractorTransformer(arm=arm, d=32, l=32, n_layers=2, ffn_width=64,
                                 vocab=128, n_heads=4).to(DEV).eval()
    apply_init_policy(model, seed=4)
    ids = torch.randint(0, 128, (2, 32), device=DEV)
    with torch.no_grad():
        full = model(ids)
        cache = ModelCache.for_model(model)
        model_forward_incremental(model, ids[:, :16], cache)
        step = model_forward_incremental(model, ids[:, 16:17], cache)
    diff = (step[:, 0] - full[:, 16]).abs().max().item()
    scale = max(1.0, full[:, 16].abs().max().item())
    print(f"\n[gpu incremental] arm={arm} max_abs_diff={diff:.3e} scale={scale:.3e}")
    assert diff <= 1e-3 * scale


def test_quality_training_loop_really_trains_on_the_gpu(monkeypatch):
    """The device the runner is given must reach the training loop.

    Incident 2026-10-24: `run_quality.py --device cuda` advertised
    `device=cuda` in its environment manifest, but the harness built the
    model on the CPU and never moved it — the first "GPU" sweep trained on
    the CPU (14.3 s/step against 1.2 s/step for the same work in the same
    session). This test watches the device the forward actually sees.
    """
    import hextractor.harness as harness

    seen = []
    original = harness.build_model

    def spy(arm, cfg):
        model = original(arm, cfg)
        model.register_forward_pre_hook(
            lambda mod, inputs: seen.append(inputs[0].device))
        return model

    monkeypatch.setattr(harness, "build_model", spy)

    cfg = TrainConfig(d=32, l=32, n_layers=2, ffn_width=64, vocab=128,
                      n_heads=4, batch_size=4, eval_every=2, eval_batches=1)
    train = lambda: harness.synthetic_stream(cfg.vocab, cfg.l, cfg.batch_size, seed=0)
    held = lambda: harness.synthetic_stream(cfg.vocab, cfg.l, cfg.batch_size,
                                            seed=0, held_out=True)
    run = harness.train_one_seed("HE", cfg, seed=1, steps=2, train_stream=train,
                                  held_out_stream=held, device=torch.device(DEV))
    assert seen, "the model never ran"
    assert all(d.type == "cuda" for d in seen), \
        f"the training loop ran on {sorted({str(d) for d in seen})}, not on the GPU"
    assert run["final_held_out_loss"] is not None
    assert all(math.isfinite(v) for v in run["train_curve"])
