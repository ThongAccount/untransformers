"""Milestone 2 harness mechanics suites (T-m).

Validates mechanics only — forward/backward/step, shapes, determinism,
optimizer configuration, held-out isolation, dispersion statistics. No
quality verdicts anywhere: gate 0's line holds (the harness returns curves,
these tests assert the curves are correct data, not that any number is good).
"""

import torch

from hextractor.harness import (
    TrainConfig, build_model, build_optimizer, dispersion, held_out_loss,
    run_multi_seed, synthetic_stream, train_one_seed,
)
from hextractor.init import apply_init_policy

SMALL = TrainConfig(d=16, l=16, n_layers=2, ffn_width=32, vocab=64,
                    batch_size=8, eval_every=3, eval_batches=2, dropout=0.1,
                    n_heads=4)                      # 32 heads is a d=128 setting


def test_forward_backward_step_mechanics():
    cfg = SMALL
    run = train_one_seed("HE", cfg, seed=1, steps=4,
                         train_stream=lambda: synthetic_stream(cfg.vocab, cfg.l, cfg.batch_size, seed=0),
                         held_out_stream=lambda: synthetic_stream(cfg.vocab, cfg.l, cfg.batch_size, seed=0, held_out=True))
    curve = run["train_curve"]
    assert len(curve) == 4
    assert all(torch.isfinite(torch.tensor(v)) for v in curve)
    # a step must actually move parameters
    m = build_model("HE", cfg)
    apply_init_policy(m, seed=1)
    before = m.sub1[0].w_ext.detach().clone()
    opt = build_optimizer(m, cfg)
    x, y = next(synthetic_stream(cfg.vocab, cfg.l, cfg.batch_size, seed=0))
    loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, cfg.vocab), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    assert not torch.equal(before, m.sub1[0].w_ext.detach())


def test_output_shapes():
    cfg = SMALL
    m = apply_init_policy(build_model("attention-1h", cfg), seed=2)   # 1h divides any d
    x, y = next(synthetic_stream(cfg.vocab, cfg.l, cfg.batch_size, seed=0))
    logits = m(x)
    assert logits.shape == (cfg.batch_size, cfg.l, cfg.vocab)
    assert x.shape == (cfg.batch_size, cfg.l)
    assert y.shape == (cfg.batch_size, cfg.l)


def test_determinism_same_seed_identical_curve():
    cfg = SMALL
    ts = lambda: synthetic_stream(cfg.vocab, cfg.l, cfg.batch_size, seed=0)
    hs = lambda: synthetic_stream(cfg.vocab, cfg.l, cfg.batch_size, seed=0, held_out=True)
    a = train_one_seed("HE", cfg, seed=7, steps=5, train_stream=ts, held_out_stream=hs)
    b = train_one_seed("HE", cfg, seed=7, steps=5, train_stream=ts, held_out_stream=hs)
    assert a["train_curve"] == b["train_curve"]
    assert a["evals"] == b["evals"]


def test_different_seeds_differ():
    cfg = SMALL
    ts = lambda: synthetic_stream(cfg.vocab, cfg.l, cfg.batch_size, seed=0)
    hs = lambda: synthetic_stream(cfg.vocab, cfg.l, cfg.batch_size, seed=0, held_out=True)
    a = train_one_seed("HE", cfg, seed=7, steps=5, train_stream=ts, held_out_stream=hs)
    b = train_one_seed("HE", cfg, seed=8, steps=5, train_stream=ts, held_out_stream=hs)
    assert a["train_curve"] != b["train_curve"]


def test_optimizer_config_matches_paper_and_ledger():
    cfg = TrainConfig()                      # paper defaults
    m = build_model("HE", cfg)
    opt = build_optimizer(m, cfg)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.defaults["lr"] == 1e-3
    assert opt.defaults["betas"] == (0.9, 0.999)
    groups = {g["weight_decay"] for g in opt.param_groups}
    assert 1e-3 in groups                                # gamma lives in the groups
    assert groups == {1e-3, 0.0}                          # D-12 split present
    # LayerNorm gains/biases land in the no-decay group
    nd = {id(p) for g in opt.param_groups if g["weight_decay"] == 0.0 for p in g["params"]}
    assert id(m.norms1[0].weight) in nd and id(m.norms1[0].bias) in nd
    assert id(m.head.bias) in nd
    assert id(m.sub1[0].W_in_ext) not in nd


def test_held_out_stream_disjoint_from_train():
    train = synthetic_stream(64, 16, 8, seed=0)
    held = synthetic_stream(64, 16, 8, seed=0, held_out=True)
    tx, _ = next(train)
    hx, _ = next(held)
    assert not torch.equal(tx, hx)               # disjoint generator states
    # reproducibility of both streams
    assert torch.equal(next(synthetic_stream(64, 16, 8, seed=0))[0], tx)


def test_held_out_eval_does_not_mutate_or_select():
    cfg = SMALL
    m = apply_init_policy(build_model("HE", cfg), seed=2)   # params must be initialized
    hs = lambda: synthetic_stream(cfg.vocab, cfg.l, cfg.batch_size, seed=3, held_out=True)
    before = m.sub1[0].w_ext.detach().clone()
    was_training = m.training
    l1 = held_out_loss(m, hs, cfg)
    assert m.training == was_training                       # mode restored
    assert torch.equal(before, m.sub1[0].w_ext.detach())    # no param mutation
    l2 = held_out_loss(m, hs, cfg)
    assert l1 == l2                                         # deterministic eval


def test_multi_seed_dispersion_statistics():
    cfg = TrainConfig(d=16, l=16, n_layers=2, ffn_width=32, vocab=64,
                      batch_size=8, eval_every=2, eval_batches=2)
    out = run_multi_seed("HE", cfg, seeds=[1, 2], steps=4)
    assert out["arm"] == "HE" and len(out["runs"]) == 2
    disp = out["dispersion"]
    assert set(disp) == {2, 4}
    for p, stats in disp.items():
        assert stats["n"] == 2
        assert stats["std"] >= 0.0
        vals = [e["held_out_loss"] for r in out["runs"] for e in r["evals"] if e["step"] == p]
        assert abs(stats["mean"] - sum(vals) / 2) < 1e-12


def test_harness_returns_data_not_verdicts():
    cfg = SMALL
    run = train_one_seed("HE", cfg, seed=1, steps=2,
                         train_stream=lambda: synthetic_stream(cfg.vocab, cfg.l, cfg.batch_size, seed=0),
                         held_out_stream=lambda: synthetic_stream(cfg.vocab, cfg.l, cfg.batch_size, seed=0, held_out=True))
    assert set(run) == {"arm", "seed", "steps", "train_curve", "evals", "final_held_out_loss"}
