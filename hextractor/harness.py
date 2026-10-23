"""Milestone 2 harness — training loop, eval procedure, multi-seed protocol.

Mechanics only. This module produces NUMBERS (loss curves, dispersion stats)
and never interprets them; gate 0's line — no result interpretation before
the corpus and compute plan land — holds at the API level: run functions
return plain data, no pass/fail verdicts on quality.

Paper setup (REPRODUCIBILITY.md §2): AdamW lr=0.001, betas (0.9, 0.999),
weight decay gamma=0.001, dropout p=0.1, batch 64, l=128, d=128,
18 layers, FFN 512, vocab 5000. Open hyperparameters are carried in the
deviation ledger with defaults (docs/milestone-1/DEVIATIONS.md):
D-12 weight decay excludes LayerNorm/biases (default: standard practice),
D-13 gradient clipping OFF (paper silent).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .init import apply_init_policy
from .model import ExtractorTransformer


@dataclass(frozen=True)
class TrainConfig:
    # paper setup
    d: int = 128
    l: int = 128
    n_layers: int = 18
    ffn_width: int = 512
    vocab: int = 5000
    n_heads: int = 32
    batch_size: int = 64
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-3           # gamma
    eps: float = 1e-8
    dropout: float = 0.1
    # protocol
    eval_every: int = 100                # steps between held-out evals
    eval_batches: int = 8                # held-out batches per eval point
    # deviation-ledger defaults
    clip_grad: Optional[float] = None    # D-13: off
    weight_decay_no_norm: bool = True    # D-12: exclude LayerNorm/bias


def build_model(arm: str, cfg: TrainConfig) -> ExtractorTransformer:
    return ExtractorTransformer(
        arm=arm, d=cfg.d, l=cfg.l, n_layers=cfg.n_layers,
        ffn_width=cfg.ffn_width, vocab=cfg.vocab, n_heads=cfg.n_heads,
        dropout=cfg.dropout)


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW per the paper; D-12 default excludes LayerNorm gains/biases and
    all biases from weight decay."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if cfg.weight_decay_no_norm and (p.ndim <= 1 or "norm" in name):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)


BatchStream = Callable[[], Iterator[Tuple[torch.Tensor, torch.Tensor]]]


def synthetic_stream(vocab: int, l: int, batch: int, seed: int,
                     held_out: bool = False) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Deterministic synthetic token stream.

    Deterministic per (seed, held_out): the held-out stream is drawn from a
    disjoint generator state, so held-out token ids are never seen in
    training (T14's manifest replaces this provider when the corpus lands).
    """
    g = torch.Generator().manual_seed(seed + (10_000 if held_out else 0))
    while True:
        ids = torch.randint(0, vocab, (batch, l + 1), generator=g)
        yield ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


def held_out_loss(model: nn.Module, stream_factory: BatchStream,
                  cfg: TrainConfig) -> float:
    """Mean next-token CE over held-out batches, model in eval mode
    (dropout off). Reporting only — never used for training or selection."""
    model.eval()
    total, n = 0.0, 0
    it = stream_factory()
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            x, y = next(it)
            logits = model(x)
            total += F.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1)).item()
            n += 1
    model.train()
    return total / max(n, 1)


def train_one_seed(arm: str, cfg: TrainConfig, seed: int, steps: int,
                   train_stream: BatchStream, held_out_stream: BatchStream,
                   log_every: Optional[int] = None,
                   save_path: Optional[str] = None) -> Dict:
    """One seed's trajectory: per-step train losses + held-out eval points.

    Returns plain data (curves only — no verdicts). Mechanics validated by
    tests/test_harness_mechanics.py; the same seed reproduces the run
    bit-exactly (determinism is a tested property, not a claim).

    `save_path` (optional) writes a checkpoint carrying the arm, the seed, the
    step count, the config, the eval points and the final held-out loss —
    the paired quality checkpoint that Milestone 3 ties to its performance
    numbers.
    """
    torch.manual_seed(seed)
    model = build_model(arm, cfg)
    apply_init_policy(model, seed=seed)
    opt = build_optimizer(model, cfg)

    log_every = log_every if log_every is not None else cfg.eval_every
    train_it = train_stream()
    curve: List[float] = []
    evals: List[Dict] = []
    model.train()
    for step in range(steps):
        x, y = next(train_it)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        opt.step()
        curve.append(float(loss.item()))
        if (step + 1) % log_every == 0 or step == steps - 1:
            evals.append({"step": step + 1,
                          "held_out_loss": held_out_loss(model, held_out_stream, cfg)})
    final = evals[-1]["held_out_loss"] if evals else None
    if save_path is not None:
        from dataclasses import asdict

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arm": arm, "seed": seed, "steps": steps,
                    "config": asdict(cfg), "evals": evals,
                    "final_held_out_loss": final,
                    "state_dict": model.state_dict()}, save_path)
    return {"arm": arm, "seed": seed, "steps": steps,
            "train_curve": curve, "evals": evals,
            "final_held_out_loss": final}


def dispersion(runs: List[Dict]) -> Dict:
    """Per-seed dispersion across seeds at each eval point (mean/std/n).

    Pure statistics over the curves — the harness reports, it does not judge.
    """
    import math

    points = sorted({e["step"] for r in runs for e in r["evals"]})
    out = {}
    for p in points:
        vals = [e["held_out_loss"] for r in runs for e in r["evals"] if e["step"] == p]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 0.0
        out[p] = {"n": len(vals), "mean": mean, "std": math.sqrt(var)}
    return out


def run_multi_seed(arm: str, cfg: TrainConfig, seeds: List[int], steps: int,
                   train_seed: int = 0) -> Dict:
    """The full protocol: per-seed curves + dispersion summary."""
    train_stream = lambda: synthetic_stream(cfg.vocab, cfg.l, cfg.batch_size, seed=train_seed)
    held_out_stream = lambda: synthetic_stream(cfg.vocab, cfg.l, cfg.batch_size,
                                               seed=train_seed, held_out=True)
    runs = [train_one_seed(arm, cfg, s, steps, train_stream, held_out_stream)
            for s in seeds]
    return {"arm": arm, "seeds": seeds, "runs": runs,
            "dispersion": dispersion(runs)}


# --------------------------------------------------------------------------
# Corpus-backed provider (T14). The synthetic stream above is retained for the
# mechanics dry-run only (D-17); real runs use the pinned corpus snapshot.
# --------------------------------------------------------------------------
def corpus_stream_factory(bundle, cfg: TrainConfig, train_seed: int = 0,
                          held_out: bool = False):
    """BatchStream over the pinned corpus (deterministic, D-02 split).

    The held-out factory always starts from the same seed, so every eval point
    is measured on the SAME held-out batches (a fixed evaluation set) — and
    those batches are never trained on.
    """
    def factory():
        for x, y in bundle.stream(l=cfg.l, batch=cfg.batch_size,
                                  train_seed=train_seed, held_out=held_out):
            yield torch.from_numpy(x), torch.from_numpy(y)
    return factory


def run_multi_seed_corpus(arm: str, cfg: TrainConfig, seeds: List[int], steps: int,
                           bundle, train_seed: int = 0,
                           log_every: Optional[int] = None,
                           checkpoint_dir: Optional[str] = None) -> Dict:
    """Multi-seed protocol on the pinned corpus. Curves only — no verdicts.

    `checkpoint_dir` (optional) writes one checkpoint per (arm, seed) — the
    paired quality checkpoints Milestone 3 attaches to its performance rows.
    """
    train_stream = corpus_stream_factory(bundle, cfg, train_seed=train_seed, held_out=False)
    held_out_stream = corpus_stream_factory(bundle, cfg, train_seed=train_seed, held_out=True)
    runs = []
    for s in seeds:
        save_path = None
        if checkpoint_dir is not None:
            save_path = str(Path(checkpoint_dir) / f"{arm}_seed{s}.pt")
        runs.append(train_one_seed(arm, cfg, s, steps, train_stream, held_out_stream,
                                   log_every=log_every, save_path=save_path))
    return {"arm": arm, "seeds": seeds, "train_seed": train_seed, "steps": steps,
            "runs": runs, "dispersion": dispersion(runs)}
