"""Milestone 3 — measured systems comparison (protocol revision 2).

Measures, on identical hardware/software and in one process:

  A. forward training step  — latency, throughput, peak memory (batch 64 x l=128)
  B. full-forward prefill   — latency, throughput, peak memory vs context length
  C. incremental decoding   — first-token and steady-state per-token latency,
                              throughput, cache bytes
  D. sublayer-1 scaling     — per-token cost and state bytes vs context length
                              (128 … 8192), where the full model cannot go
                              (the positional table caps it at l=128)
  E. traffic                — analytic parameter bytes per step/token, and the
                              per-token state bytes each arm carries

This revision implements the Milestone-3 systems requirements S1–S10 fixed by
the protocol review (`docs/milestone-3/PROTOCOL.md` §4.3, pre-registration of
2026-10-23):

  S1  the optimized SDPA baseline (`attention-32h-sdpa`) is timed in every block
  S2  A/A null + palindromic (ABBA) arm order; the null's spread is the drift floor
  S3  >= 10 timed trials per row (decode >= 5); median/p10/p90/min/max/n published
  S4  first-token decode reported separately from steady-state decode
  S5  TF32 / matmul-precision / cuDNN flags set explicitly and recorded
  S6  CPU memory rows labelled process-level (ru_maxrss is a process high-water mark)
  S7  traffic rows labelled analytic and carry their formulas; per-token state
      traffic added to the traffic block
  S8  timed models load the saved quality checkpoints (or say "architecture at
      initialisation" — never imply a quality tie that was not measured)
  S9  model-level rows (<= l) and sublayer rows are separate and labelled
  S10 every row carries its units and denominators

Everything is reported as DATA with its conditions (device, threads, shape,
warmup, trials, numerics flags, git commit). No verdicts: the write-up is a
separate step, and every wall-clock row carries the kernel caveat (HE here is
an eager Python offset loop; the attention baseline can dispatch a fused
kernel) — see the caveat block at the bottom of the output.

Usage:
    python experiments/run_systems.py --device cpu --out results/systems.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hextractor.harness import TrainConfig, build_model, build_optimizer  # noqa: E402
from hextractor.incremental import (  # noqa: E402
    AttentionState, HEState, ModelCache, attention_forward_incremental,
    he_forward_incremental, model_forward_incremental,
    sdpa_attention_forward_incremental,
)
from hextractor.init import apply_init_policy  # noqa: E402
from hextractor.layers import (  # noqa: E402
    AttentionSublayer, HESublayer, SDPAAttentionSublayer,
)
from hextractor.ops_count import count_ops_attention_fwd, count_ops_fwd  # noqa: E402

SCHEMA = "untransformers-systems-run/2"
ARMS_ALL = ("HE", "attention-32h", "attention-32h-sdpa")

KERNEL_CAVEAT = (
    "HE here is an eager Python offset loop (hextractor/layers.py: the module "
    "docstring states there are no fused kernels); the attention arms may "
    "dispatch a fused SDPA kernel. A wall-clock difference between them "
    "therefore measures implementation maturity under these conditions, not the "
    "HE mechanism as such."
)
CPU_MEMORY_SCOPE = (
    "process-level: ru_maxrss is the high-water mark of the whole process and "
    "includes every arm measured before this one (S6)."
)
CUDA_MEMORY_SCOPE = (
    "per-arm: torch.cuda.max_memory_allocated/max_memory_reserved, counters "
    "reset before the arm's first timed slot (S6)."
)
INIT_QUALITY_LEVEL = (
    "architecture at initialisation — no quality checkpoint was loaded, so "
    "this row is NOT tied to any quality level (S8)."
)


# --------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------
def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _rss_bytes() -> int:
    """Current resident set size (Linux), in bytes."""
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:  # noqa: BLE001
        return 0


def _peak_rss_bytes() -> int:
    """Process high-water RSS (ru_maxrss is in KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_mem(device: torch.device) -> Dict:
    if device.type == "cuda":
        return {"scope": CUDA_MEMORY_SCOPE,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
    return {"scope": CPU_MEMORY_SCOPE,
            "peak_rss_bytes": _peak_rss_bytes(),
            "rss_bytes": _rss_bytes()}


def _pct(sorted_xs: Sequence[float], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    idx = min(len(sorted_xs) - 1, max(0, int(round(q * (len(sorted_xs) - 1)))))
    return float(sorted_xs[idx])


def _stats(xs: List[float]) -> Dict:
    """median / mean / p10 / p90 / min / max / n (S3)."""
    if not xs:
        return {"median": None, "mean": None, "p10": None, "p90": None,
                "min": None, "max": None, "n": 0}
    s = sorted(xs)
    return {
        "median": float(statistics.median(s)),
        "mean": float(sum(s) / len(s)),
        "p10": _pct(s, 0.10),
        "p90": _pct(s, 0.90),
        "min": float(s[0]),
        "max": float(s[-1]),
        "n": len(s),
    }


def _timeit(fn, device, trials: int, warmup: int) -> Dict:
    """Run fn `warmup` times (untimed), then `trials` timed. Returns stats."""
    for _ in range(warmup):
        fn()
    _sync(device)
    times: List[float] = []
    for _ in range(trials):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    out = _stats(times)
    out["trials"] = trials
    out["warmup"] = warmup
    out["times"] = times
    return out


# --------------------------------------------------------------------------
# order schedule (S2): palindromic ABBA + A/A null
# --------------------------------------------------------------------------
def order_schedule(arms: Sequence[str]) -> List[str]:
    """Palindromic measurement order: `arms + reversed(arms)`.

    Every arm is therefore measured twice — once early and once late in the
    run — which yields an A/A pair per arm. The difference between the two
    medians of an arm is the run-order drift floor (clock/thermal drift on a
    shared accelerator), and any arm-to-arm difference smaller than that floor
    is not distinguishable from drift (S2).
    """
    return list(arms) + list(reversed(arms))


def _order_effect(slot_rows: List[Dict], block: str) -> Dict:
    """Per-arm A/A null + drift floor from the slot rows of one block."""
    by_arm: Dict[str, List[Dict]] = {}
    for row in slot_rows:
        by_arm.setdefault(row["arm"], []).append(row)

    aa_null: Dict[str, Dict] = {}
    worst_abs = 0.0
    worst_rel = 0.0
    for arm, rows in by_arm.items():
        rows = sorted(rows, key=lambda r: r["order_index"])
        if len(rows) < 2:
            continue
        first = rows[0]["latency_stats"]["median"]
        last = rows[-1]["latency_stats"]["median"]
        if first is None or last is None:
            continue
        diff = last - first
        rel = abs(diff) / first if first else float("inf")
        aa_null[arm] = {
            "first_slot_order_index": rows[0]["order_index"],
            "last_slot_order_index": rows[-1]["order_index"],
            "first_slot_median_seconds": first,
            "last_slot_median_seconds": last,
            "difference_seconds": diff,
            "absolute_difference_seconds": abs(diff),
            "relative_difference": rel,
        }
        worst_abs = max(worst_abs, abs(diff))
        worst_rel = max(worst_rel, rel)

    return {
        "block": block,
        "aa_null": aa_null,
        "drift_floor_seconds": worst_abs,
        "drift_floor_relative": worst_rel,
        "interpretation": (
            "A/A null: the same arm measured at the start and at the end of the "
            "measurement order. The spread of this null is the run-order drift "
            "floor; an arm-to-arm difference smaller than it is not resolvable "
            "against clock/thermal drift."
        ),
    }


# --------------------------------------------------------------------------
# quality checkpoints (S8)
# --------------------------------------------------------------------------
def _checkpoint_seeds(ckdir: Optional[Path], arm: str) -> List[int]:
    if ckdir is None or not ckdir.exists():
        return []
    seeds: List[int] = []
    for p in sorted(ckdir.glob(f"{arm}_seed*.pt")):
        try:
            seeds.append(int(p.stem.rsplit("seed", 1)[1]))
        except (IndexError, ValueError):
            continue
    return seeds


def _load_checkpoint(model, arm: str, ckdir: Optional[Path],
                     preferred_seed: int) -> Dict:
    """Load the saved quality checkpoint into `model`; report the quality level.

    Never silently skip a failure: the returned block always says whether the
    timed model was a trained checkpoint or the architecture at initialisation.
    """
    seeds = _checkpoint_seeds(ckdir, arm)
    if not seeds:
        return {"loaded": False, "arm": arm, "seed": None, "checkpoint": None,
                "steps": None, "final_held_out_loss": None,
                "quality_level": INIT_QUALITY_LEVEL,
                "note": "no checkpoint found for this arm"}
    seed = preferred_seed if preferred_seed in seeds else seeds[0]
    path = Path(ckdir) / f"{arm}_seed{seed}.pt"
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        state = blob.get("state_dict")
        if not state:
            raise RuntimeError("checkpoint carries no state_dict")
        model.load_state_dict(state, strict=True)
    except Exception as exc:  # noqa: BLE001
        return {"loaded": False, "arm": arm, "seed": seed, "checkpoint": str(path),
                "steps": None, "final_held_out_loss": None,
                "quality_level": INIT_QUALITY_LEVEL,
                "note": f"checkpoint could not be loaded ({exc})"}
    return {
        "loaded": True,
        "arm": arm,
        "seed": seed,
        "checkpoint": str(path),
        "steps": blob.get("steps"),
        "final_held_out_loss": blob.get("final_held_out_loss"),
        "quality_level": "trained quality checkpoint (held-out loss above)",
        "note": "the timed model carries the checkpoint's weights",
    }


# --------------------------------------------------------------------------
# A. training step
# --------------------------------------------------------------------------
def measure_training_step(arm: str, cfg: TrainConfig, device: torch.device,
                          trials: int, warmup: int, seed: int = 1,
                          ckdir: Optional[Path] = None,
                          checkpoint_seed: int = -1) -> Dict:
    """One timed slot of the training-step block."""
    torch.manual_seed(seed)
    model = build_model(arm, cfg).to(device)
    apply_init_policy(model, seed=seed)
    quality = _load_checkpoint(model, arm, ckdir, checkpoint_seed)
    model.train()                                   # dropout ON, as in training
    opt = build_optimizer(model, cfg)
    x = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.l), device=device)
    y = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.l), device=device)

    def one_step():
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    t = _timeit(one_step, device, trials, warmup)
    mem = _peak_mem(device)
    tokens = cfg.batch_size * cfg.l
    out = {
        "arm": arm,
        "scope": "model-level",
        "step_seconds": t,
        "tokens_per_step": tokens,
        "tokens_per_second": tokens / t["median"] if t["median"] else None,
        "memory": mem,
        "mode": "train (dropout on)",
        "n_params": sum(p.numel() for p in model.parameters()),
        "quality": quality,
        "kernel_caveat": KERNEL_CAVEAT,
        "units": {
            "latency": "seconds per training step (forward + backward + optimizer step)",
            "throughput": "tokens per second; denominator = batch_size * l tokens per step",
            "memory": "bytes",
        },
        "denominator": f"batch_size({cfg.batch_size}) * l({cfg.l}) = {tokens} tokens/step",
    }
    if quality.get("loaded"):
        out["quality"]["note"] = (
            f"the timed block additionally takes {trials + warmup} optimizer steps "
            f"from this checkpoint")
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# B. full-forward prefill
# --------------------------------------------------------------------------
def measure_prefill(arm: str, cfg: TrainConfig, device: torch.device,
                    context_lengths: List[int], trials: int, warmup: int,
                    seed: int = 1, ckdir: Optional[Path] = None,
                    checkpoint_seed: int = -1) -> List[Dict]:
    """One timed slot of the prefill block, for every context length."""
    torch.manual_seed(seed)
    model = build_model(arm, cfg).to(device)
    apply_init_policy(model, seed=seed)
    quality = _load_checkpoint(model, arm, ckdir, checkpoint_seed)
    model.eval()
    rows = []
    for T in context_lengths:
        ids = torch.randint(0, cfg.vocab, (1, T), device=device)
        with torch.no_grad():
            t = _timeit(lambda: model(ids), device, trials, warmup)
        rows.append({"arm": arm, "scope": "model-level", "context_tokens": T,
                     "batch": 1, "latency": t,
                     "tokens_per_second": T / t["median"] if t["median"] else None,
                     "memory": _peak_mem(device), "quality": quality,
                     "kernel_caveat": KERNEL_CAVEAT,
                     "units": {
                         "latency": "seconds for one full forward over context_tokens tokens",
                         "throughput": "tokens per second; denominator = context_tokens (batch 1)",
                         "memory": "bytes",
                     },
                     "denominator": f"batch 1 * context_tokens({T}) = {T} tokens/forward"})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------
# C. incremental decoding
# --------------------------------------------------------------------------
def measure_incremental_decode(arm: str, cfg: TrainConfig, device: torch.device,
                               context_lengths: List[int], gen_tokens: int,
                               trials: int, warmup: int, seed: int = 1,
                               ckdir: Optional[Path] = None,
                               checkpoint_seed: int = -1) -> List[Dict]:
    """One timed slot of the decode block.

    The first generated token (immediately after the prompt) is timed
    separately from the steady-state tokens (S4): it carries prefill-adjacent
    effects (cache growth on the first append) and must not be pooled with
    the steady state.
    """
    torch.manual_seed(seed)
    model = build_model(arm, cfg).to(device)
    apply_init_policy(model, seed=seed)
    quality = _load_checkpoint(model, arm, ckdir, checkpoint_seed)
    model.eval()
    rows = []
    for T in context_lengths:
        prompt = torch.randint(0, cfg.vocab, (1, T), device=device)
        first_token: List[float] = []
        steady: List[float] = []
        cache_bytes = 0
        for trial in range(trials + warmup):
            cache = ModelCache.for_model(model)
            with torch.no_grad():
                model_forward_incremental(model, prompt, cache)
                cache_bytes = cache.state_bytes()
                if trial < warmup:
                    continue
                for gen in range(gen_tokens):
                    tok = torch.randint(0, cfg.vocab, (1,), device=device)
                    _sync(device)
                    t0 = time.perf_counter()
                    model_forward_incremental(model, tok[:, None], cache)
                    _sync(device)
                    dt = time.perf_counter() - t0
                    (first_token if gen == 0 else steady).append(dt)
        all_tokens = first_token + steady
        rows.append({
            "arm": arm,
            "scope": "model-level",
            "context_tokens": T,
            "generated_tokens": gen_tokens,
            "max_positions": cfg.l,
            "per_token_seconds": _stats(all_tokens),
            "first_token_seconds": _stats(first_token),
            "steady_state_per_token_seconds": _stats(steady),
            "tokens_per_second": (1.0 / _stats(steady)["median"]
                                  if steady and _stats(steady)["median"] else None),
            "tokens_per_second_including_first_token": (
                1.0 / _stats(all_tokens)["median"]
                if all_tokens and _stats(all_tokens)["median"] else None),
            "cache_bytes_at_prompt": cache_bytes,
            "cache_bytes_per_layer": cache_bytes // cfg.n_layers,
            "memory": _peak_mem(device),
            "quality": quality,
            "kernel_caveat": KERNEL_CAVEAT,
            "_times": {"per_token": all_tokens, "first_token": first_token,
                       "steady_state": steady},
            "units": {
                "per_token_latency": "seconds per generated token (steady state)",
                "first_token_latency": "seconds for the first generated token after the prompt",
                "throughput": "tokens per second; denominator = 1 token per decode step",
                "cache_bytes": "bytes held by the cache after the prompt",
            },
            "denominator": (f"decode trials={trials} (+{warmup} warmup) x "
                            f"gen_tokens={gen_tokens} tokens per context"),
        })
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------
# D. sublayer-1 scaling beyond the model's positional cap
# --------------------------------------------------------------------------
def _make_sublayer(arm: str, cfg: TrainConfig):
    if arm == "HE":
        return HESublayer(cfg.d, cfg.l, variant="HE")
    if arm == "attention-32h-sdpa":
        return SDPAAttentionSublayer(cfg.d, n_heads=cfg.n_heads)
    return AttentionSublayer(cfg.d, n_heads=cfg.n_heads)


def _sublayer_incremental(arm: str, sub, x, state, start):
    if arm == "HE":
        return he_forward_incremental(sub, x, state, start=start)
    if arm == "attention-32h-sdpa":
        return sdpa_attention_forward_incremental(sub, x, state, start=start)
    return attention_forward_incremental(sub, x, state, start=start)


def measure_sublayer_scaling(arms: List[str], cfg: TrainConfig, device: torch.device,
                             context_lengths: List[int], trials: int, warmup: int,
                             seed: int = 1) -> List[Dict]:
    """Per-token cost of sublayer-1 alone, beyond the model's positional cap.

    This is the fair algorithmic comparison (S9): both arms are measured the
    same way at the sublayer level, one token at a time, against a state that
    has been warmed up to the target context length. HE's state is flat past
    l-1; attention's KV cache grows linearly with the context.
    """
    rows = []
    for arm in arms:
        torch.manual_seed(seed)
        sub = _make_sublayer(arm, cfg).to(device)
        with torch.no_grad():
            for p in sub.parameters():
                p.copy_(torch.randn_like(p) * 0.01)
        sub.eval()
        for T in context_lengths:
            x = torch.randn(1, 1, cfg.d, device=device)
            state = HEState() if arm == "HE" else AttentionState()
            with torch.no_grad():
                chunk = torch.randn(1, T - 1, cfg.d, device=device)
                _sublayer_incremental(arm, sub, chunk, state, 0)
            times: List[float] = []
            for i in range(trials + warmup):
                _sync(device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    _sublayer_incremental(arm, sub, x, state, T - 1)
                _sync(device)
                dt = time.perf_counter() - t0
                if i >= warmup:
                    times.append(dt)
            stats = _stats(times)
            stats["trials"] = trials
            stats["warmup"] = warmup
            if arm == "HE":
                state_bytes = (cfg.l - 1) * cfg.d * 4
                state_formula = "(l - 1) * d * 4 bytes = "
                f"({cfg.l} - 1) * {cfg.d} * 4"
                state_note = ("HE keeps only the last l-1 projected vectors, so its "
                              "state is flat in the context length")
            else:
                state_bytes = 2 * T * cfg.d * 4
                state_formula = f"2 * T * d * 4 bytes = 2 * {T} * {cfg.d} * 4"
                state_note = ("attention keeps a K and a V cache of T tokens, so its "
                              "state grows linearly with the context")
            rows.append({
                "arm": arm,
                "scope": "sublayer-1",
                "context_tokens": T,
                "per_token_seconds": stats,
                "state_bytes_per_layer": state_bytes,
                "state_bytes_all_layers": state_bytes * cfg.n_layers,
                "state_bytes_label": "analytic (formula, not measured)",
                "state_bytes_formula": state_formula,
                "state_bytes_note": state_note,
                "analytic_mults_per_token": (
                    count_ops_fwd("HE", 1, cfg.d, cfg.l)[0] if arm == "HE"
                    else _attention_incremental_mults(cfg.d, cfg.n_heads, T)),
                "kernel_caveat": KERNEL_CAVEAT,
                "units": {
                    "per_token_latency": "seconds for one incremental token at sublayer-1",
                    "state_bytes": "bytes per layer (analytic)",
                    "analytic_mults_per_token": "multiplications per incremental token (analytic)",
                },
                "denominator": f"one token at context T={T}, sublayer-1 only",
            })
        del sub
    return rows


def _attention_incremental_mults(d: int, n_heads: int, T: int) -> int:
    """Analytic multiplications for ONE incremental attention token at context T.

    Q,K,V,O projections: 4*d*d. Scores over the T cached keys: T*d. Context
    mix: T*d. (The T13 convention is for a full forward; this is the
    per-token incremental form, stated separately.)
    """
    return 4 * d * d + 2 * T * d


# --------------------------------------------------------------------------
# E. traffic (analytic) + model inventory
# --------------------------------------------------------------------------
def weight_traffic(cfg: TrainConfig) -> Dict:
    """Analytic parameter traffic per arm (S7).

    Every number here is a formula over the parameter count — nothing is
    measured. The formulas are printed next to the numbers, and the note no
    longer contradicts them: a training step reads the parameters three times
    (forward, dgrad, wgrad), a decode step once.
    """
    rows = {}
    for arm in ARMS_ALL:
        model = build_model(arm, cfg)
        n_params = sum(p.numel() for p in model.parameters())
        param_bytes = n_params * 4                      # fp32
        rows[arm] = {
            "label": "analytic (formula over the parameter count, not measured)",
            "n_params": n_params,
            "parameter_bytes_fp32": param_bytes,
            "parameter_bytes_formula": "n_params * 4 bytes (fp32)",
            "training_step_parameter_bytes_read": 3 * param_bytes,
            "training_step_parameter_bytes_formula":
                "3 * n_params * 4 bytes (forward + dgrad + wgrad)",
            "decode_step_parameter_bytes_read": param_bytes,
            "decode_step_parameter_bytes_formula":
                "1 * n_params * 4 bytes (one forward pass)",
            "note": ("analytic: a training step reads the parameter bytes three times "
                     "(forward, dgrad, wgrad); a decode step reads them once. Both "
                     "arms carry the same parameter count by construction, so this "
                     "block cannot show a traffic difference between the arms."),
        }
        del model
    return rows


def state_traffic(cfg: TrainConfig, context_lengths: List[int]) -> Dict:
    """Analytic per-token state / KV traffic per arm (S7)."""
    rows = {}
    for arm in ARMS_ALL:
        if arm == "HE":
            per_token = (cfg.l - 1) * cfg.d * 4
            rows[arm] = {
                "label": "analytic (formula, not measured)",
                "state_bytes_per_layer_per_token": per_token,
                "state_bytes_per_layer_formula":
                    f"(l - 1) * d * 4 = ({cfg.l} - 1) * {cfg.d} * 4 bytes",
                "state_bytes_all_layers_per_token": per_token * cfg.n_layers,
                "grows_with_context": False,
                "note": ("HE's per-token state is flat in the context length: the "
                         "offset gates only look back l-1 positions"),
                "per_context": [],
            }
            for T in context_lengths:
                rows[arm]["per_context"].append({
                    "context_tokens": T,
                    "state_bytes_per_layer": per_token,
                    "state_bytes_all_layers": per_token * cfg.n_layers,
                    "formula": f"(l - 1) * d * 4 = ({cfg.l} - 1) * {cfg.d} * 4 bytes",
                })
        else:
            rows[arm] = {
                "label": "analytic (formula, not measured)",
                "state_bytes_per_layer_per_token": None,
                "state_bytes_per_layer_formula": "2 * T * d * 4 bytes (K and V caches)",
                "state_bytes_all_layers_per_token": None,
                "grows_with_context": True,
                "note": ("attention's KV cache holds 2 * T * d floats per layer and "
                         "grows linearly with the context; the implementation here "
                         "reallocates it with torch.cat on every token (declared "
                         "limitation, PROTOCOL.md §8.D)"),
                "per_context": [],
            }
            for T in context_lengths:
                per_layer = 2 * T * cfg.d * 4
                rows[arm]["per_context"].append({
                    "context_tokens": T,
                    "state_bytes_per_layer": per_layer,
                    "state_bytes_all_layers": per_layer * cfg.n_layers,
                    "formula": f"2 * T * d * 4 = 2 * {T} * {cfg.d} * 4 bytes",
                })
    return rows


# --------------------------------------------------------------------------
# environment manifest
# --------------------------------------------------------------------------
def _git_info() -> Dict:
    def run(*args) -> Optional[str]:
        try:
            r = subprocess.run(["git", *args], cwd=str(REPO_ROOT), text=True,
                               capture_output=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:  # noqa: BLE001
            return None
    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    return {"git_commit": commit,
            "git_dirty": bool(dirty) if dirty is not None else None,
            "git_branch": run("rev-parse", "--abbrev-ref", "HEAD")}


def set_numerics() -> Dict:
    """Set the numerics explicitly and report them (S5).

    A T4 number is not reproducible across runs if TF32 / matmul precision /
    cuDNN flags are left at their defaults, so they are pinned here and the
    values are recorded with the measurement.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "dtype": "float32",
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "note": ("pinned by the runner before any measurement; TF32 is OFF and "
                 "matmul precision is 'highest' (S5)"),
    }


def env_manifest(device: torch.device, numerics: Dict, arms: List[str]) -> Dict:
    man = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": device.type,
        "cpu_count": os.cpu_count(),
        "numerics": numerics,
        "arms": list(arms),
    }
    man.update(_git_info())
    if device.type == "cuda":
        man.update({
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
            "gpu_memory_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        })
    return man


# --------------------------------------------------------------------------
# aggregation helpers
# --------------------------------------------------------------------------
def _merge_memory(rows: List[Dict]) -> Dict:
    """Per-arm memory: max over the arm's timed slots (CUDA), process-level (CPU).

    The CUDA allocator counters are reset before every timed slot, so each
    slot's peak is measured independently and the arm's peak is the maximum
    over its slots. On CPU `ru_maxrss` is a process high-water mark and is
    reported as such (S6).
    """
    if not rows:
        return {}
    last = rows[-1].get("memory") or {}
    if "peak_allocated_bytes" not in last:
        return last
    merged = dict(last)
    for field_name in ("peak_allocated_bytes", "peak_reserved_bytes"):
        vals = [r["memory"][field_name] for r in rows
                if (r.get("memory") or {}).get(field_name) is not None]
        if vals:
            merged[field_name] = int(max(vals))
    merged["scope"] = (
        "per-arm: max over this arm's timed slots of "
        "torch.cuda.max_memory_allocated/max_memory_reserved (counters reset "
        "before every slot) (S6)")
    return merged


def _pool_slots(slots: List[Dict], key: str = "latency_stats") -> List[Dict]:
    """Group slot rows by arm (and context where present) and pool the times."""
    groups: Dict = {}
    for row in slots:
        gkey = (row["arm"], row.get("context_tokens"))
        groups.setdefault(gkey, []).append(row)

    out = []
    for (arm, ctx), rows in groups.items():
        rows = sorted(rows, key=lambda r: r["order_index"])
        pooled: List[float] = []
        for r in rows:
            pooled.extend(r[key].get("times") or [])
        base = dict(rows[-1])
        base.pop("_times", None)
        base["slots"] = len(rows)
        base["order_indices"] = [r["order_index"] for r in rows]
        stats = _stats(pooled)
        stats["trials"] = rows[0][key].get("trials")
        stats["warmup"] = rows[0][key].get("warmup")
        stats["samples_pooled"] = len(pooled)
        base[key] = stats
        # the public latency fields carry the pooled statistics too
        for field_name in ("step_seconds", "latency"):
            if field_name in base:
                base[field_name] = dict(stats)
        base["memory"] = _merge_memory(rows)
        out.append(base)
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS_ALL))
    ap.add_argument("--systems-arms", nargs="+", default=list(ARMS_ALL),
                    help="arms used for the incremental-decoding sweep")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default="results/systems.json")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--l", type=int, default=128)
    ap.add_argument("--layers", type=int, default=18)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--trials", type=int, default=10,
                    help="timed trials per row (S3: >= 10)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--gen-tokens", type=int, default=16)
    ap.add_argument("--decode-trials", type=int, default=5,
                    help="timed decode trials per context (S3: >= 5)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--checkpoint-seed", type=int, default=-1)
    ap.add_argument("--contexts", nargs="+", type=int,
                    default=[1, 8, 32, 64, 96, 112, 128],
                    help="model-level context lengths (capped by l)")
    ap.add_argument("--sublayer-contexts", nargs="+", type=int,
                    default=[128, 256, 512, 1024, 2048, 4096, 8192])
    args = ap.parse_args()

    if args.trials < 10:
        print(f"[systems] WARNING: --trials {args.trials} is below the S3 minimum of 10",
              flush=True)
    if args.decode_trials < 5:
        print(f"[systems] WARNING: --decode-trials {args.decode_trials} is below the "
              f"S3 minimum of 5", flush=True)

    if args.threads:
        torch.set_num_threads(args.threads)
    numerics = set_numerics()
    device = torch.device("cuda" if (args.device == "cuda" or
                                     (args.device == "auto" and torch.cuda.is_available()))
                          else "cpu")

    cfg = TrainConfig(d=args.d, l=args.l, n_layers=args.layers, ffn_width=args.ffn,
                      vocab=args.vocab, batch_size=args.batch, n_heads=args.heads)

    ckdir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if ckdir is not None and not ckdir.exists():
        print(f"[systems] WARNING: checkpoint dir {ckdir} does not exist — every "
              f"timed model will be 'architecture at initialisation'", flush=True)

    # The paper shape carries a positional table of `l` rows, so the total
    # number of positions (prompt + generated) cannot exceed l. Contexts are
    # filtered accordingly; the cap is reported with every row (S9).
    model_contexts = [c for c in args.contexts if c + args.gen_tokens <= cfg.l]

    out = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": env_manifest(device, numerics, args.arms),
        "config": {"d": cfg.d, "l": cfg.l, "n_layers": cfg.n_layers,
                   "ffn_width": cfg.ffn_width, "vocab": cfg.vocab,
                   "batch_size": cfg.batch_size, "n_heads": cfg.n_heads,
                   "trials": args.trials, "warmup": args.warmup,
                   "gen_tokens": args.gen_tokens,
                   "decode_trials": args.decode_trials,
                   "contexts": model_contexts,
                   "sublayer_contexts": list(args.sublayer_contexts),
                   "tokens_per_step": cfg.batch_size * cfg.l,
                   "order_schedule": order_schedule(args.arms),
                   "order_schedule_kind": "palindromic (ABBA): arms + reversed(arms)",
                   "scope_note": ("model-level rows cover contexts <= l (the "
                                  "positional table's cap); sublayer-1 rows are a "
                                  "separate regime (S9)"),
                   },
        "training_step": [],
        "prefill": [],
        "incremental_decode": [],
        "sublayer_scaling": [],
        "weight_traffic": {},
        "state_traffic": {},
        "paired_quality": {},
        "order_effect": {},
        "caveats": [],
    }

    print(f"[systems] device={device} arms={args.arms}", flush=True)
    print(f"[systems] order schedule: {out['config']['order_schedule']}", flush=True)

    # --- A. training step (interleaved, A/A null) --------------------------
    schedule = out["config"]["order_schedule"]
    ts_slots: List[Dict] = []
    for idx, arm in enumerate(schedule):
        print(f"[training-step] slot {idx} arm={arm}", flush=True)
        _reset_peak(device)
        row = measure_training_step(arm, cfg, device, args.trials, args.warmup,
                                    ckdir=ckdir, checkpoint_seed=args.checkpoint_seed)
        row["order_index"] = idx
        row["latency_stats"] = row["step_seconds"]
        ts_slots.append(row)
    out["training_step_slots"] = ts_slots
    out["training_step"] = _pool_slots(ts_slots)
    out["order_effect"]["training_step"] = _order_effect(ts_slots, "training_step")

    # --- B. prefill (interleaved, A/A null) --------------------------------
    pre_slots: List[Dict] = []
    for idx, arm in enumerate(schedule):
        print(f"[prefill] slot {idx} arm={arm}", flush=True)
        _reset_peak(device)
        rows = measure_prefill(arm, cfg, device, model_contexts, args.trials,
                               args.warmup, ckdir=ckdir,
                               checkpoint_seed=args.checkpoint_seed)
        for r in rows:
            r["order_index"] = idx
            r["latency_stats"] = r["latency"]
            pre_slots.append(r)
    out["prefill_slots"] = pre_slots
    out["prefill"] = _pool_slots(pre_slots)
    out["order_effect"]["prefill"] = _order_effect(pre_slots, "prefill")

    # --- C. incremental decoding (interleaved, A/A null) --------------------
    dec_slots: List[Dict] = []
    dec_schedule = order_schedule(args.systems_arms)
    for idx, arm in enumerate(dec_schedule):
        print(f"[incremental-decode] slot {idx} arm={arm}", flush=True)
        _reset_peak(device)
        rows = measure_incremental_decode(arm, cfg, device, model_contexts,
                                          args.gen_tokens, args.decode_trials,
                                          args.warmup, ckdir=ckdir,
                                          checkpoint_seed=args.checkpoint_seed)
        for r in rows:
            r["order_index"] = idx
            r["latency_stats"] = {
                "median": r["steady_state_per_token_seconds"]["median"],
                "mean": r["steady_state_per_token_seconds"]["mean"],
                "p10": r["steady_state_per_token_seconds"]["p10"],
                "p90": r["steady_state_per_token_seconds"]["p90"],
                "min": r["steady_state_per_token_seconds"]["min"],
                "max": r["steady_state_per_token_seconds"]["max"],
                "n": r["steady_state_per_token_seconds"]["n"],
            }
            dec_slots.append(r)
    out["incremental_decode_slots"] = dec_slots
    out["order_effect"]["incremental_decode"] = _order_effect(dec_slots, "incremental_decode")

    # The decode rows are aggregated per (arm, context) with the per-token
    # statistics pooled across the two slots of that arm.
    dec_groups: Dict = {}
    for r in dec_slots:
        dec_groups.setdefault((r["arm"], r["context_tokens"]), []).append(r)
    dec_out = []
    for (arm, ctx), rows in dec_groups.items():
        rows = sorted(rows, key=lambda r: r["order_index"])
        base = dict(rows[-1])
        time_key = {"per_token_seconds": "per_token",
                    "first_token_seconds": "first_token",
                    "steady_state_per_token_seconds": "steady_state"}
        for field_name, tkey in time_key.items():
            pooled: List[float] = []
            for r in rows:
                pooled.extend(r["_times"].get(tkey) or [])
            merged = _stats(pooled)
            merged["n_slots"] = len(rows)
            merged["samples_per_slot"] = rows[0][field_name].get("n")
            base[field_name] = merged
        base.pop("_times", None)
        base["slots"] = len(rows)
        base["order_indices"] = [r["order_index"] for r in rows]
        base["memory"] = _merge_memory(rows)
        steady_med = base["steady_state_per_token_seconds"]["median"]
        base["tokens_per_second"] = 1.0 / steady_med if steady_med else None
        dec_out.append(base)
    dec_out.sort(key=lambda r: (r["arm"], r["context_tokens"]))
    out["incremental_decode"] = dec_out

    # --- D. sublayer-1 scaling ---------------------------------------------
    print(f"[sublayer-scaling] {' '.join(ARMS_ALL)}", flush=True)
    out["sublayer_scaling"] = measure_sublayer_scaling(
        list(ARMS_ALL), cfg, device, list(args.sublayer_contexts),
        args.trials, args.warmup)

    # --- E. traffic --------------------------------------------------------
    out["weight_traffic"] = weight_traffic(cfg)
    out["state_traffic"] = state_traffic(cfg, list(args.sublayer_contexts))

    # --- paired quality checkpoints ----------------------------------------
    # The systems rows above carry the quality level of the weights they timed
    # (S8). This block keeps the full checkpoint metadata for the gate.
    if ckdir is not None:
        for arm in ("HE", "attention-32h"):
            for seed in _checkpoint_seeds(ckdir, arm):
                p = ckdir / f"{arm}_seed{seed}.pt"
                if args.checkpoint_seed >= 0 and seed != args.checkpoint_seed:
                    continue
                try:
                    blob = torch.load(p, map_location="cpu", weights_only=False)
                except Exception as exc:  # noqa: BLE001
                    out["paired_quality"].setdefault(arm, {})[str(seed)] = {
                        "steps": None, "final_held_out_loss": None,
                        "checkpoint": str(p), "error": str(exc)}
                    continue
                out["paired_quality"].setdefault(arm, {})[str(seed)] = {
                    "steps": blob.get("steps"),
                    "final_held_out_loss": blob.get("final_held_out_loss"),
                    "checkpoint": str(p),
                }

    out["caveats"] = [
        KERNEL_CAVEAT,
        ("the attention KV cache is reallocated with torch.cat on every decoded "
         "token (O(T) copy per token), which inflates the attention decode "
         "latency relative to a preallocated cache (PROTOCOL.md §8.D, out of "
         "budget for this round)"),
        ("CPU memory rows are process-level; the CUDA rows are per-arm "
         "(see the 'scope' field of every memory block)"),
        ("traffic blocks are analytic formulas, not measurements"),
        ("rows whose quality block says 'architecture at initialisation' carry "
         "no quality level and must not be paired with a quality number"),
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"results written: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
