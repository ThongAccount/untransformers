"""Milestone 3 — measured systems comparison.

Measures, on identical hardware/software and in one process:

  A. forward training step  — latency, throughput, peak memory (batch 64 x l=128)
  B. full-forward prefill   — latency, throughput, peak memory vs context length
  C. incremental decoding   — per-token latency, throughput, cache bytes
  D. sublayer-1 scaling     — per-token cost and state bytes vs context length
                              (128 … 8192), where the full model cannot go
                              (the positional table caps it at l=128)
  E. weight traffic         — analytic parameter bytes per step/token, and the
                              measured state bytes each arm carries per token

Everything is reported as DATA with its conditions (device, threads, shape,
warmup, trials). No verdicts: the write-up is a separate step.

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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hextractor.harness import TrainConfig, build_model, build_optimizer  # noqa: E402
from hextractor.incremental import (  # noqa: E402
    AttentionState, HEState, ModelCache, attention_forward_incremental,
    he_forward_incremental, model_forward_incremental,
)
from hextractor.init import apply_init_policy  # noqa: E402
from hextractor.layers import AttentionSublayer, HESublayer  # noqa: E402
from hextractor.ops_count import count_ops_attention_fwd, count_ops_fwd  # noqa: E402


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
        return {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
    return {"peak_rss_bytes": _peak_rss_bytes(), "rss_bytes": _rss_bytes()}


def _median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def _pct(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return float(s[idx])


def _timeit(fn, device, trials: int, warmup: int) -> Dict:
    """Run fn `warmup` times (untimed), then `trials` timed. Returns seconds list."""
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(trials):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return {"times": times, "median": _median(times),
            "mean": float(sum(times) / len(times)) if times else float("nan"),
            "p10": _pct(times, 0.10), "p90": _pct(times, 0.90),
            "min": float(min(times)) if times else float("nan"),
            "trials": trials, "warmup": warmup}


# --------------------------------------------------------------------------
# A. training step
# --------------------------------------------------------------------------
def measure_training_step(arm: str, cfg: TrainConfig, device: torch.device,
                          trials: int, warmup: int, seed: int = 1) -> Dict:
    torch.manual_seed(seed)
    model = build_model(arm, cfg).to(device)
    apply_init_policy(model, seed=seed)
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

    _reset_peak(device)
    t = _timeit(one_step, device, trials, warmup)
    mem = _peak_mem(device)
    tokens = cfg.batch_size * cfg.l
    out = {
        "arm": arm,
        "step_seconds": t,
        "tokens_per_step": tokens,
        "tokens_per_second": tokens / t["median"] if t["median"] > 0 else None,
        "memory": mem,
        "mode": "train (dropout on)",
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# B. full-forward prefill
# --------------------------------------------------------------------------
def measure_prefill(arm: str, cfg: TrainConfig, device: torch.device,
                    context_lengths: List[int], trials: int, warmup: int,
                    seed: int = 1) -> List[Dict]:
    torch.manual_seed(seed)
    model = build_model(arm, cfg).to(device)
    apply_init_policy(model, seed=seed)
    model.eval()
    rows = []
    for T in context_lengths:
        ids = torch.randint(0, cfg.vocab, (1, T), device=device)
        _reset_peak(device)
        with torch.no_grad():
            t = _timeit(lambda: model(ids), device, trials, warmup)
        mem = _peak_mem(device)
        rows.append({"arm": arm, "context_tokens": T, "batch": 1,
                     "latency": t, "tokens_per_second": T / t["median"] if t["median"] > 0 else None,
                     "memory": mem})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------
# C. incremental decoding
# --------------------------------------------------------------------------
def measure_incremental_decode(arm: str, cfg: TrainConfig, device: torch.device,
                               context_lengths: List[int], gen_tokens: int,
                               trials: int, warmup: int, seed: int = 1) -> List[Dict]:
    torch.manual_seed(seed)
    model = build_model(arm, cfg).to(device)
    apply_init_policy(model, seed=seed)
    model.eval()
    rows = []
    for T in context_lengths:
        prompt = torch.randint(0, cfg.vocab, (1, T), device=device)
        step_times: List[float] = []
        cache_bytes = 0
        for trial in range(trials + warmup):
            cache = ModelCache.for_model(model)
            with torch.no_grad():
                model_forward_incremental(model, prompt, cache)
                cache_bytes = cache.state_bytes()
                if trial < warmup:
                    continue
                for _ in range(gen_tokens):
                    tok = torch.randint(0, cfg.vocab, (1,), device=device)
                    _sync(device)
                    t0 = time.perf_counter()
                    model_forward_incremental(model, tok[:, None], cache)
                    _sync(device)
                    step_times.append(time.perf_counter() - t0)
        rows.append({
            "arm": arm,
            "context_tokens": T,
            "generated_tokens": gen_tokens,
            "per_token_seconds": {
                "median": _median(step_times),
                "mean": float(sum(step_times) / len(step_times)) if step_times else None,
                "p10": _pct(step_times, 0.10),
                "p90": _pct(step_times, 0.90),
                "min": float(min(step_times)) if step_times else None,
                "samples": len(step_times),
            },
            "tokens_per_second": 1.0 / _median(step_times) if step_times else None,
            "cache_bytes_at_prompt": cache_bytes,
            "cache_bytes_per_layer": cache_bytes // cfg.n_layers,
        })
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------
# D. sublayer-1 scaling beyond the model's positional cap
# --------------------------------------------------------------------------
def measure_sublayer_scaling(arms: List[str], cfg: TrainConfig, device: torch.device,
                             context_lengths: List[int], trials: int, warmup: int,
                             seed: int = 1) -> List[Dict]:
    rows = []
    for arm in arms:
        torch.manual_seed(seed)
        if arm == "HE":
            sub = HESublayer(cfg.d, cfg.l, variant="HE").to(device)
        else:
            sub = AttentionSublayer(cfg.d, n_heads=cfg.n_heads).to(device)
        with torch.no_grad():
            for p in sub.parameters():
                p.copy_(torch.randn_like(p) * 0.01)
        sub.eval()
        for T in context_lengths:
            x = torch.randn(1, 1, cfg.d, device=device)
            state = HEState() if arm == "HE" else AttentionState()
            # warm the state up to the target context length
            with torch.no_grad():
                chunk = torch.randn(1, T - 1, cfg.d, device=device)
                if arm == "HE":
                    he_forward_incremental(sub, chunk, state, start=0)
                else:
                    attention_forward_incremental(sub, chunk, state, start=0)
            times = []
            for i in range(trials + warmup):
                _sync(device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    if arm == "HE":
                        he_forward_incremental(sub, x, state, start=T - 1)
                    else:
                        attention_forward_incremental(sub, x, state, start=T - 1)
                _sync(device)
                dt = time.perf_counter() - t0
                if i >= warmup:
                    times.append(dt)
            state_bytes = 0
            if arm == "HE":
                state_bytes = (cfg.l - 1) * cfg.d * 4
            else:
                state_bytes = 2 * T * cfg.d * 4
            rows.append({
                "arm": arm,
                "scope": "sublayer-1",
                "context_tokens": T,
                "per_token_seconds": {"median": _median(times), "min": float(min(times)),
                                      "samples": len(times)},
                "state_bytes_per_layer": state_bytes,
                "state_bytes_all_layers": state_bytes * cfg.n_layers,
                "analytic_mults_per_token": (
                    count_ops_fwd("HE", 1, cfg.d, cfg.l)[0] if arm == "HE"
                    else _attention_incremental_mults(cfg.d, cfg.n_heads, T)),
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
# E. weight traffic (analytic) + model inventory
# --------------------------------------------------------------------------
def weight_traffic(cfg: TrainConfig) -> Dict:
    rows = {}
    for arm in ("HE", "attention-32h"):
        model = build_model(arm, cfg)
        n_params = sum(p.numel() for p in model.parameters())
        param_bytes = n_params * 4                      # fp32
        rows[arm] = {
            "n_params": n_params,
            "parameter_bytes_fp32": param_bytes,
            "training_step_parameter_bytes_read": 3 * param_bytes,   # fwd + dgrad + wgrad
            "decode_step_parameter_bytes_read": param_bytes,          # one forward pass
            "note": "analytic: weights are streamed once per step; "
                    "both arms carry the same parameter count by construction",
        }
        del model
    return rows


# --------------------------------------------------------------------------
# environment manifest
# --------------------------------------------------------------------------
def env_manifest(device: torch.device) -> Dict:
    man = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": device.type,
        "cpu_count": os.cpu_count(),
    }
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
# main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["HE", "attention-32h"])
    ap.add_argument("--systems-arms", nargs="+", default=["HE", "attention-32h"],
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
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--gen-tokens", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--checkpoint-seed", type=int, default=-1)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if (args.device == "cuda" or
                                     (args.device == "auto" and torch.cuda.is_available()))
                          else "cpu")

    cfg = TrainConfig(d=args.d, l=args.l, n_layers=args.layers, ffn_width=args.ffn,
                      vocab=args.vocab, batch_size=args.batch, n_heads=args.heads)

    out = {
        "schema": "untransformers-systems-run/1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": env_manifest(device),
        "config": {"d": cfg.d, "l": cfg.l, "n_layers": cfg.n_layers,
                   "ffn_width": cfg.ffn_width, "vocab": cfg.vocab,
                   "batch_size": cfg.batch_size, "n_heads": cfg.n_heads,
                   "trials": args.trials, "warmup": args.warmup,
                   "gen_tokens": args.gen_tokens},
        "training_step": [],
        "prefill": [],
        "incremental_decode": [],
        "sublayer_scaling": [],
        "weight_traffic": {},
        "paired_quality": {},
    }

    print(f"[systems] device={device} arms={args.arms}", flush=True)

    for arm in args.arms:
        print(f"[training-step] {arm}", flush=True)
        out["training_step"].append(
            measure_training_step(arm, cfg, device, args.trials, args.warmup))

    for arm in args.arms:
        print(f"[prefill] {arm}", flush=True)
        out["prefill"] += measure_prefill(arm, cfg, device,
                                          [1, 8, 32, 64, 96, 128],
                                          args.trials, args.warmup)

    for arm in args.systems_arms:
        print(f"[incremental-decode] {arm}", flush=True)
        out["incremental_decode"] += measure_incremental_decode(
            arm, cfg, device, [1, 8, 32, 64, 96, 128], args.gen_tokens,
            max(1, args.trials // 2), args.warmup)

    print("[sublayer-scaling] HE attention-32h", flush=True)
    out["sublayer_scaling"] = measure_sublayer_scaling(
        ["HE", "attention-32h"], cfg, device,
        [128, 256, 512, 1024, 2048, 4096, 8192], args.trials, args.warmup)

    out["weight_traffic"] = weight_traffic(cfg)

    # Paired quality checkpoints (Milestone 3 gate: every performance number
    # is tied to a stated quality level).
    if args.checkpoint_dir:
        ckdir = Path(args.checkpoint_dir)
        for arm in ("HE", "attention-32h"):
            for seed in range(1, 8):
                p = ckdir / f"{arm}_seed{seed}.pt"
                if not p.exists():
                    continue
                if args.checkpoint_seed >= 0 and seed != args.checkpoint_seed:
                    continue
                blob = torch.load(p, map_location="cpu", weights_only=False)
                out["paired_quality"].setdefault(arm, {})[str(seed)] = {
                    "steps": blob.get("steps"),
                    "final_held_out_loss": blob.get("final_held_out_loss"),
                    "checkpoint": str(p),
                }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"results written: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
