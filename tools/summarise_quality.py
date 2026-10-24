#!/usr/bin/env python3
"""The pre-registered statistics for a quality run (protocol §3.2, §3.3, §2.3).

Everything here is fixed BEFORE the numbers land (Researcher's memo, commit
`dbc08fc`), and this script only computes — it never interprets.

It reports, for the **primary endpoint** (held-out loss at the final step) and
the **secondary endpoint** (mean of the last four eval points):

* the per-seed values of both arms (nothing is hidden);
* the paired differences `d_s = L_HE(s) - L_attn(s)` (sign convention stated);
* `d̄`, the sample SD with **ddof=1**, the SE `s_d/√n`, the 95% CI
  `d̄ ± t₀.₉₇₅,n₋₁ · s_d/√n` and the **MDD at 80% power**
  `(t₀.₉₇₅,n₋₁ + t₀.₈₀,n₋₁) · s_d/√n`;
* the per-arm mean and sample SD (ddof=1);
* perplexity `exp(loss)` per arm, the ratio `exp(d̄)` and the CI endpoints
  mapped through `exp()`;
* the final train loss per seed.

It applies the decision rule that was fixed in advance: the comparison is
**resolved** iff the 95% CI for `d̄` excludes 0; otherwise the output says
"not resolved at n = 3", prints the MDD and the number of seeds that would
have been needed at 80% power for the observed effect.

With `--eval-windows` it also reports (a) the **eval-set bootstrap** (§3.3:
1,000 resamples of the 512 eval windows, paired across arms) and (b) the
**book-aligned sensitivity analysis** (§2.3: the three fully unseen books,
exhaustive stride-1, labelled post-hoc and never a replacement for the
primary endpoint).

Usage:
    python tools/summarise_quality.py --file results/quality_gpu_full.json \
        --out results/quality_summary.json \
        --eval-windows results/eval_windows.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# Two-sided 95% t quantiles, df = 1..30 (t_{0.975, df}).
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
# One-sided 80% t quantiles, df = 1..30 (t_{0.80, df}).
T80 = [1.282, 1.061, 0.845, 0.765, 0.706, 0.666, 0.636, 0.613, 0.593, 0.577,
       0.563, 0.552, 0.542, 0.533, 0.525, 0.518, 0.512, 0.507, 0.502, 0.497,
       0.493, 0.489, 0.486, 0.483, 0.480, 0.477, 0.474, 0.471, 0.469, 0.466]
NORMAL_975 = 1.959964
NORMAL_80 = 0.841621


def t_quantile(table: Sequence[float], df: int, normal: float) -> float:
    if df <= 0:
        return float("nan")
    if df <= len(table):
        return table[df - 1]
    return normal


def mean_sd(values: Sequence[float]) -> Dict:
    n = len(values)
    m = sum(values) / n if n else None
    if n > 1:
        var = sum((x - m) ** 2 for x in values) / (n - 1)     # ddof=1
        sd = math.sqrt(var)
    else:
        sd = None
    return {"n": n, "mean": m, "std_ddof1": sd}


def paired_block(values_a: Dict[int, float], values_b: Dict[int, float]) -> Dict:
    """The pre-registered paired statistics for one endpoint."""
    seeds = sorted(set(values_a) & set(values_b))
    diffs = [values_a[s] - values_b[s] for s in seeds]
    n = len(diffs)
    out: Dict = {
        "seeds": seeds,
        "sign_convention": "d_s = L_HE(s) - L_attn(s); positive means HE's loss is HIGHER (worse)",
        "per_seed_arm_a": {str(s): values_a[s] for s in seeds},
        "per_seed_arm_b": {str(s): values_b[s] for s in seeds},
        "differences": {str(s): d for s, d in zip(seeds, diffs)},
        "arm_a": mean_sd([values_a[s] for s in seeds]),
        "arm_b": mean_sd([values_b[s] for s in seeds]),
    }
    if n == 0:
        out["resolved"] = False
        out["verdict"] = "no paired seeds"
        return out
    d_mean = sum(diffs) / n
    out["mean_difference"] = d_mean
    if n > 1:
        sd = math.sqrt(sum((d - d_mean) ** 2 for d in diffs) / (n - 1))
    else:
        sd = 0.0
    out["std_difference_ddof1"] = sd
    se = sd / math.sqrt(n)
    out["standard_error"] = se
    df = n - 1
    t975 = t_quantile(T975, df, NORMAL_975)
    t80 = t_quantile(T80, df, NORMAL_80)
    half = t975 * se
    out["t_critical_95"] = t975
    out["ci95"] = [d_mean - half, d_mean + half]
    out["ci95_excludes_zero"] = bool(d_mean - half > 0 or d_mean + half < 0)
    out["mdd_80_power"] = (t975 + t80) * se
    out["resolved"] = out["ci95_excludes_zero"]
    out["verdict"] = ("resolved (the 95% CI for d̄ excludes 0)" if out["resolved"]
                      else f"not resolved at n = {n}")
    # how many seeds would have been needed for 80% power at the observed effect
    if d_mean != 0.0:
        needed = None
        for cand in range(max(n, 2), 1001):
            se_c = sd / math.sqrt(cand)
            mdd_c = (t_quantile(T975, cand - 1, NORMAL_975) +
                     t_quantile(T80, cand - 1, NORMAL_80)) * se_c
            if mdd_c <= abs(d_mean):
                needed = cand
                break
        out["seeds_needed_for_80_power_at_observed_effect"] = needed
    else:
        out["seeds_needed_for_80_power_at_observed_effect"] = None
    # perplexity, with the interval mapped through exp()
    out["perplexity_arm_a"] = {str(s): math.exp(values_a[s]) for s in seeds}
    out["perplexity_arm_b"] = {str(s): math.exp(values_b[s]) for s in seeds}
    out["perplexity_ratio"] = math.exp(d_mean)
    out["perplexity_ratio_ci95"] = [math.exp(v) for v in out["ci95"]]
    return out


def endpoint_values(results: Dict, arm: str, seeds: Sequence[int],
                    endpoint: str) -> Dict[int, float]:
    """`final` = the held-out loss at the last step; `last4` = the mean of the
    last four eval points (1250, 1500, 1750, 2000 for the pre-registered run)."""
    out: Dict[int, float] = {}
    runs = {r["seed"]: r for r in (results.get(arm) or {}).get("runs", [])}
    for s in seeds:
        run = runs.get(s)
        if not run:
            continue
        evals = run.get("evals") or []
        if endpoint == "final":
            out[s] = float(run["final_held_out_loss"])
        else:
            tail = [float(e["held_out_loss"]) for e in evals[-4:]]
            out[s] = sum(tail) / len(tail)
    return out


def bootstrap_se(window_losses_a: Sequence[float], window_losses_b: Sequence[float],
                 n_resamples: int = 1000, rng_seed: int = 20261024) -> Dict:
    """The eval-set bootstrap of protocol §3.3.

    The 512 eval windows are resampled with replacement; the SAME resample is
    used for both arms, so the paired difference is bootstrapped jointly.
    """
    a = np.asarray(window_losses_a, dtype=np.float64)
    b = np.asarray(window_losses_b, dtype=np.float64)
    if a.size == 0 or a.size != b.size:
        return {"available": False,
                "reason": "the two arms do not have the same number of eval windows"}
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, a.size, size=(n_resamples, a.size))
    means_a = a[idx].mean(axis=1)
    means_b = b[idx].mean(axis=1)
    diffs = means_a - means_b
    return {
        "available": True,
        "n_resamples": n_resamples,
        "n_windows": int(a.size),
        "rng_seed": rng_seed,
        "bootstrap_se_arm_a": float(means_a.std(ddof=1)),
        "bootstrap_se_arm_b": float(means_b.std(ddof=1)),
        "bootstrap_se_paired_difference": float(diffs.std(ddof=1)),
        "note": ("this SE describes the sampling of the eval windows only; it is NOT "
                 "included in the training-noise CI (2.484·s_d) and the two are not "
                 "simply additive"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="the quality results JSON")
    ap.add_argument("--out", default="")
    ap.add_argument("--eval-windows", default="",
                    help="the checkpoint-eval JSON (eval-set bootstrap + book-aligned)")
    ap.add_argument("--arm-a", default="HE")
    ap.add_argument("--arm-b", default="attention-32h")
    ap.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = ap.parse_args()

    path = Path(args.file)
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") or {}
    proto = data.get("protocol") or {}
    seeds = list(proto.get("seeds") or [])
    if not seeds:
        raise SystemExit(f"FAIL: {path} carries no protocol.seeds")

    summary: Dict = {
        "schema": "untransformers-quality-summary/1",
        "source_results": str(path),
        "label": data.get("label"),
        "created_utc": data.get("created_utc"),
        "conditions": {
            "arms": proto.get("arms"),
            "seeds": seeds,
            "steps": proto.get("steps"),
            "tokens_per_step": proto.get("tokens_per_step"),
            "tokens_processed_per_run": proto.get("tokens_processed_per_run"),
            "nominal_epochs_of_train_region": proto.get("epochs_of_our_corpus"),
            "train_seed": proto.get("train_seed"),
            "eval_every": proto.get("eval_every"),
            "eval_batches": proto.get("eval_batches"),
            "corpus_id": (data.get("corpus") or {}).get("corpus_id"),
            "tokenizer_sha256": (data.get("corpus") or {}).get("tokenizer_sha256"),
            "device": (data.get("environment") or {}).get("device"),
            "torch": (data.get("environment") or {}).get("torch"),
            "python": (data.get("environment") or {}).get("python"),
            "commit": data.get("commit"),
        },
        "endpoints": {},
    }

    for endpoint, label in (("final", "primary: held-out loss at the final step"),
                            ("last4", "secondary: mean of the last four eval points")):
        va = endpoint_values(results, args.arm_a, seeds, endpoint)
        vb = endpoint_values(results, args.arm_b, seeds, endpoint)
        block = paired_block(va, vb)
        block["definition"] = label
        summary["endpoints"][endpoint] = block

    # final train loss per seed (underfitting / divergence visible)
    train_final = {}
    for arm in (args.arm_a, args.arm_b):
        runs = {r["seed"]: r for r in (results.get(arm) or {}).get("runs", [])}
        train_final[arm] = {str(s): float(runs[s]["train_curve"][-1])
                            for s in seeds if s in runs and runs[s].get("train_curve")}
    summary["final_train_loss"] = train_final

    # the full curves (protocol §3.2: "the full per-seed eval curve for both
    # arms (all 8 eval points), so the reader can see whether the final point
    # is a plateau or a spike")
    curves = {}
    for arm in (args.arm_a, args.arm_b):
        runs = {r["seed"]: r for r in (results.get(arm) or {}).get("runs", [])}
        curves[arm] = {
            str(s): {
                "evals": [{"step": int(e["step"]),
                           "held_out_loss": float(e["held_out_loss"])}
                          for e in (runs[s].get("evals") or [])],
                "train_curve": [float(v) for v in (runs[s].get("train_curve") or [])],
            }
            for s in seeds if s in runs
        }
    summary["curves"] = curves

    # --- eval-set bootstrap + book-aligned sensitivity ----------------------
    if args.eval_windows:
        ev = json.loads(Path(args.eval_windows).read_text(encoding="utf-8"))
        eval_set = ev.get("eval_set") or {}
        boot = {}
        for s in seeds:
            key_a = f"{args.arm_a}_seed{s}"
            key_b = f"{args.arm_b}_seed{s}"
            if key_a in eval_set and key_b in eval_set:
                boot[str(s)] = bootstrap_se(
                    eval_set[key_a].get("per_window_loss") or [],
                    eval_set[key_b].get("per_window_loss") or [],
                    args.bootstrap_resamples)
        summary["eval_set_bootstrap"] = {
            "per_seed": boot,
            "definition": ("the 512 eval windows are resampled with replacement "
                           "(1,000 resamples); the same resample is used for both arms"),
            "recomputed_vs_reported": {
                key: (eval_set.get(key) or {}).get("recomputed_minus_reported")
                for key in eval_set
            },
        }

        aligned = ev.get("book_aligned") or {}
        overall = aligned.get("overall") or {}
        va = {s: overall[f"{args.arm_a}_seed{s}"]["mean_loss"]
              for s in seeds if f"{args.arm_a}_seed{s}" in overall}
        vb = {s: overall[f"{args.arm_b}_seed{s}"]["mean_loss"]
              for s in seeds if f"{args.arm_b}_seed{s}" in overall}
        block = paired_block(va, vb)
        block["definition"] = ("post-hoc sensitivity analysis: the three fully unseen "
                               "books (ebooks 3536, 3795, 3796), exhaustive stride-1 "
                               "sweep; reported ALONGSIDE the primary endpoint and may "
                               "not replace it")
        block["per_book"] = {key: aligned.get("books", {}).get(key)
                             for key in (f"{args.arm_a}_seed{s}" for s in seeds)}
        summary["book_aligned_sensitivity"] = block

    text = json.dumps(summary, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"written: {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
