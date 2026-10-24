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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# the Student-t quantile, COMPUTED rather than tabulated
# --------------------------------------------------------------------------
# The protocol (§3.2) fixes the statistics, not a table: the 95% CI is
# `d̄ ± 2.484 · s_d` (t₀.₉₇₅,₂ = 4.3027) and the MDD is `3.097 · s_d`
# ((t₀.₉₇₅,₂ + t₀.₈₀,₂)/√3, t₀.₈₀,₂ = 1.0607). Those are the exact quantiles
# rounded to four decimals.
#
# This builder used to read them from a three-decimal table. Its 95% column
# was right, but its 80% column was wrong for 29 of its 30 rows (it listed the
# 80% *normal* quantile and one wrong entry, e.g. 0.845 at df = 3 where the
# exact value is 0.978), and the "seeds needed for 80% power" loop used that
# column — understating the seeds required by 10-24%. The quantiles are now
# inverted from the Student-t CDF numerically, and `verify_protocol_constants`
# fails the run if the computed values stop reproducing the protocol's stated
# constants.
def _betacf(a: float, b: float, x: float, itmax: int = 300,
            eps: float = 3e-16) -> float:
    """The continued fraction of the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """The regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
              + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: int) -> float:
    """P(T <= t) for a Student-t with `df` degrees of freedom."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    tail = 0.5 * _betai(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


_T_QUANTILE_CACHE: Dict[Tuple[float, int], float] = {}


def student_t_ppf(p: float, df: int) -> float:
    """The Student-t quantile t_{p, df}, by bisection on the CDF.

    Exact to ~1e-12 (the bracket is halved 200 times), so the protocol's
    constants are reproduced rather than rounded into them.
    """
    if df <= 0 or not (0.0 < p < 1.0):
        return float("nan")
    key = (p, df)
    cached = _T_QUANTILE_CACHE.get(key)
    if cached is not None:
        return cached
    lo, hi = 0.0, 1.0
    while student_t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    q = 0.5 * (lo + hi)
    _T_QUANTILE_CACHE[key] = q
    return q


# the constants the protocol states, with the precision at which it states them
PROTOCOL_CONSTANTS: Dict[str, Dict[str, float]] = {
    "t_975_df2": {"stated": 4.3027, "tolerance": 5e-5},
    "t_80_df2": {"stated": 1.0607, "tolerance": 5e-5},
    "ci_half_width_constant": {"stated": 2.484, "tolerance": 5e-4},
    "mdd_constant": {"stated": 3.097, "tolerance": 5e-4},
}


def verify_protocol_constants() -> Dict[str, Dict[str, float]]:
    """Check the computed quantiles against the protocol's stated constants.

    If this fails, the statistics have drifted from the pre-registration and
    the run must stop rather than print numbers the protocol does not claim.
    """
    t975 = student_t_ppf(0.975, 2)
    t80 = student_t_ppf(0.80, 2)
    computed = {
        "t_975_df2": t975,
        "t_80_df2": t80,
        "ci_half_width_constant": t975 / math.sqrt(3),
        "mdd_constant": (t975 + t80) / math.sqrt(3),
    }
    out: Dict[str, Dict[str, float]] = {}
    for key, spec in PROTOCOL_CONSTANTS.items():
        diff = computed[key] - spec["stated"]
        out[key] = {
            "stated": spec["stated"],
            "computed": computed[key],
            "difference": diff,
            "tolerance": spec["tolerance"],
            "ok": abs(diff) <= spec["tolerance"],
        }
    return out


def _eval_windows(results: Dict) -> Optional[int]:
    """The number of windows in one eval point: eval batches x batch size."""
    batch = (results.get("config") or {}).get("batch_size")
    batches = (results.get("protocol") or {}).get("eval_batches")
    if batch and batches:
        return int(batch) * int(batches)
    return None


def _eval_target_tokens(results: Dict) -> Optional[int]:
    """The target tokens in one eval point: windows x the context length."""
    windows = _eval_windows(results)
    l_ctx = (results.get("config") or {}).get("l")
    if windows and l_ctx:
        return int(windows) * int(l_ctx)
    return None


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
        se = sd / math.sqrt(n)
        df = n - 1
        t975 = student_t_ppf(0.975, df)
        t80 = student_t_ppf(0.80, df)
        half = t975 * se
        out["std_difference_ddof1"] = sd
        out["standard_error"] = se
        out["t_critical_95"] = t975
        out["t_critical_80"] = t80
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
                mdd_c = (student_t_ppf(0.975, cand - 1) +
                         student_t_ppf(0.80, cand - 1)) * se_c
                if mdd_c <= abs(d_mean):
                    needed = cand
                    break
            out["seeds_needed_for_80_power_at_observed_effect"] = needed
        else:
            out["seeds_needed_for_80_power_at_observed_effect"] = None
    else:
        # one seed: the spread, the interval and the MDD are undefined. They are
        # written as null (never NaN, which is not valid JSON).
        out["std_difference_ddof1"] = None
        out["standard_error"] = None
        out["t_critical_95"] = None
        out["t_critical_80"] = None
        out["ci95"] = [None, None]
        out["ci95_excludes_zero"] = False
        out["mdd_80_power"] = None
        out["resolved"] = False
        out["verdict"] = f"not resolved at n = {n} (a single seed has no interval)"
        out["seeds_needed_for_80_power_at_observed_effect"] = None
    # perplexity, with the interval mapped through exp()
    out["perplexity_arm_a"] = {str(s): math.exp(values_a[s]) for s in seeds}
    out["perplexity_arm_b"] = {str(s): math.exp(values_b[s]) for s in seeds}
    out["perplexity_ratio"] = math.exp(d_mean)
    # a null bound stays null: exp(None) is a TypeError, not a number
    out["perplexity_ratio_ci95"] = [None if v is None else math.exp(v)
                                    for v in out["ci95"]]
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

    # The protocol's stated constants must still come out of the computed
    # quantiles; if they do not, the statistics have drifted from the
    # pre-registration and nothing is printed.
    constants = verify_protocol_constants()
    drifted = {k: v for k, v in constants.items() if not v["ok"]}
    if drifted:
        for key, row in drifted.items():
            print(f"FAIL: the computed {key} = {row['computed']!r} does not reproduce "
                  f"the protocol's stated {row['stated']!r} (difference "
                  f"{row['difference']!r}, tolerance {row['tolerance']!r})")
        return 1

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
            "context_length_l": (data.get("config") or {}).get("l"),
            "batch_size": (data.get("config") or {}).get("batch_size"),
            "eval_windows": _eval_windows(data),
            "eval_target_tokens": _eval_target_tokens(data),
            "corpus_id": (data.get("corpus") or {}).get("corpus_id"),
            "tokenizer_sha256": (data.get("corpus") or {}).get("tokenizer_sha256"),
            "device": (data.get("environment") or {}).get("device"),
            "torch": (data.get("environment") or {}).get("torch"),
            "python": (data.get("environment") or {}).get("python"),
            "commit": data.get("commit"),
        },
        "endpoints": {},
        "protocol_constants": constants,
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
