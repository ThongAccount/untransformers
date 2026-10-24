#!/usr/bin/env python3
"""Reviewer-side independent re-computation for the Job-B verdict.

**This is not a gate.** Nothing in `GATES.md` depends on it, and it is not
part of the builder's pipeline. It exists so that the reviewer's verdict is a
*command with an exit code* rather than prose: every number the review
publishes is recomputed here, from the raw artefacts, by code that does not
import the builder's statistics module.

What it recomputes (all of it pre-registered in `docs/milestone-3/PROTOCOL.md`):

* the paired per-seed differences, the sample SD with **ddof=1**, the 95% CI,
  the MDD at 80% power, the perplexity ratio and its interval — for the primary
  endpoint (final step) and the secondary one (mean of the last four eval
  points);
* the t quantiles themselves, by inverting the Student-t CDF numerically, so
  the reviewer's interval does not inherit the builder's rounded t table;
* the eval-set bootstrap SE (same resample stream as the builder's, on
  purpose: the comparison then isolates the arithmetic, not the resample);
* the book-aligned sensitivity block;
* the budget arithmetic (tokens processed, nominal epochs, the fraction of
  the paper's token budget);
* the run-order null and the drift floor of the systems artefact, recomputed
  **per (arm, context)** from the raw slot rows, with a resolvability verdict
  for every arm pair at every context.

Usage:
    python tools/review_recompute.py --file results/quality_gpu_full.json \
        --summary results/quality_summary.json \
        --eval-windows results/eval_windows.json \
        --systems results/systems.json \
        --out results/review_recompute.json

Exit 0 prints `REVIEW_RECOMPUTE_OK`; exit 1 prints `REVIEW_RECOMPUTE_FAIL`
and lists the checks that failed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# The paper's budget: 60,000 batches of 64 x 128 tokens (arXiv:2308.07661v2,
# Table 1). Used only to express our budget as a fraction of theirs.
PAPER_TRAIN_TOKENS = 60_000 * 64 * 128

ARM_A = "HE"
ARM_B = "attention-32h"
ARM_C = "attention-32h-sdpa"

# the repository root, for the source-level checks (which key the renderer reads)
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Student-t quantiles, computed here rather than taken from a table
# ---------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    maxit, eps, fpmin = 500, 3.0e-16, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
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
    x = df / (df + t * t)
    tail = 0.5 * _betai(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def student_t_ppf(p: float, df: int) -> float:
    """Inverse Student-t CDF by bisection (accuracy ~1e-12)."""
    if df <= 0:
        return float("nan")
    lo, hi = -1.0e4, 1.0e4
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# the pre-registered statistics, recomputed independently
# ---------------------------------------------------------------------------
def endpoint_values(results: Dict, arm: str, seeds: Sequence[int], endpoint: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    runs = {r["seed"]: r for r in (results.get(arm) or {}).get("runs", [])}
    for s in seeds:
        run = runs.get(s)
        if not run:
            continue
        if endpoint == "final":
            out[s] = float(run["final_held_out_loss"])
        else:
            tail = [float(e["held_out_loss"]) for e in (run.get("evals") or [])[-4:]]
            out[s] = sum(tail) / len(tail)
    return out


def mean_sd(values: Sequence[float]) -> Dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std_ddof1": None}
    m = sum(values) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in values) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": m, "std_ddof1": sd}


def paired_stats(values_a: Dict[int, float], values_b: Dict[int, float]) -> Dict:
    seeds = sorted(set(values_a) & set(values_b))
    diffs = [values_a[s] - values_b[s] for s in seeds]
    n = len(diffs)
    out: Dict = {
        "seeds": seeds,
        "n": n,
        "sign_convention": "d_s = L_HE(s) - L_attn(s); positive means HE's loss is HIGHER (worse)",
        "per_seed_arm_a": {str(s): values_a[s] for s in seeds},
        "per_seed_arm_b": {str(s): values_b[s] for s in seeds},
        "differences": {str(s): d for s, d in zip(seeds, diffs)},
        "arm_a": mean_sd([values_a[s] for s in seeds]),
        "arm_b": mean_sd([values_b[s] for s in seeds]),
    }
    if n == 0:
        out["verdict"] = "no paired seeds"
        out["resolved"] = False
        return out
    d_mean = sum(diffs) / n
    out["mean_difference"] = d_mean
    if n > 1:
        sd = math.sqrt(sum((d - d_mean) ** 2 for d in diffs) / (n - 1))
        se = sd / math.sqrt(n)
        t975 = student_t_ppf(0.975, n - 1)
        t80 = student_t_ppf(0.80, n - 1)
        half = t975 * se
        out.update({
            "std_difference_ddof1": sd,
            "standard_error": se,
            "t_critical_95": t975,
            "t_critical_80": t80,
            "ci95": [d_mean - half, d_mean + half],
            "ci95_excludes_zero": bool(d_mean - half > 0 or d_mean + half < 0),
            "mdd_80_power": (t975 + t80) * se,
            "resolved": bool(d_mean - half > 0 or d_mean + half < 0),
            "verdict": ("resolved (the 95% CI for d-bar excludes 0)"
                        if (d_mean - half > 0 or d_mean + half < 0)
                        else f"not resolved at n = {n}"),
            "perplexity_ratio": math.exp(d_mean),
            "perplexity_ratio_ci95": [math.exp(d_mean - half), math.exp(d_mean + half)],
        })
        needed = None
        if d_mean != 0.0:
            for cand in range(max(n, 2), 1001):
                se_c = sd / math.sqrt(cand)
                mdd_c = (student_t_ppf(0.975, cand - 1) + student_t_ppf(0.80, cand - 1)) * se_c
                if mdd_c <= abs(d_mean):
                    needed = cand
                    break
        out["seeds_needed_for_80_power_at_observed_effect"] = needed
    else:
        out.update({"std_difference_ddof1": None, "standard_error": None,
                    "t_critical_95": None, "ci95": [None, None],
                    "ci95_excludes_zero": False, "mdd_80_power": None,
                    "resolved": False,
                    "verdict": f"not resolved at n = {n} (a single seed has no interval)",
                    "perplexity_ratio": math.exp(d_mean),
                    "perplexity_ratio_ci95": [None, None],
                    "seeds_needed_for_80_power_at_observed_effect": None})
    return out


def bootstrap_se(window_losses_a: Sequence[float], window_losses_b: Sequence[float],
                 n_resamples: int = 1000, rng_seed: int = 20261024) -> Dict:
    """The eval-set bootstrap of protocol §3.3 (same resample stream as the
    builder's, deliberately: the comparison then isolates the arithmetic)."""
    a = np.asarray(window_losses_a, dtype=np.float64)
    b = np.asarray(window_losses_b, dtype=np.float64)
    if a.size == 0 or a.size != b.size:
        return {"available": False, "reason": "the two arms do not have the same number of eval windows"}
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
    }


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
class Checks:
    def __init__(self) -> None:
        self.rows: List[Dict] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    def close(self, name: str, mine: Optional[float], theirs: Optional[float],
              rel: float = 1e-9, abs_: float = 1e-12) -> bool:
        if mine is None or theirs is None:
            return self.add(name, mine is None and theirs is None,
                            f"reviewer={mine!r} artefact={theirs!r} (one of them is null)")
        ok = math.isclose(mine, theirs, rel_tol=rel, abs_tol=abs_)
        return self.add(name, ok, f"reviewer={mine!r} artefact={theirs!r} "
                                  f"abs_diff={abs(mine - theirs):.3e}")

    @property
    def failed(self) -> List[Dict]:
        return [r for r in self.rows if not r["ok"]]


# ---------------------------------------------------------------------------
# systems: the null and the floor, recomputed per (arm, context)
# ---------------------------------------------------------------------------
def _median_of(row: Dict) -> Optional[float]:
    for key in ("latency", "step_seconds", "latency_stats", "per_token_seconds",
                "steady_state_per_token_seconds"):
        block = row.get(key)
        if isinstance(block, dict) and block.get("median") is not None:
            return float(block["median"])
    return None


def systems_null(systems: Dict) -> Dict:
    """Recompute the A/A null per (arm, context) from the raw slot rows.

    The null is the same arm measured at the start and at the end of the
    palindromic order **at the same context**. The floor for a context is the
    largest such spread over the arms at that context (the conservative
    choice, stated before the numbers landed).
    """
    out: Dict = {}
    blocks = (("training_step", "training_step_slots", None),
              ("prefill", "prefill_slots", "context_tokens"),
              ("incremental_decode", "incremental_decode_slots", "context_tokens"))
    for block, slots_key, ctx_key in blocks:
        slots = systems.get(slots_key) or []
        if not slots:
            continue
        groups: Dict = {}
        for r in slots:
            key = (r.get("arm"), r.get(ctx_key) if ctx_key else None)
            groups.setdefault(key, []).append(r)
        per_context: Dict = {}
        for (arm, ctx), rows in groups.items():
            rows = sorted(rows, key=lambda r: r.get("order_index", 0))
            first = _median_of(rows[0])
            last = _median_of(rows[-1])
            if first is None or last is None:
                continue
            per_context.setdefault(str(ctx), {})[arm] = {
                "first_slot_median_seconds": first,
                "last_slot_median_seconds": last,
                "absolute_difference_seconds": abs(last - first),
            }
        floors = {ctx: max((v["absolute_difference_seconds"] for v in arms.values()),
                           default=0.0)
                  for ctx, arms in per_context.items()}
        artefact_floor = ((systems.get("order_effect") or {}).get(block) or {}).get(
            "drift_floor_seconds")
        same_context_max = max(floors.values()) if floors else None
        out[block] = {
            "null_per_arm_and_context": per_context,
            "floor_per_context_seconds": floors,
            "artefact_block_floor_seconds": artefact_floor,
            "reviewer_max_same_context_floor_seconds": same_context_max,
        }
    return out


def systems_verdicts(systems: Dict, nulls: Dict) -> List[Dict]:
    """One verdict per arm pair per context: resolvable iff |difference| > floor."""
    rows: List[Dict] = []
    pairs = ((ARM_A, ARM_B), (ARM_A, ARM_C), (ARM_B, ARM_C))
    for block, pooled_key in (("training_step", "training_step"),
                              ("prefill", "prefill"),
                              ("incremental_decode", "incremental_decode")):
        pooled = systems.get(pooled_key) or []
        if not pooled:
            continue
        block_nulls = (nulls.get(block) or {}).get("floor_per_context_seconds") or {}
        by_arm_ctx: Dict = {}
        for r in pooled:
            ctx = str(r.get("context_tokens"))
            med = _median_of(r)
            if med is None:
                continue
            by_arm_ctx.setdefault(ctx, {})[r.get("arm")] = med
        for ctx, arms in sorted(by_arm_ctx.items(), key=lambda kv: (len(kv[0]), kv[0])):
            floor = block_nulls.get(ctx)
            for a1, a2 in pairs:
                if a1 not in arms or a2 not in arms:
                    continue
                diff = abs(arms[a1] - arms[a2])
                verdict = ("resolvable" if (floor is not None and diff > floor)
                           else "not resolvable")
                rows.append({"block": block, "context_tokens": ctx,
                             "arm_a": a1, "arm_b": a2,
                             "difference_seconds": diff,
                             "floor_seconds": floor, "verdict": verdict})
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="the quality results JSON")
    ap.add_argument("--summary", default="", help="the builder's quality summary JSON")
    ap.add_argument("--eval-windows", default="", help="the checkpoint-eval JSON")
    ap.add_argument("--systems", default="", help="the systems artefact JSON")
    ap.add_argument("--out", default="", help="write the reviewer's recomputation here")
    args = ap.parse_args()

    checks = Checks()
    review: Dict = {"schema": "untransformers-review-recompute/1",
                     "source_results": str(args.file),
                     "source_summary": args.summary or None,
                     "source_eval_windows": args.eval_windows or None,
                     "source_systems": args.systems or None}

    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    results = data.get("results") or {}
    proto = data.get("protocol") or {}
    seeds = list(proto.get("seeds") or [])
    checks.add("the results file carries protocol.seeds", bool(seeds), f"seeds={seeds}")

    # --- budget arithmetic -------------------------------------------------
    steps = proto.get("steps")
    tps = proto.get("tokens_per_step")
    train_region = proto.get("corpus_tokens_train_region")
    if steps and tps and train_region:
        tokens = steps * tps
        epochs = tokens / train_region
        checks.close("tokens processed per run", float(tokens),
                     float(proto.get("tokens_processed_per_run") or 0))
        checks.close("nominal epochs of the train region", epochs,
                     float(proto.get("epochs_of_our_corpus") or 0))
        review["budget"] = {
            "steps": steps, "tokens_per_step": tps,
            "tokens_processed_per_run": tokens,
            "nominal_epochs_of_train_region": epochs,
            "paper_train_tokens": PAPER_TRAIN_TOKENS,
            "fraction_of_the_paper_token_budget": tokens / PAPER_TRAIN_TOKENS,
        }
        checks.add("the token budget is a fraction of the paper's",
                   0 < tokens / PAPER_TRAIN_TOKENS < 1,
                   f"{tokens / PAPER_TRAIN_TOKENS:.6f} of the paper's {PAPER_TRAIN_TOKENS} tokens")

    # --- endpoints ---------------------------------------------------------
    recomputed: Dict = {}
    for endpoint in ("final", "last4"):
        va = endpoint_values(results, ARM_A, seeds, endpoint)
        vb = endpoint_values(results, ARM_B, seeds, endpoint)
        block = paired_stats(va, vb)
        recomputed[endpoint] = block
        checks.add(f"{endpoint}: every seed has both arms",
                   len(block["seeds"]) == len(seeds),
                   f"paired seeds={block['seeds']} requested={seeds}")
        review[f"endpoint_{endpoint}"] = block

    # the artefact's own paired_final block
    paired = data.get("paired_final") or {}
    if paired:
        # The merge writes `differences_arm_a_minus_arm_b` and the renderer reads
        # that key. The reviewer's second pass found the renderer reading a key
        # the merge never writes, which emptied the per-seed difference column
        # of QUALITY.md; it was fixed before any measured number existed. Check
        # the key the renderer depends on, and that no cell would be empty.
        diffs = paired.get("differences_arm_a_minus_arm_b") or {}
        checks.add("the artefact carries the key the renderer reads "
                   "('differences_arm_a_minus_arm_b')",
                   bool(diffs),
                   "the merge writes 'differences_arm_a_minus_arm_b' and the renderer "
                   "reads it; without it the rendered per-seed difference column is empty")
        for s in seeds:
            checks.add(f"paired_final carries a difference for seed {s}",
                       diffs.get(str(s)) is not None,
                       f"difference={diffs.get(str(s))!r}")
        renderer = REPO_ROOT / "tools" / "render_results_tables.py"
        renderer_src = renderer.read_text(encoding="utf-8") if renderer.exists() else ""
        checks.add("the renderer reads the key the merge writes",
                   "differences_arm_a_minus_arm_b" in renderer_src,
                   "tools/render_results_tables.py must read "
                   "'differences_arm_a_minus_arm_b', or the per-seed difference "
                   "column of QUALITY.md renders empty")
        for s in seeds:
            mine = recomputed["final"]["differences"].get(str(s))
            theirs = diffs.get(str(s))
            checks.close(f"paired_final difference, seed {s}", mine, theirs)
        checks.close("paired_final mean difference", recomputed["final"]["mean_difference"],
                     paired.get("mean_difference"))
        sd_reported = paired.get("std_difference")
        sd_mine = recomputed["final"]["std_difference_ddof1"]
        diffs_all = [recomputed["final"]["differences"][k]
                     for k in recomputed["final"]["differences"]]
        if sd_reported is not None and diffs_all:
            m = recomputed["final"]["mean_difference"]
            ss = sum((d - m) ** 2 for d in diffs_all)
            sd_population = math.sqrt(ss / len(diffs_all))
            sd_sample = math.sqrt(ss / (len(diffs_all) - 1)) if len(diffs_all) > 1 else None
            if math.isclose(sd_reported, sd_population, rel_tol=1e-6):
                kind = "population SD (ddof=0)"
            elif sd_sample is not None and math.isclose(sd_reported, sd_sample, rel_tol=1e-6):
                kind = "sample SD (ddof=1)"
            else:
                kind = f"neither ddof=0 ({sd_population!r}) nor ddof=1 ({sd_sample!r})"
            review["paired_final_sd_classification"] = kind
            checks.add("paired_final.std_difference is identified as population or sample",
                       kind.startswith("population") or kind.startswith("sample"),
                       f"{kind}; at n={len(diffs_all)} the population SD understates the "
                       f"sample SD by a factor sqrt(n/(n-1)) = "
                       f"{math.sqrt(len(diffs_all) / max(len(diffs_all) - 1, 1)):.4f}")

    # --- the builder's summary --------------------------------------------
    summary = {}
    if args.summary:
        summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        for endpoint in ("final", "last4"):
            theirs = (summary.get("endpoints") or {}).get(endpoint) or {}
            mine = recomputed[endpoint]
            checks.close(f"{endpoint}: mean difference", mine.get("mean_difference"),
                         theirs.get("mean_difference"))
            checks.close(f"{endpoint}: sample SD (ddof=1)", mine.get("std_difference_ddof1"),
                         theirs.get("std_difference_ddof1"))
            checks.close(f"{endpoint}: standard error", mine.get("standard_error"),
                         theirs.get("standard_error"))
            ci_mine, ci_theirs = mine.get("ci95") or [None, None], theirs.get("ci95") or [None, None]
            for i, label in enumerate(("lower", "upper")):
                checks.close(f"{endpoint}: 95% CI {label} bound", ci_mine[i], ci_theirs[i],
                             rel=1e-4, abs_=1e-9)
            checks.close(f"{endpoint}: MDD at 80% power", mine.get("mdd_80_power"),
                         theirs.get("mdd_80_power"), rel=1e-4, abs_=1e-9)
            checks.add(f"{endpoint}: the resolution verdict agrees",
                       bool(mine.get("resolved")) == bool(theirs.get("resolved")),
                       f"reviewer={mine.get('verdict')!r} artefact={theirs.get('verdict')!r}")
            mine_needed = mine.get("seeds_needed_for_80_power_at_observed_effect")
            their_needed = theirs.get("seeds_needed_for_80_power_at_observed_effect")
            if mine_needed is not None and their_needed is not None:
                # exact now: both sides invert the Student-t CDF numerically, so
                # any difference means one of them still uses a table
                checks.add(f"{endpoint}: the seeds-needed figure agrees exactly",
                           int(mine_needed) == int(their_needed),
                           f"reviewer={mine_needed} artefact={their_needed}")
            elif (mine_needed is None) != (their_needed is None):
                checks.add(f"{endpoint}: the seeds-needed figure agrees",
                           False, f"reviewer={mine_needed!r} artefact={their_needed!r}")
            checks.close(f"{endpoint}: perplexity ratio", mine.get("perplexity_ratio"),
                         theirs.get("perplexity_ratio"), rel=1e-9)
            checks.close(f"{endpoint}: the builder's t critical value",
                         mine.get("t_critical_95"), theirs.get("t_critical_95"),
                         rel=1e-3, abs_=1e-4)
            checks.close(f"{endpoint}: the builder's 80% t critical value",
                         mine.get("t_critical_80"), theirs.get("t_critical_80"),
                         rel=1e-6, abs_=1e-9)
            review[f"summary_agreement_{endpoint}"] = {
                "reviewer": mine, "artefact": theirs}

        # the protocol's stated constants, as the builder computed them
        constants = summary.get("protocol_constants") or {}
        if constants:
            t975_2 = student_t_ppf(0.975, 2)
            t80_2 = student_t_ppf(0.80, 2)
            for key, expected in (("t_975_df2", t975_2), ("t_80_df2", t80_2)):
                checks.close(f"protocol constant {key} (the builder's computation)",
                             expected, (constants.get(key) or {}).get("computed"),
                             rel=1e-9, abs_=1e-12)
            checks.close("protocol constant: the CI half-width constant",
                         t975_2 / math.sqrt(3),
                         (constants.get("ci_half_width_constant") or {}).get("computed"),
                         rel=1e-9, abs_=1e-12)
            checks.close("protocol constant: the MDD constant",
                         (t975_2 + t80_2) / math.sqrt(3),
                         (constants.get("mdd_constant") or {}).get("computed"),
                         rel=1e-9, abs_=1e-12)
            checks.add("the builder's protocol constants all reproduce",
                       all((row.get("ok") is True) for row in constants.values()),
                       "verify_protocol_constants() refuses to print if the computed "
                       "quantiles stop reproducing the protocol's stated constants")
            review["protocol_constants_recomputed"] = constants

    # --- eval windows: bootstrap + book-aligned sensitivity ----------------
    if args.eval_windows:
        ev = json.loads(Path(args.eval_windows).read_text(encoding="utf-8"))
        eval_set = ev.get("eval_set") or {}
        boot_recomputed: Dict = {}
        for s in seeds:
            ka, kb = f"{ARM_A}_seed{s}", f"{ARM_B}_seed{s}"
            if ka not in eval_set or kb not in eval_set:
                continue
            la = eval_set[ka].get("per_window_loss") or []
            lb = eval_set[kb].get("per_window_loss") or []
            checks.add(f"eval windows, seed {s}: both arms have the same number of windows",
                       len(la) == len(lb), f"{len(la)} vs {len(lb)}")
            boot_recomputed[str(s)] = bootstrap_se(la, lb)
            reported = ((summary.get("eval_set_bootstrap") or {}).get("per_seed") or {}).get(str(s))
            if reported:
                for key in ("bootstrap_se_arm_a", "bootstrap_se_arm_b",
                            "bootstrap_se_paired_difference"):
                    checks.close(f"bootstrap SE ({key}), seed {s}",
                                 boot_recomputed[str(s)].get(key), reported.get(key))
        review["eval_set_bootstrap_recomputed"] = boot_recomputed

        aligned = (ev.get("book_aligned") or {}).get("overall") or {}
        va = {s: aligned[f"{ARM_A}_seed{s}"]["mean_loss"] for s in seeds if f"{ARM_A}_seed{s}" in aligned}
        vb = {s: aligned[f"{ARM_B}_seed{s}"]["mean_loss"] for s in seeds if f"{ARM_B}_seed{s}" in aligned}
        sens = paired_stats(va, vb)
        review["book_aligned_sensitivity_recomputed"] = sens
        theirs_sens = summary.get("book_aligned_sensitivity") or {}
        if theirs_sens:
            checks.close("book-aligned: mean difference", sens.get("mean_difference"),
                         theirs_sens.get("mean_difference"))
            checks.close("book-aligned: sample SD (ddof=1)", sens.get("std_difference_ddof1"),
                         theirs_sens.get("std_difference_ddof1"))
            checks.add("book-aligned: the resolution verdict agrees",
                       bool(sens.get("resolved")) == bool(theirs_sens.get("resolved")),
                       f"reviewer={sens.get('verdict')!r} artefact={theirs_sens.get('verdict')!r}")
        ba = (ev.get("book_aligned") or {})
        for key, block in (ba.get("books") or {}).items():
            if block.get("stride") is not None:
                checks.add(f"book-aligned {key}: the sweep is stride 1",
                           int(block.get("stride")) == 1, f"stride={block.get('stride')}")
                break

    # --- systems: null, floor, verdicts ------------------------------------
    if args.systems:
        systems = json.loads(Path(args.systems).read_text(encoding="utf-8"))
        nulls = systems_null(systems)
        verdicts = systems_verdicts(systems, nulls)
        review["systems_null_recomputed"] = nulls
        review["systems_verdicts"] = verdicts
        order_effect = systems.get("order_effect") or {}
        for block, entry in nulls.items():
            artefact_floor = entry.get("artefact_block_floor_seconds")
            same_max = entry.get("reviewer_max_same_context_floor_seconds")
            floors = entry.get("floor_per_context_seconds") or {}
            block_entry = order_effect.get(block) or {}
            artefact_floors = block_entry.get("floor_by_context_seconds") or {}
            artefact_null = block_entry.get("aa_null_by_context") or {}
            checks.add(f"{block}: the artefact carries a same-context A/A null",
                       bool(artefact_null) and bool(artefact_floors),
                       "the null must compare the arm's first and last slot AT THE "
                       "SAME CONTEXT ('aa_null_by_context' + "
                       "'floor_by_context_seconds'); an arm-level null across the "
                       "block measures the growth of the latency with the context, "
                       "not the run-order drift")
            for ctx, floor in sorted(floors.items()):
                checks.close(f"{block}: the drift floor at context {ctx}",
                             floor, artefact_floors.get(ctx))
            for ctx, arms in sorted((entry.get("null_per_arm_and_context") or {}).items()):
                for arm, mine_row in sorted(arms.items()):
                    theirs_row = (artefact_null.get(ctx) or {}).get(arm) or {}
                    checks.add(f"{block}: the artefact carries the A/A null for "
                               f"{arm} at context {ctx}",
                               bool(theirs_row), f"artefact row={theirs_row!r}")
                    checks.close(f"{block}: the A/A null for {arm} at context {ctx}",
                                 mine_row.get("absolute_difference_seconds"),
                                 abs(theirs_row.get("difference_seconds") or 0.0))
            if artefact_floor is not None and same_max is not None:
                checks.close(f"{block}: the block drift floor is the largest "
                             f"same-context floor", same_max, artefact_floor)
            checks.add(f"{block}: every arm-to-arm difference has a verdict",
                       all(v.get("floor_seconds") is not None for v in verdicts
                           if v["block"] == block),
                       "the drift floor is missing for at least one context")
        # every arm must be timed in every block
        for block, pooled_key in (("training_step", "training_step"),
                                  ("prefill", "prefill"),
                                  ("incremental_decode", "incremental_decode")):
            arms = {r.get("arm") for r in (systems.get(pooled_key) or [])}
            checks.add(f"{block}: the optimized SDPA arm is timed too",
                       ARM_C in arms, f"arms={sorted(a for a in arms if a)}")

    # --- report ------------------------------------------------------------
    for row in checks.rows:
        print(f"[{'ok ' if row['ok'] else 'FAIL'}] {row['check']}"
              + (f" — {row['detail']}" if row["detail"] else ""))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
        print(f"reviewer recomputation written: {out}")
    if checks.failed:
        print(f"REVIEW_RECOMPUTE_FAIL: {len(checks.failed)} of {len(checks.rows)} checks failed")
        return 1
    print(f"REVIEW_RECOMPUTE_OK: {len(checks.rows)} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
