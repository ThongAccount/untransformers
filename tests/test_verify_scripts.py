"""The gate checkers themselves are checked.

A gate that cannot fail is not a gate. These tests build minimal synthetic
artefacts, run the three verifiers as subprocesses, and assert that a
well-formed artefact passes while a broken one is rejected.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def _quality(arms=("HE", "attention-32h"), seeds=(1, 2)):
    results = {}
    for arm in arms:
        results[arm] = {
            "arm": arm,
            "seeds": list(seeds),
            "steps": 3,
            "runs": [{"arm": arm, "seed": s, "steps": 3,
                      "train_curve": [1.0, 1.0, 1.0],
                      "evals": [{"step": 3, "held_out_loss": 1.5}],
                      "final_held_out_loss": 1.5}
                     for s in seeds],
        }
    a, b = arms[0], arms[1]
    fa = {str(r["seed"]): r["final_held_out_loss"] for r in results[a]["runs"]}
    fb = {str(r["seed"]): r["final_held_out_loss"] for r in results[b]["runs"]}
    diffs = {s: fa[s] - fb[s] for s in fa}
    n = len(diffs)
    mean = sum(diffs.values()) / n
    var = sum((v - mean) ** 2 for v in diffs.values()) / n
    return {
        "schema": "untransformers-quality-run/1",
        "label": "pilot",
        "created_utc": "2026-10-23T00:00:00+00:00",
        "protocol": {"arms": list(arms), "seeds": list(seeds), "train_seed": 0,
                     "steps": 3, "eval_every": 3, "eval_batches": 1,
                     "tokens_per_step": 64, "tokens_processed_per_run": 192,
                     "same_batches_same_order": True, "paired_init_seeds": True},
        "config": {"d": 32, "l": 32},
        "corpus": {"corpus_id": "x" * 64, "tokenizer_sha256": "y" * 64,
                   "n_tokens": 100, "books": 100},
        "environment": {"python": "3.12", "platform": "linux", "torch": "2.0",
                        "device": "cpu"},
        "results": results,
        "paired_final": {"seeds": list(seeds),
                         "arm_a_final_held_out_loss": fa,
                         "arm_b_final_held_out_loss": fb,
                         "differences_arm_a_minus_arm_b": diffs,
                         "mean_difference": mean,
                         "std_difference": var ** 0.5,
                         "n": n},
    }


def _systems():
    arms = ("HE", "attention-32h")
    cfg = {"d": 32, "l": 32, "n_layers": 2, "ffn_width": 64, "vocab": 5000,
           "batch_size": 8, "n_heads": 4, "trials": 3, "warmup": 1,
           "gen_tokens": 4}
    step = lambda arm: {"arm": arm, "step_seconds": {"median": 0.1},
                        "tokens_per_step": cfg["batch_size"] * cfg["l"],
                        "tokens_per_second": 100.0,
                        "memory": {"peak_rss_bytes": 1}, "mode": "train",
                        "n_params": 1000}
    pre = lambda arm: [{"arm": arm, "context_tokens": c,
                        "batch": 1, "latency": {"median": 0.01},
                        "tokens_per_second": 10.0, "memory": {"peak_rss_bytes": 1}}
                       for c in (1, 8)]
    inc = lambda arm: [{"arm": arm, "context_tokens": c, "generated_tokens": 4,
                        "max_positions": 32,
                        "per_token_seconds": {"median": 0.001, "samples": 8},
                        "tokens_per_second": 1000.0,
                        "cache_bytes_at_prompt": 100,
                        "cache_bytes_per_layer": 50}
                       for c in (1, 8)]
    sub = []
    for arm in arms:
        for c in (128, 256, 512, 1024):
            state = 100 if arm == "HE" else 2 * c * 32 * 4
            sub.append({"arm": arm, "scope": "sublayer-1", "context_tokens": c,
                        "per_token_seconds": {"median": 0.001, "samples": 3},
                        "state_bytes_per_layer": state,
                        "state_bytes_all_layers": state * 2,
                        "analytic_mults_per_token": 1})
    return {
        "schema": "untransformers-systems-run/1",
        "created_utc": "2026-10-23T00:00:00+00:00",
        "environment": {"python": "3.12", "platform": "linux", "torch": "2.0",
                        "torch_threads": 1, "device": "cpu", "cpu_count": 1},
        "config": cfg,
        "training_step": [step(a) for a in arms],
        "prefill": [r for a in arms for r in pre(a)],
        "incremental_decode": [r for a in arms for r in inc(a)],
        "sublayer_scaling": sub,
        "weight_traffic": {a: {"n_params": 1000, "parameter_bytes_fp32": 4000,
                               "training_step_parameter_bytes_read": 12000,
                               "decode_step_parameter_bytes_read": 4000,
                               "note": "analytic"} for a in arms},
        "paired_quality": {a: {"1": {"steps": 3, "final_held_out_loss": 1.5,
                                     "checkpoint": f"/x/{a}_seed1.pt"}} for a in arms},
    }


def _run(script, *args):
    return subprocess.run([PY, str(REPO / "tools" / "verify" / script), *args],
                          text=True, capture_output=True)


def test_quality_checker_passes_and_fails(tmp_path):
    good = tmp_path / "q.json"
    good.write_text(json.dumps(_quality()), encoding="utf-8")
    r = _run("check_quality_results.py", "--file", str(good), "--label", "pilot",
             "--arms", "HE", "attention-32h", "--seeds", "1", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PILOT_OK" in r.stdout

    bad = tmp_path / "q_bad.json"
    data = _quality()
    data["results"]["HE"]["runs"].pop()          # a seed went missing
    bad.write_text(json.dumps(data), encoding="utf-8")
    r = _run("check_quality_results.py", "--file", str(bad), "--label", "pilot",
             "--arms", "HE", "attention-32h", "--seeds", "1", "2")
    assert r.returncode == 1
    assert "FAIL" in r.stdout


def test_systems_checker_passes_and_fails(tmp_path):
    good = tmp_path / "s.json"
    good.write_text(json.dumps(_systems()), encoding="utf-8")
    r = _run("check_systems_results.py", "--file", str(good))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SYSTEMS_OK" in r.stdout
    r = _run("check_systems_results.py", "--file", str(good), "--require-paired-quality")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PAIRED_QUALITY_OK" in r.stdout

    bad = tmp_path / "s_bad.json"
    data = _systems()
    data["training_step"] = [r for r in data["training_step"] if r["arm"] != "HE"]
    bad.write_text(json.dumps(data), encoding="utf-8")
    r = _run("check_systems_results.py", "--file", str(bad))
    assert r.returncode == 1


def test_docs_checker_requires_content(tmp_path):
    systems = tmp_path / "SYSTEMS.md"
    scope = tmp_path / "SCOPE.md"
    systems.write_text("# Systems\n\n## Conditions\ndevice=cpu\n## Model shape\n"
                       "d=32 l=32 layers=2\n## Training step\n"
                       "median 0.1 s\n## Prefill\n## Incremental decoding\n"
                       "## Sublayer scaling\n## Weight traffic\n## Paired quality\n"
                       "## Context lengths\n1 8 32\n" + "x" * 500, encoding="utf-8")
    scope.write_text("# Scope\n\n## Options\n### execution-only\n### broaden\n### stop\n"
                     "## Decision\nexecution-only\n## Rationale\nbecause the measured "
                     "evidence in results/ supports it\n## Evidence\nmeasured\n"
                     "## What would change it\nfalsifier: a GPU rerun\n" + "y" * 500,
                     encoding="utf-8")
    r = _run("check_docs.py", "--systems", str(systems), "--scope", str(scope))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "DOCS_OK" in r.stdout

    scope.write_text("# Scope\n\nTODO write this\n", encoding="utf-8")
    r = _run("check_docs.py", "--systems", str(systems), "--scope", str(scope))
    assert r.returncode == 1
