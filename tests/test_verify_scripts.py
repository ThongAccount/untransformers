"""The gate checkers themselves are checked.

A gate that cannot fail is not a gate. These tests build minimal synthetic
artefacts, run the verifiers as subprocesses, and assert that a
well-formed artefact passes while a broken one is rejected.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def _pairing_evidence(arms=("HE", "attention-32h"), seeds=(1, 2)):
    """The evidence block that certifies the pairing claims (F1).

    It is what `tools/verify/verify_pairing_evidence.py` writes; the synthetic
    version here has the same shape so the checker can be tested against it.
    The shape is pinned by `hextractor/pairing.py::build_pairing_evidence`:
    the protocol (arms/seeds) lives inside `protocol`, and every per-seed
    entry carries the seed-level aggregates plus the per-arm detail.
    """
    per_seed = {}
    for s in seeds:
        per_arm = {}
        for arm in arms[1:]:
            per_arm[arm] = {
                "compared_against": arms[0],
                "n_shared_parameters": 42,
                "n_only_in_first_arm": [],
                "n_only_in_this_arm": [],
                "n_generator_drawn": 12,
                "n_generator_drawn_bitwise_equal": 12,
                "differing_parameters": [],
                "bitwise_equal": True,
                "shared_parameter_names_sha256": "e" * 64,
            }
        per_seed[str(s)] = {
            "bitwise_equal": True,
            "n_generator_drawn": 12,
            "n_generator_drawn_bitwise_equal": 12,
            "n_shared_parameters": 42,
            "per_arm": per_arm,
        }
    return {
        "schema": "untransformers-pairing-evidence/1",
        "created_utc": "2026-10-23T00:00:00+00:00",
        "verifier": {"tool": "tools/verify/verify_pairing_evidence.py",
                     "version": "1", "git_commit": "b" * 40, "git_dirty": False},
        "corpus": {"corpus_id": "x" * 64, "tokenizer_sha256": "y" * 64},
        "protocol": {"arms": list(arms), "seeds": list(seeds),
                     "train_seed": 0, "l": 32, "batch_size": 8, "steps": 3},
        "claims": {
            "same_batches_same_order": {
                "verified": True,
                "method": ("the train/held-out streams are pure functions of "
                           "(train_seed, split, corpus_id, tokenizer_sha256, l, batch) "
                           "and take no arm or seed argument; the first N windows are "
                           "hashed and compared"),
                "n_batches_hashed": 100,
                "train_stream_sha256": "c" * 64,
                "held_out_stream_sha256": "d" * 64,
            },
            "paired_init_seeds": {
                "verified": True,
                "method": ("both arms are rebuilt with the same seed and the shared "
                           "(non-sublayer-1) parameters are compared bitwise"),
                "per_seed": per_seed,
            },
        },
        "checkpoints": {},
    }


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
        "pairing_evidence": _pairing_evidence(arms, seeds),
    }


SYSTEMS_ARMS = ("HE", "attention-32h", "attention-32h-sdpa")


def _stats(median):
    """A statistics block in the shape the runner publishes (S3)."""
    return {"median": median, "mean": median, "p10": median * 0.9,
            "p90": median * 1.1, "min": median * 0.8, "max": median * 1.2, "n": 10}


def _quality_block(loaded=False):
    if loaded:
        return {"loaded": True, "arm": "HE", "seed": 1, "checkpoint": "/x/HE_seed1.pt",
                "steps": 2000, "final_held_out_loss": 6.5,
                "quality_level": "trained quality checkpoint (held-out loss above)"}
    return {"loaded": False, "arm": "HE", "seed": None, "checkpoint": None,
            "steps": None, "final_held_out_loss": None,
            "quality_level": "architecture at initialisation — no quality checkpoint "
                             "was loaded, so this row is NOT tied to any quality level "
                             "(S8)."}


def _systems():
    arms = SYSTEMS_ARMS
    cfg = {"d": 32, "l": 32, "n_layers": 2, "ffn_width": 64, "vocab": 5000,
           "batch_size": 8, "n_heads": 4, "trials": 10, "warmup": 1,
           "gen_tokens": 4, "decode_trials": 5,
           "order_schedule": list(arms) + list(reversed(arms))}
    quality = _quality_block()
    step = lambda arm: {"arm": arm, "scope": "model-level",
                         "step_seconds": _stats(0.1),
                         "tokens_per_step": cfg["batch_size"] * cfg["l"],
                         "tokens_per_second": 100.0,
                         "memory": {"scope": "process-level: ru_maxrss is the "
                                             "high-water mark of the whole process (S6).",
                                    "peak_rss_bytes": 1},
                         "mode": "train (dropout on)", "n_params": 1000,
                         "slots": 2, "order_indices": [0, 5],
                         "quality": quality,
                         "units": {"latency": "seconds per training step",
                                   "throughput": "tokens per second",
                                   "memory": "bytes"},
                         "denominator": "batch_size(8) * l(32) = 256 tokens/step"}
    pre = lambda arm: [{"arm": arm, "scope": "model-level", "context_tokens": c,
                         "batch": 1, "latency": _stats(0.01),
                         "tokens_per_second": 10.0,
                         "memory": {"scope": "process-level (S6)", "peak_rss_bytes": 1},
                         "quality": quality,
                         "units": {"latency": "seconds for one full forward",
                                   "throughput": "tokens per second", "memory": "bytes"},
                         "denominator": f"batch 1 * context_tokens({c}) = {c} tokens/forward"}
                        for c in (1, 8)]
    inc = lambda arm: [{"arm": arm, "scope": "model-level", "context_tokens": c,
                         "generated_tokens": 4, "max_positions": 32,
                         "per_token_seconds": _stats(0.001),
                         "first_token_seconds": _stats(0.002),
                         "steady_state_per_token_seconds": _stats(0.001),
                         "tokens_per_second": 1000.0,
                         "cache_bytes_at_prompt": 100,
                         "cache_bytes_per_layer": 50,
                         "slots": 2,
                         "memory": {"scope": "process-level (S6)", "peak_rss_bytes": 1},
                         "quality": quality,
                         "units": {"per_token_latency": "seconds per generated token",
                                   "first_token_latency": "seconds for the first token",
                                   "throughput": "tokens per second",
                                   "cache_bytes": "bytes held by the cache"},
                         "denominator": "decode trials=5 x gen_tokens=4 tokens per context"}
                       for c in (1, 8)]
    sub = []
    for arm in arms:
        for c in (128, 256, 512, 1024):
            state = 100 if arm == "HE" else 2 * c * 32 * 4
            sub.append({"arm": arm, "scope": "sublayer-1", "context_tokens": c,
                        "per_token_seconds": _stats(0.001),
                        "state_bytes_per_layer": state,
                        "state_bytes_all_layers": state * 2,
                        "state_bytes_label": "analytic (formula, not measured)",
                        "state_bytes_formula": f"2 * {c} * 32 * 4 bytes",
                        "analytic_mults_per_token": 1,
                        "units": {"per_token_latency": "seconds for one token",
                                  "state_bytes": "bytes per layer (analytic)"},
                        "denominator": f"one token at context T={c}, sublayer-1 only"})
    order_effect = {
        block: {"block": block,
                "aa_null": {a: {"difference_seconds": 0.0,
                                 "first_slot_median_seconds": 0.1,
                                 "last_slot_median_seconds": 0.1,
                                 "relative_difference": 0.0} for a in arms},
                "drift_floor_seconds": 0.0}
        for block in ("training_step", "prefill", "incremental_decode")
    }
    weight_traffic = {
        a: {"label": "analytic (formula over the parameter count, not measured)",
            "n_params": 1000, "parameter_bytes_fp32": 4000,
            "parameter_bytes_formula": "n_params * 4 bytes (fp32)",
            "training_step_parameter_bytes_read": 12000,
            "training_step_parameter_bytes_formula": "3 * n_params * 4 bytes",
            "decode_step_parameter_bytes_read": 4000,
            "decode_step_parameter_bytes_formula": "1 * n_params * 4 bytes",
            "note": "analytic"}
        for a in arms
    }
    state_traffic = {
        a: {"label": "analytic (formula, not measured)",
            "state_bytes_per_layer_per_token": None,
            "state_bytes_per_layer_formula":
                "(l - 1) * d * 4 bytes" if a == "HE" else "2 * T * d * 4 bytes",
            "grows_with_context": a != "HE",
            "per_context": [{"context_tokens": 128, "state_bytes_per_layer": 128,
                             "formula": "..."}]}
        for a in arms
    }
    return {
        "schema": "untransformers-systems-run/2",
        "created_utc": "2026-10-23T00:00:00+00:00",
        "environment": {"python": "3.12", "platform": "linux", "torch": "2.0",
                        "torch_threads": 1, "device": "cpu", "cpu_count": 1,
                        "numerics": {"dtype": "float32", "allow_tf32_matmul": False,
                                     "allow_tf32_cudnn": False,
                                     "float32_matmul_precision": "highest",
                                     "cudnn_benchmark": False,
                                     "cudnn_deterministic": True},
                        "git_commit": "a" * 40, "git_dirty": False,
                        "git_branch": "research/milestone-1-extractor",
                        "arms": list(arms)},
        "config": cfg,
        "training_step": [step(a) for a in arms],
        "prefill": [r for a in arms for r in pre(a)],
        "incremental_decode": [r for a in arms for r in inc(a)],
        "sublayer_scaling": sub,
        "weight_traffic": weight_traffic,
        "state_traffic": state_traffic,
        "order_effect": order_effect,
        "paired_quality": {a: {"1": {"steps": 2000, "final_held_out_loss": 6.5,
                                     "checkpoint": f"/x/{a}_seed1.pt"}}
                           for a in ("HE", "attention-32h")},
        "caveats": ["kernel caveat"],
    }


def _run(script, *args):
    return subprocess.run([PY, str(REPO / "tools" / "verify" / script), *args],
                          text=True, capture_output=True)


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_quality_checker_passes_and_fails(tmp_path):
    good = _write(tmp_path, "q.json", _quality())
    r = _run("check_quality_results.py", "--file", str(good), "--label", "pilot",
             "--arms", "HE", "attention-32h", "--seeds", "1", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PILOT_OK" in r.stdout

    bad = _write(tmp_path, "q_bad.json", _quality())
    data = json.loads(bad.read_text(encoding="utf-8"))
    data["results"]["HE"]["runs"].pop()          # a seed went missing
    bad.write_text(json.dumps(data), encoding="utf-8")
    r = _run("check_quality_results.py", "--file", str(bad), "--label", "pilot",
             "--arms", "HE", "attention-32h", "--seeds", "1", "2")
    assert r.returncode == 1
    assert "FAIL" in r.stdout


def test_quality_checker_requires_pairing_evidence(tmp_path):
    """F1: the gate must not certify a pairing claim it has not verified."""
    data = _quality()
    data.pop("pairing_evidence")
    p = _write(tmp_path, "q_no_evidence.json", data)
    r = _run("check_quality_results.py", "--file", str(p), "--label", "pilot",
             "--arms", "HE", "attention-32h", "--seeds", "1", "2")
    assert r.returncode == 1
    assert "pairing" in r.stdout.lower()

    data = _quality()
    data["pairing_evidence"]["claims"]["paired_init_seeds"]["verified"] = False
    p = _write(tmp_path, "q_unverified.json", data)
    r = _run("check_quality_results.py", "--file", str(p), "--label", "pilot",
             "--arms", "HE", "attention-32h", "--seeds", "1", "2")
    assert r.returncode == 1

    # the evidence may also be supplied as a separate artefact
    data = _quality()
    ev = data.pop("pairing_evidence")
    p = _write(tmp_path, "q_external.json", data)
    e = _write(tmp_path, "evidence.json", ev)
    r = _run("check_quality_results.py", "--file", str(p), "--label", "pilot",
             "--arms", "HE", "attention-32h", "--seeds", "1", "2",
             "--pairing-evidence", str(e))
    assert r.returncode == 0, r.stdout + r.stderr


def test_quality_checker_rejects_contradicted_pairing_literal(tmp_path):
    """A protocol literal that contradicts the evidence is a failure, not a pass."""
    data = _quality()
    data["protocol"]["paired_init_seeds"] = False
    p = _write(tmp_path, "q_contradiction.json", data)
    r = _run("check_quality_results.py", "--file", str(p), "--label", "pilot",
             "--arms", "HE", "attention-32h", "--seeds", "1", "2")
    assert r.returncode == 1


def test_systems_checker_passes_and_fails(tmp_path):
    good = _write(tmp_path, "s.json", _systems())
    r = _run("check_systems_results.py", "--file", str(good))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SYSTEMS_OK" in r.stdout
    r = _run("check_systems_results.py", "--file", str(good), "--require-paired-quality")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PAIRED_QUALITY_OK" in r.stdout

    bad = _write(tmp_path, "s_bad.json", _systems())
    data = json.loads(bad.read_text(encoding="utf-8"))
    data["training_step"] = [r for r in data["training_step"] if r["arm"] != "HE"]
    bad.write_text(json.dumps(data), encoding="utf-8")
    r = _run("check_systems_results.py", "--file", str(bad))
    assert r.returncode == 1


def test_systems_checker_requires_the_sdpa_baseline(tmp_path):
    """S1: the optimized baseline must be timed in every block."""
    data = _systems()
    data["training_step"] = [r for r in data["training_step"]
                             if r["arm"] != "attention-32h-sdpa"]
    p = _write(tmp_path, "s_no_sdpa.json", data)
    r = _run("check_systems_results.py", "--file", str(p))
    assert r.returncode == 1
    assert "attention-32h-sdpa" in r.stdout


def test_systems_checker_requires_the_aa_null(tmp_path):
    """S2: every arm needs an A/A null and a drift floor."""
    data = _systems()
    data["order_effect"]["training_step"]["aa_null"].pop("HE")
    p = _write(tmp_path, "s_no_aa.json", data)
    r = _run("check_systems_results.py", "--file", str(p))
    assert r.returncode == 1

    data = _systems()
    data["order_effect"]["prefill"].pop("drift_floor_seconds")
    p = _write(tmp_path, "s_no_floor.json", data)
    r = _run("check_systems_results.py", "--file", str(p))
    assert r.returncode == 1


def test_systems_checker_requires_enough_trials(tmp_path):
    """S3: >= 10 timed trials and >= 5 decode trials."""
    data = _systems()
    data["config"]["trials"] = 5
    p = _write(tmp_path, "s_few_trials.json", data)
    r = _run("check_systems_results.py", "--file", str(p))
    assert r.returncode == 1

    data = _systems()
    data["config"]["decode_trials"] = 2
    p = _write(tmp_path, "s_few_decode.json", data)
    r = _run("check_systems_results.py", "--file", str(p))
    assert r.returncode == 1


def test_systems_checker_requires_the_numerics_manifest(tmp_path):
    """S5: a number without its precision settings is not reproducible."""
    data = _systems()
    data["environment"]["numerics"].pop("float32_matmul_precision")
    p = _write(tmp_path, "s_no_numerics.json", data)
    r = _run("check_systems_results.py", "--file", str(p))
    assert r.returncode == 1

    data = _systems()
    data["environment"].pop("git_commit")
    p = _write(tmp_path, "s_no_commit.json", data)
    r = _run("check_systems_results.py", "--file", str(p))
    assert r.returncode == 1


def test_systems_checker_requires_the_palindromic_order(tmp_path):
    """S2: the measurement order must be ABBA so every arm has an A/A pair."""
    data = _systems()
    data["config"]["order_schedule"] = list(SYSTEMS_ARMS)
    p = _write(tmp_path, "s_sequential.json", data)
    r = _run("check_systems_results.py", "--file", str(p))
    assert r.returncode == 1


def test_systems_checker_requires_units_quality_and_analytic_labels(tmp_path):
    """S7/S8/S10: units, a stated quality level and analytic labels."""
    data = _systems()
    data["training_step"][0].pop("units")
    p = _write(tmp_path, "s_no_units.json", data)
    r = _run("check_systems_results.py", "--file", str(p))
    assert r.returncode == 1

    data = _systems()
    data["prefill"][0].pop("quality")
    p = _write(tmp_path, "s_no_quality.json", data)
    r = _run("check_systems_results.py", "--file", str(p))
    assert r.returncode == 1

    data = _systems()
    data["weight_traffic"]["HE"]["label"] = "measured"
    p = _write(tmp_path, "s_unlabelled_traffic.json", data)
    r = _run("check_systems_results.py", "--file", str(p))
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


def test_doc_number_checker_passes_on_the_artefacts_numbers_and_fails_on_a_stale_one(tmp_path):
    """The prose of a write-up is typed by hand; a hand-typed number is where a
    stale figure hides. The checker must pass when the prose agrees with the
    artefact and fail when one constant is wrong."""
    results = {
        "protocol": {
            "tokens_processed_per_run": 16_384_000,
            "epochs_of_our_corpus": 1.8127392432109876,
            "corpus_tokens_train_region": 9_038_476,
            "corpus_tokens_heldout_region": 475_710,
            "eval_batches": 8,
        },
        "config": {"batch_size": 64, "l": 128},
    }
    rpath = tmp_path / "results.json"
    rpath.write_text(json.dumps(results), encoding="utf-8")

    good = tmp_path / "GOOD.md"
    good.write_text(
        "# T\n\n"
        "This run processed 16,384,000 tokens per run — 1.813 nominal epochs, "
        "and 3.333% of the paper's 491,520,000-token budget.\n"
        "The held-out region is 475,710 tokens and is scored on 512 windows "
        "(65,536 target tokens).\n",
        encoding="utf-8")

    r = subprocess.run([PY, str(REPO / "tools/verify/check_doc_numbers.py"),
                        "--doc", str(good), "--results", str(rpath)],
                       text=True, capture_output=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "DOC_NUMBERS_OK" in r.stdout

    # a stale token budget in the prose must be caught
    stale = tmp_path / "STALE.md"
    stale.write_text(good.read_text(encoding="utf-8").replace("16,384,000", "8,192,000"),
                     encoding="utf-8")
    r = subprocess.run([PY, str(REPO / "tools/verify/check_doc_numbers.py"),
                        "--doc", str(stale), "--results", str(rpath)],
                       text=True, capture_output=True)
    assert r.returncode != 0
    assert "16,384,000" in (r.stdout + r.stderr)
    assert "tokens processed per run" in (r.stdout + r.stderr)


def test_doc_number_checker_recomputes_the_unseen_book_share_from_the_manifest(tmp_path):
    """The 85.5% figure is a property of the corpus and the split, not of a
    run: the checker recomputes it from the manifest and the cut."""
    manifest = {"tokens": {"per_book": [
        {"ebook_id": 1, "n_tokens": 100},
        {"ebook_id": 2, "n_tokens": 100},
        {"ebook_id": 3, "n_tokens": 100},
        {"ebook_id": 4, "n_tokens": 100},
    ]}}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")

    # the cut falls inside book 3 (at token 250 of 400): book 3's tail (50
    # tokens) is held out but the book was seen, and book 4 (100 tokens) was
    # never seen at all -> 100 of the 150 held-out tokens are from unseen books
    results = {
        "protocol": {"tokens_processed_per_run": 16_384_000,
                     "epochs_of_our_corpus": 1.8127,
                     "corpus_tokens_train_region": 250,
                     "corpus_tokens_heldout_region": 150,
                     "eval_batches": 8},
        "config": {"batch_size": 64, "l": 128},
    }
    rpath = tmp_path / "results.json"
    rpath.write_text(json.dumps(results), encoding="utf-8")

    doc = tmp_path / "DOC.md"
    doc.write_text("16,384,000 tokens; 1.813 nominal epochs; 3.333% of the paper; "
                   "150 held-out tokens; 512 windows; 65,536 target tokens; "
                   "66.7% of the held-out tokens come from books never seen in "
                   "training.\n", encoding="utf-8")
    r = subprocess.run([PY, str(REPO / "tools/verify/check_doc_numbers.py"),
                        "--doc", str(doc), "--results", str(rpath),
                        "--manifest", str(mpath)], text=True, capture_output=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "66.7" in r.stdout
