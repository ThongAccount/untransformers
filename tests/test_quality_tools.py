"""The quality-side tools are checked: the merge and the statistics.

Two things are easy to get wrong and hard to see:

* the **merge** — six independent `(arm, seed)` runs have to become one
  results file with the arms, the seeds, the paired block and the pairing
  evidence, and it has to REFUSE parts that disagree about the protocol;
* the **statistics** — the pre-registered numbers (protocol §3.2) are computed
  from the per-seed values, so the arithmetic is pinned here: the paired
  differences, the sample SD with ddof=1, the 95% CI, the MDD, the perplexity
  ratio, and the "not resolved at n = 3" verdict when the CI covers 0.

The tests build synthetic artefacts; the real ones are checked by the gates.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

# The pairing evidence is derived from the REAL corpus (the stream fingerprints
# and the shared-parameter comparison run against the pinned manifest), so a
# synthetic artefact has to carry the real corpus identity or the gate will
# refuse it — which is exactly what it is supposed to do.
_MANIFEST = json.loads((REPO / "data" / "manifest.json").read_text(encoding="utf-8"))
CORPUS_ID = _MANIFEST["corpus_id"]
TOKENIZER_SHA256 = _MANIFEST["tokenizer"]["sha256"]


def _part(arm: str, seed: int, loss_a: float, loss_b: float, steps: int = 3) -> dict:
    """One per-run artefact, in the shape `experiments/run_quality.py` writes."""
    return {
        "schema": "untransformers-quality-run/1",
        "label": "full",
        "created_utc": "2026-10-24T00:00:00+00:00",
        "commit": "c" * 40,
        "protocol": {
            "arms": [arm], "seeds": [seed], "train_seed": 0, "steps": steps,
            "eval_every": 2, "eval_batches": 1, "tokens_per_step": 64,
            "tokens_processed_per_run": 64 * steps,
            "epochs_of_our_corpus": 0.001,
            "same_batches_same_order": True, "paired_init_seeds": True,
        },
        "config": {"d": 32, "l": 32, "n_layers": 2, "ffn_width": 64,
                   "vocab": 5000, "batch_size": 8, "lr": 0.001,
                   "betas": [0.9, 0.999], "weight_decay": 0.001, "dropout": 0.1,
                   "eval_every": 2, "eval_batches": 1, "clip_grad": None,
                   "weight_decay_no_norm": True},
        "corpus": {"corpus_id": CORPUS_ID, "tokenizer_sha256": TOKENIZER_SHA256,
                   "manifest_sha256": None, "n_tokens": 9514186, "books": 100},
        "environment": {"python": "3.12", "platform": "linux", "torch": "2.0",
                        "torch_threads": 1, "device": "cpu"},
        "results": {
            arm: {
                "arm": arm, "seeds": [seed], "train_seed": 0, "steps": steps,
                "runs": [{"arm": arm, "seed": seed, "steps": steps,
                          "train_curve": [2.0, 1.5, loss_a],
                          "evals": [{"step": 2, "held_out_loss": loss_b},
                                    {"step": 3, "held_out_loss": loss_a}],
                          "final_held_out_loss": loss_a}],
                "dispersion": {},
            }
        },
    }


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_merge_produces_one_gate_ready_results_file(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _write(runs / "HE_seed1.json", _part("HE", 1, 1.50, 1.60))
    _write(runs / "HE_seed2.json", _part("HE", 2, 1.40, 1.55))
    _write(runs / "attention-32h_seed1.json", _part("attention-32h", 1, 1.70, 1.80))
    _write(runs / "attention-32h_seed2.json", _part("attention-32h", 2, 1.60, 1.75))
    out = tmp_path / "merged.json"

    r = subprocess.run([PY, str(REPO / "experiments/merge_quality_runs.py"),
                        "--runs", str(runs), "--out", str(out), "--label", "full",
                        "--commit", "d" * 40],
                       text=True, capture_output=True)
    assert r.returncode == 0, r.stdout + r.stderr
    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["protocol"]["arms"] == ["HE", "attention-32h"]
    assert data["protocol"]["seeds"] == [1, 2]
    assert [x["seed"] for x in data["results"]["HE"]["runs"]] == [1, 2]
    assert [x["seed"] for x in data["results"]["attention-32h"]["runs"]] == [1, 2]
    assert data["paired_final"]["n"] == 2
    # the pairing claims are derived, not asserted
    assert data["pairing_evidence"]["schema"] == "untransformers-pairing-evidence/1"
    assert data["protocol"]["same_batches_same_order"] is True
    assert data["protocol"]["paired_init_seeds"] is True

    # ... and the gate accepts it
    g = subprocess.run([PY, str(REPO / "tools/verify/check_quality_results.py"),
                        "--file", str(out), "--label", "full",
                        "--arms", "HE", "attention-32h", "--seeds", "1", "2"],
                       text=True, capture_output=True)
    assert g.returncode == 0, g.stdout + g.stderr
    assert "FULL_OK" in g.stdout


def test_merge_refuses_parts_that_disagree_about_the_protocol(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _write(runs / "HE_seed1.json", _part("HE", 1, 1.5, 1.6))
    other = _part("attention-32h", 1, 1.7, 1.8)
    other["config"]["d"] = 64          # a different shape
    _write(runs / "attention-32h_seed1.json", other)
    out = tmp_path / "merged.json"

    r = subprocess.run([PY, str(REPO / "experiments/merge_quality_runs.py"),
                        "--runs", str(runs), "--out", str(out), "--label", "full"],
                       text=True, capture_output=True)
    assert r.returncode != 0
    assert "disagrees" in (r.stdout + r.stderr)
    assert not out.exists()


def test_statistics_are_the_pre_registered_ones(tmp_path):
    """d̄, the ddof=1 SD, the CI, the MDD and the perplexity ratio."""
    sys.path.insert(0, str(REPO / "tools"))
    import summarise_quality as sq  # noqa: E402

    # two seeds, differences d1 = 0.10, d2 = 0.20 -> d̄ = 0.15, s_d = 0.0707107
    block = sq.paired_block({1: 1.50, 2: 1.40}, {1: 1.40, 2: 1.20})
    assert math.isclose(block["mean_difference"], 0.15, abs_tol=1e-12)
    assert math.isclose(block["std_difference_ddof1"], math.sqrt(0.005),
                        rel_tol=1e-9)
    assert math.isclose(block["standard_error"], math.sqrt(0.005) / math.sqrt(2),
                        rel_tol=1e-9)
    # CI = d̄ ± t(0.975, df=1) * SE, with t = 12.706
    t = 12.706
    half = t * math.sqrt(0.005) / math.sqrt(2)
    assert math.isclose(block["ci95"][0], 0.15 - half, rel_tol=1e-9)
    assert math.isclose(block["ci95"][1], 0.15 + half, rel_tol=1e-9)
    # with df = 1 the t quantile is 12.706, so the half-width is 0.635 > d̄:
    # this interval COVERS zero and the comparison is NOT resolved
    assert block["ci95_excludes_zero"] is False
    assert block["resolved"] is False
    assert block["verdict"] == "not resolved at n = 2"
    # MDD = (t(0.975,1) + t(0.80,1)) * SE = (12.706 + 1.282) * SE
    assert math.isclose(block["mdd_80_power"], (12.706 + 1.282) * math.sqrt(0.005) / math.sqrt(2),
                        rel_tol=1e-9)
    # perplexity: exp(loss) per arm, ratio exp(d̄)
    assert math.isclose(block["perplexity_ratio"], math.exp(0.15), rel_tol=1e-12)
    assert math.isclose(block["perplexity_ratio_ci95"][0], math.exp(0.15 - half),
                        rel_tol=1e-9)


def test_a_large_paired_effect_is_resolved():
    """The same arithmetic with an effect that clears the CI."""
    sys.path.insert(0, str(REPO / "tools"))
    import summarise_quality as sq  # noqa: E402

    block = sq.paired_block({1: 2.50, 2: 2.40}, {1: 1.50, 2: 1.50})
    assert math.isclose(block["mean_difference"], 0.95, abs_tol=1e-12)
    assert block["ci95_excludes_zero"] is True
    assert block["resolved"] is True
    assert block["ci95"][0] > 0
    # the MDD is the smallest effect this design could have detected at 80% power
    assert block["mdd_80_power"] < abs(block["mean_difference"])


def test_a_ci_that_covers_zero_is_not_resolved(tmp_path):
    sys.path.insert(0, str(REPO / "tools"))
    import summarise_quality as sq  # noqa: E402

    # identical losses -> d̄ = 0, the CI covers 0 -> "not resolved at n = 3"
    block = sq.paired_block({1: 1.5, 2: 1.4, 3: 1.45},
                             {1: 1.5, 2: 1.4, 3: 1.45})
    assert block["ci95_excludes_zero"] is False
    assert block["resolved"] is False
    assert block["verdict"] == "not resolved at n = 3"
    assert block["mdd_80_power"] == 0.0


def test_bootstrap_se_is_a_se_and_the_resample_is_seeded(tmp_path):
    sys.path.insert(0, str(REPO / "tools"))
    import summarise_quality as sq  # noqa: E402

    a = [0.1, 0.2, 0.3, 0.4]
    b = [0.2, 0.3, 0.4, 0.5]
    first = sq.bootstrap_se(a, b, n_resamples=200)
    second = sq.bootstrap_se(a, b, n_resamples=200)
    assert first == second, "the bootstrap must be reproducible"
    assert first["available"] is True
    assert first["bootstrap_se_paired_difference"] >= 0.0
    assert first["n_windows"] == 4
    assert sq.bootstrap_se([], b)["available"] is False


def test_the_chunked_book_sweep_scores_the_same_windows_as_the_plain_one():
    """`region_window_losses` builds the windows in bounded chunks; it has to
    produce the same per-window losses as the straightforward path."""
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO))
    import numpy as np  # noqa: E402
    import torch  # noqa: E402

    from eval_checkpoints import per_window_losses, region_window_losses  # noqa: E402
    from hextractor.init import apply_init_policy  # noqa: E402
    from hextractor.model import ExtractorTransformer  # noqa: E402

    torch.manual_seed(0)
    model = ExtractorTransformer(arm="HE", d=32, l=8, n_layers=1, ffn_width=32,
                                 vocab=50, dropout=0.0)
    apply_init_policy(model, seed=0)
    model.eval()

    l = 8
    region = (np.arange(40) % 50).astype(np.int64)
    starts = np.arange(0, region.size - l, dtype=np.int64)
    wins = np.stack([region[s:s + l + 1] for s in starts])

    chunked = region_window_losses(model, region, l, starts, torch.device("cpu"), batch=7)
    plain = per_window_losses(model, wins[:, :-1], wins[:, 1:], torch.device("cpu"),
                               batch=64)
    assert chunked.shape == plain.shape == starts.shape
    assert np.allclose(chunked, plain, atol=1e-6), "the chunked sweep must agree"

    # a coarser stride scores a subset of the same windows, unchanged
    coarse_starts = np.arange(0, region.size - l, 3, dtype=np.int64)
    coarse = region_window_losses(model, region, l, coarse_starts, torch.device("cpu"),
                                   batch=5)
    assert np.allclose(coarse, plain[coarse_starts], atol=1e-6)


def test_the_doc_filler_replaces_the_markers_with_the_rendered_tables(tmp_path):
    """The write-ups must carry rendered numbers, not retyped ones: the filler
    replaces every `<!-- TABLES:... -->` marker with the section it renders."""
    doc = tmp_path / "DOC.md"
    doc.write_text("# T\n\n<!-- TABLES:quality-summary -->\n\n"
                   "<!-- TABLES:quality-bootstrap -->\n", encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "conditions": {"arms": ["HE", "attention-32h"], "steps": 2000},
        "endpoints": {"final": {
            "definition": "primary: held-out loss at the final step",
            "sign_convention": "d_s = L_HE(s) - L_attn(s)",
            "seeds": [1], "per_seed_arm_a": {"1": 1.0},
            "per_seed_arm_b": {"1": 1.5}, "differences": {"1": -0.5},
            "arm_a": {"mean": 1.0, "std_ddof1": None},
            "arm_b": {"mean": 1.5, "std_ddof1": None},
            "mean_difference": -0.5, "standard_error": 0.0,
            "ci95": [-0.5, -0.5], "ci95_excludes_zero": True,
            "mdd_80_power": 0.0, "perplexity_ratio": 0.6065306597126334,
            "perplexity_ratio_ci95": [0.6065306597126334, 0.6065306597126334],
            "verdict": "resolved"}},
        "eval_set_bootstrap": {"per_seed": {"1": {
            "bootstrap_se_arm_a": 0.01, "bootstrap_se_arm_b": 0.02,
            "bootstrap_se_paired_difference": 0.005}},
            "note": "eval-set noise only"},
    }, indent=2), encoding="utf-8")

    r = subprocess.run([PY, str(REPO / "tools/fill_doc_tables.py"),
                        "--doc", str(doc), "--summary", str(summary), "--write"],
                       text=True, capture_output=True)
    assert r.returncode == 0, r.stdout + r.stderr
    text = doc.read_text(encoding="utf-8")
    assert "TABLES:" not in text, "every marker must be replaced"
    assert "Endpoint: primary: held-out loss at the final step" in text
    assert "2000" in text
    assert "-0.500000" in text
    assert "Eval-set bootstrap" in text


def test_the_doc_filler_refuses_a_marker_it_cannot_render(tmp_path):
    doc = tmp_path / "DOC.md"
    doc.write_text("# T\n\n<!-- TABLES:quality-nonsense -->\n", encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"conditions": {}, "endpoints": {}}), encoding="utf-8")
    r = subprocess.run([PY, str(REPO / "tools/fill_doc_tables.py"),
                        "--doc", str(doc), "--summary", str(summary), "--write"],
                       text=True, capture_output=True)
    assert r.returncode != 0
    assert "quality-nonsense" in (r.stdout + r.stderr)
    # the document is left untouched
    assert "TABLES:quality-nonsense" in doc.read_text(encoding="utf-8")


def test_a_single_seed_has_no_interval_and_no_nan():
    """One seed has no spread: the interval and the MDD are undefined, and they
    must be written as null (NaN is not valid JSON)."""
    sys.path.insert(0, str(REPO / "tools"))
    import summarise_quality as sq  # noqa: E402

    block = sq.paired_block({1: 1.5}, {1: 1.4})
    assert block["mean_difference"] == pytest.approx(0.1)
    assert block["std_difference_ddof1"] is None
    assert block["ci95"] == [None, None]
    assert block["mdd_80_power"] is None
    assert block["resolved"] is False
    assert block["perplexity_ratio_ci95"] == [None, None]
    text = json.dumps(block)
    assert "NaN" not in text, "the summary must stay valid JSON"
