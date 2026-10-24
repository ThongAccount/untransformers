"""The reviewer's own re-computation is checked too.

The Job-B verdict rests on `tools/review_recompute.py`, so the tool needs its
own tests: a reviewer tool that is quietly wrong would certify a wrong result.
What is pinned here:

* the Student-t quantiles it inverts — against the Cauchy closed form at df = 1
  and against the protocol's stated constants at df = 2;
* the paired statistics and the "not resolved at n = 3" verdict;
* the **same-context** A/A null — the floor must be the same-context spread,
  not the growth of the latency with the context — and the resolvability rule
  (a difference is resolvable only when it exceeds the floor of its own
  context);
* the renderer-key invariant: an artefact without the key the renderer reads
  must make the tool fail, not pass quietly.
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_reviewer():
    spec = importlib.util.spec_from_file_location(
        "review_recompute", REPO / "tools" / "review_recompute.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rr = _load_reviewer()


# ---------------------------------------------------------------------------
# the Student-t quantiles
# ---------------------------------------------------------------------------
def test_the_student_t_quantiles_match_the_closed_form_and_the_protocol():
    # df = 1 is the Cauchy distribution: t_p = tan(pi * (p - 0.5))
    assert rr.student_t_ppf(0.975, 1) == pytest.approx(math.tan(math.pi * 0.475), rel=1e-9)
    assert rr.student_t_ppf(0.80, 1) == pytest.approx(math.tan(math.pi * 0.30), rel=1e-9)
    # the protocol's stated constants at df = 2
    assert rr.student_t_ppf(0.975, 2) == pytest.approx(4.3027, abs=5e-5)
    assert rr.student_t_ppf(0.80, 2) == pytest.approx(1.0607, abs=5e-5)
    assert rr.student_t_ppf(0.975, 2) / math.sqrt(3) == pytest.approx(2.484, abs=5e-4)
    assert ((rr.student_t_ppf(0.975, 2) + rr.student_t_ppf(0.80, 2)) / math.sqrt(3)
            == pytest.approx(3.097, abs=5e-4))
    # the 80% quantiles the builder's old table got wrong for 29 of its 30 rows
    assert rr.student_t_ppf(0.80, 3) == pytest.approx(0.978472, abs=1e-5)
    assert rr.student_t_ppf(0.80, 5) == pytest.approx(0.919544, abs=1e-5)
    assert rr.student_t_ppf(0.80, 30) == pytest.approx(0.853767, abs=1e-5)


# ---------------------------------------------------------------------------
# the paired statistics
# ---------------------------------------------------------------------------
def test_the_paired_statistics_and_the_resolved_verdict():
    a = {1: 3.10, 2: 3.12, 3: 3.08}
    b = {1: 3.00, 2: 3.00, 3: 3.00}
    block = rr.paired_stats(a, b)
    assert block["n"] == 3
    assert block["mean_difference"] == pytest.approx(0.10)
    assert block["std_difference_ddof1"] == pytest.approx(0.02)
    assert block["t_critical_95"] == pytest.approx(4.3027, abs=5e-5)
    assert block["t_critical_80"] == pytest.approx(1.0607, abs=5e-5)
    assert block["ci95"][0] > 0.0
    assert block["resolved"] is True
    # the sign convention: positive means HE's loss is HIGHER (worse)
    assert block["perplexity_ratio"] == pytest.approx(math.exp(0.10))


def test_a_difference_that_is_not_resolved_is_reported_as_not_resolved():
    c = {1: 3.00, 2: 3.20, 3: 2.80}
    d = {1: 3.05, 2: 3.15, 3: 2.95}
    block = rr.paired_stats(c, d)
    assert block["resolved"] is False
    assert "not resolved at n = 3" in block["verdict"]
    assert block["ci95"][0] < 0.0 < block["ci95"][1]


# ---------------------------------------------------------------------------
# the systems null and the resolvability rule
# ---------------------------------------------------------------------------
def _slot(arm: str, ctx: int, order_index: int, median: float) -> dict:
    return {"arm": arm, "context_tokens": ctx, "order_index": order_index,
            "latency_stats": {"median": median, "times": [median]}}


def _slot_system():
    """Two contexts, two slots per arm: the same-context spread is ~0.01-0.02 s,
    the cross-context gap is ~0.30 s."""
    slots = [
        _slot("HE", 128, 0, 0.100), _slot("HE", 256, 1, 0.400),
        _slot("HE", 128, 6, 0.110), _slot("HE", 256, 7, 0.420),
        _slot("attention-32h", 128, 2, 0.500), _slot("attention-32h", 256, 3, 0.800),
        _slot("attention-32h", 128, 4, 0.505), _slot("attention-32h", 256, 5, 0.810),
    ]
    pooled = [
        {"arm": "HE", "context_tokens": 128, "latency_stats": {"median": 0.100}},
        {"arm": "attention-32h", "context_tokens": 128, "latency_stats": {"median": 0.150}},
        {"arm": "HE", "context_tokens": 256, "latency_stats": {"median": 0.400}},
        {"arm": "attention-32h", "context_tokens": 256, "latency_stats": {"median": 0.405}},
    ]
    return {"prefill_slots": slots, "prefill": pooled}


def test_the_a_a_null_is_a_same_context_null():
    systems = _slot_system()
    nulls = rr.systems_null(systems)
    floors = nulls["prefill"]["floor_per_context_seconds"]
    # HE: 0.110 - 0.100 = 0.010 at context 128, 0.420 - 0.400 = 0.020 at 256;
    # attention: 0.005 and 0.010. The floor of a context is the largest over arms.
    assert floors["128"] == pytest.approx(0.010)
    assert floors["256"] == pytest.approx(0.020)
    # the cross-context gap (0.400 - 0.100 = 0.300) must NOT become the floor
    assert max(floors.values()) < 0.3


def test_a_difference_is_resolvable_only_above_the_floor_of_its_context():
    systems = _slot_system()
    nulls = rr.systems_null(systems)
    verdicts = rr.systems_verdicts(systems, nulls)
    by_context = {v["context_tokens"]: v for v in verdicts}
    # context 128: 0.050 s apart, above the 0.010 s floor -> resolvable
    assert by_context["128"]["difference_seconds"] == pytest.approx(0.050)
    assert by_context["128"]["floor_seconds"] == pytest.approx(0.010)
    assert by_context["128"]["verdict"] == "resolvable"
    # context 256: 0.005 s apart, below the 0.020 s floor -> not resolvable
    assert by_context["256"]["difference_seconds"] == pytest.approx(0.005)
    assert by_context["256"]["floor_seconds"] == pytest.approx(0.020)
    assert by_context["256"]["verdict"] == "not resolvable"


def test_two_systems_legs_are_compared_against_the_drift_floor():
    """The discarded first systems leg is an independent measurement of the
    same quantities: the two legs must agree within the quoted leg's floor."""
    systems = _slot_system()
    systems["order_effect"] = {
        "prefill": {"floor_by_context_seconds": {"128": 0.010, "256": 0.020}}}
    alt = _slot_system()
    for row in alt["prefill"]:
        row["latency_stats"]["median"] += (0.002 if row["context_tokens"] == 128
                                           else 0.030)
    comparison = rr.systems_cross_leg(systems, alt)
    assert comparison["n_compared"] == 4
    # the context-128 rows differ by 0.002 s (inside the 0.010 s floor);
    # the context-256 rows by 0.030 s (outside the 0.020 s floor)
    assert comparison["n_within_floor"] == 2
    assert comparison["n_above_floor"] == 2
    assert comparison["max_difference_seconds"] == pytest.approx(0.030)


# ---------------------------------------------------------------------------
# the renderer-key invariant
# ---------------------------------------------------------------------------
def _minimal_results() -> dict:
    def runs(losses):
        return [{"seed": s, "final_held_out_loss": loss,
                 "evals": [{"held_out_loss": loss} for _ in range(4)]}
                for s, loss in losses.items()]

    return {
        "protocol": {"seeds": [1, 2, 3]},
        "results": {"HE": {"runs": runs({1: 3.10, 2: 3.12, 3: 3.08})},
                    "attention-32h": {"runs": runs({1: 3.00, 2: 3.00, 3: 3.00})}},
        "paired_final": {
            "seeds": [1, 2, 3],
            "arm_a_final_held_out_loss": {"1": 3.10, "2": 3.12, "3": 3.08},
            "arm_b_final_held_out_loss": {"1": 3.00, "2": 3.00, "3": 3.00},
            "mean_difference": 0.10,
            "std_difference": 0.02,
        },
    }


def test_an_artefact_without_the_renderers_difference_key_makes_the_tool_fail(tmp_path,
                                                                             monkeypatch,
                                                                             capsys):
    import json

    path = tmp_path / "quality.json"
    path.write_text(json.dumps(_minimal_results()), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["review_recompute.py", "--file", str(path)])
    assert rr.main() == 1
    out = capsys.readouterr().out
    assert "the artefact carries the key the renderer reads" in out
    assert "REVIEW_RECOMPUTE_FAIL" in out


def test_an_artefact_with_the_difference_key_passes_the_key_check(tmp_path, monkeypatch,
                                                                 capsys):
    import json

    data = _minimal_results()
    data["paired_final"]["differences_arm_a_minus_arm_b"] = {
        "1": 0.10, "2": 0.12, "3": 0.08}
    path = tmp_path / "quality.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["review_recompute.py", "--file", str(path)])
    assert rr.main() == 0
    out = capsys.readouterr().out
    assert "REVIEW_RECOMPUTE_OK" in out
