"""The A/A null of the systems comparison must be a SAME-CONTEXT null.

The prefill and incremental-decode blocks carry one row per context length per
slot. An arm-level comparison across the whole block would measure the growth
of the latency with the context, not the run-order drift — and every verdict
judged against such a floor would be right for the wrong reason (the same
shape of error as a gate that cannot fail).

The floor was fixed before the systems numbers existed
(`docs/milestone-3/SCOPE.md` §3.2): for a block and a context it is the
max-min spread across the A/A null's repetitions (the same arm at the start
and at the end of the palindromic order, AT THAT CONTEXT), taken as the
largest such spread over the arms at that context.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

import run_systems as rs  # noqa: E402


def _slot(arm, order_index, ctx, median):
    row = {"arm": arm, "order_index": order_index,
           "latency_stats": {"median": median, "n": 10}}
    if ctx is not None:
        row["context_tokens"] = ctx
    return row


def test_the_null_is_computed_per_arm_and_context():
    slots = [
        _slot("HE", 0, 1, 0.10), _slot("HE", 0, 112, 1.10),
        _slot("attention-32h", 1, 1, 0.11), _slot("attention-32h", 1, 112, 1.11),
        _slot("attention-32h", 2, 1, 0.12), _slot("attention-32h", 2, 112, 1.12),
        _slot("HE", 3, 1, 0.105), _slot("HE", 3, 112, 1.105),
    ]
    out = rs._order_effect(slots, "prefill")
    by_ctx = out["aa_null_by_context"]
    assert set(by_ctx) == {"1", "112"}
    # the null at context 1 compares the first and the last slot of the SAME
    # arm at that context: HE 0.105 - 0.10 = 0.005, attention 0.12 - 0.11 = 0.01
    assert abs(by_ctx["1"]["HE"]["difference_seconds"] - 0.005) < 1e-12
    assert abs(by_ctx["1"]["attention-32h"]["difference_seconds"] - 0.01) < 1e-12
    assert by_ctx["1"]["HE"]["first_slot_order_index"] == 0
    assert by_ctx["1"]["HE"]["last_slot_order_index"] == 3
    # the floor of a context is the largest same-context spread over the arms
    assert abs(out["floor_by_context_seconds"]["1"] - 0.01) < 1e-12
    assert abs(out["floor_by_context_seconds"]["112"] - 0.01) < 1e-12
    # the block floor is the largest of those floors — never a cross-context gap
    # (the cross-context gap here is 1.10 - 0.10 = 1.00)
    assert abs(out["drift_floor_seconds"] - 0.01) < 1e-12
    assert out["drift_floor_seconds"] < 1.0


def test_the_training_step_block_has_a_model_shape_null():
    slots = [_slot("HE", 0, None, 0.50), _slot("HE", 3, None, 0.52)]
    out = rs._order_effect(slots, "training_step")
    assert set(out["aa_null_by_context"]) == {"None"}
    entry = out["aa_null_by_context"]["None"]["HE"]
    assert abs(entry["difference_seconds"] - 0.02) < 1e-12
    assert abs(out["floor_by_context_seconds"]["None"] - 0.02) < 1e-12
    assert abs(out["drift_floor_seconds"] - 0.02) < 1e-12


def test_an_arm_measured_once_has_no_null():
    slots = [_slot("HE", 0, 1, 0.10)]
    out = rs._order_effect(slots, "prefill")
    assert out["aa_null_by_context"] == {}
    assert out["drift_floor_seconds"] == 0.0
