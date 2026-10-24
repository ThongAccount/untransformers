"""The results-table renderer is checked.

The write-ups must carry the measured numbers, not retyped numbers, so the
tables are rendered straight from the artefact. A renderer that silently drops
a row would put a half-truth in the docs; these tests make it fail loudly.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import render_results_tables as rrt  # noqa: E402


def _quality():
    return {
        "label": "full",
        "config": {"d": 128, "l": 128, "layers": 18, "ffn": 512, "vocab": 5000},
        "protocol": {"steps": 2000, "eval_every": 250, "eval_batches": 8,
                     "seeds": [1, 2, 3], "train_seed": 0,
                     "tokens_per_step": 8192, "tokens_processed_per_run": 16_384_000},
        "corpus": {"corpus_id": "a" * 64, "tokenizer_sha256": "b" * 64,
                   "n_tokens": 9_514_186},
        "environment": {"device": "cuda", "gpu": "Tesla T4", "torch": "2.11.0+cu130",
                        "python": "3.13.15", "torch_threads": 1},
        "results": {
            "HE": {"arm": "HE", "seeds": [1, 2],
                   "runs": [{"arm": "HE", "seed": 1, "steps": 2000,
                             "final_held_out_loss": 6.5, "evals": []},
                            {"arm": "HE", "seed": 2, "steps": 2000,
                             "final_held_out_loss": 6.7, "evals": []}],
                   "dispersion": {"250": {"n": 2, "mean": 7.0, "std": 0.1}}},
        },
        "paired_final": {"seeds": [1, 2],
                         "arm_a_final_held_out_loss": {"1": 6.5, "2": 6.7},
                         "arm_b_final_held_out_loss": {"1": 6.4, "2": 6.6},
                         "differences_arm_a_minus_b": {"1": 0.1, "2": 0.1},
                         "mean_difference": 0.1, "std_difference": 0.0, "n": 2},
    }


def _systems():
    return {
        "environment": {"device": "cuda", "gpu": "Tesla T4", "torch": "2.11.0+cu130",
                        "python": "3.13.15", "torch_threads": 1,
                        "numerics": {"dtype": "float32", "allow_tf32_matmul": False,
                                     "allow_tf32_cudnn": False,
                                     "float32_matmul_precision": "highest",
                                     "cudnn_benchmark": False,
                                     "cudnn_deterministic": True},
                        "arms": ["HE", "attention-32h", "attention-32h-sdpa"],
                        "git_commit": "c" * 40, "git_branch": "research/x",
                        "git_dirty": False},
        "config": {"d": 128, "l": 128, "n_layers": 18, "ffn_width": 512, "vocab": 5000,
                   "batch_size": 64, "n_heads": 32, "trials": 10, "warmup": 2,
                   "gen_tokens": 16, "decode_trials": 5,
                   "order_schedule": ["HE", "attention-32h", "attention-32h-sdpa",
                                      "attention-32h-sdpa", "attention-32h", "HE"],
                   "order_schedule_kind": "palindromic (ABBA)"},
        "training_step": [{"arm": "HE",
                           "step_seconds": {"median": 0.5, "mean": 0.5, "p10": 0.4,
                                            "p90": 0.6, "min": 0.35, "n": 20},
                           "tokens_per_second": 1000.0, "n_params": 10,
                           "memory": {"peak_rss_bytes": 1, "scope": "per-arm"},
                           "quality": {"loaded": True, "seed": 1,
                                       "final_held_out_loss": 6.5},
                           "kernel_caveat": "eager loop vs fused kernel"}],
        "prefill": [{"arm": "HE", "context_tokens": 8,
                      "latency": {"median": 0.01, "p10": 0.009, "p90": 0.011,
                                  "min": 0.009, "n": 20},
                      "tokens_per_second": 10.0, "memory": {"peak_rss_bytes": 1},
                      "quality": {"loaded": False}}],
        "incremental_decode": [{"arm": "HE", "context_tokens": 8,
                                "per_token_seconds": {"median": 0.002, "n": 16},
                                "first_token_seconds": {"median": 0.003},
                                "steady_state_per_token_seconds": {"median": 0.0019},
                                "tokens_per_second": 500.0,
                                "tokens_per_second_including_first_token": 480.0,
                                "cache_bytes_at_prompt": 64,
                                "cache_bytes_per_layer": 32,
                                "quality": {"loaded": False}}],
        "sublayer_scaling": [{"arm": "HE", "context_tokens": 128,
                              "per_token_seconds": {"median": 0.002, "n": 10},
                              "state_bytes_per_layer": 10,
                              "state_bytes_all_layers": 180,
                              "state_bytes_formula": "(l - 1) * d * 4 bytes"}],
        "state_traffic": {"HE": {"state_bytes_per_layer_per_token": 10,
                                 "state_bytes_all_layers_per_token": 180,
                                 "state_bytes_per_layer_formula": "(l - 1) * d * 4",
                                 "grows_with_context": False,
                                 "note": "HE's state is flat"}},
        "weight_traffic": {"HE": {"n_params": 10, "parameter_bytes_fp32": 40,
                                  "training_step_parameter_bytes_read": 120,
                                  "decode_step_parameter_bytes_read": 40,
                                  "training_step_parameter_bytes_formula":
                                      "3 * n_params * 4 bytes",
                                  "decode_step_parameter_bytes_formula":
                                      "1 * n_params * 4 bytes",
                                  "note": "analytic"}},
        "order_effect": {"training_step": {
            "block": "training_step",
            "aa_null": {"HE": {"first_slot_order_index": 0, "last_slot_order_index": 5,
                               "first_slot_median_seconds": 0.5,
                               "last_slot_median_seconds": 0.51,
                               "difference_seconds": 0.01,
                               "relative_difference": 0.02}},
            "drift_floor_seconds": 0.01, "drift_floor_relative": 0.02}},
        "caveats": ["the CPU memory rows are process-level"],
        "paired_quality": {"HE": {"1": {"steps": 2000, "final_held_out_loss": 6.5}}},
    }


def test_quality_tables_carry_conditions_and_numbers(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps(_quality()), encoding="utf-8")
    md = rrt.render_quality(p)
    assert "Tesla T4" in md and "2.11.0+cu130" in md
    assert "16_384_000" in md or "16384000" in md
    assert "| HE | 1 | 2000 | 6.500000 |" in md
    assert "0.100000" in md          # the paired difference
    assert "6.000000" not in md      # nothing invented
    assert "| 250 | 2 | 7.000000 | 0.100000 |" in md


def test_systems_tables_carry_conditions_and_numbers(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(_systems()), encoding="utf-8")
    md = rrt.render_systems(p)
    assert "Tesla T4" in md
    # the conditions: the numerics flags and the commit
    assert "highest" in md and "allow_tf32" in md
    assert "c" * 40 in md
    # the training step: median, mean, p10, p90, min, n, throughput, memory scope
    assert "| HE | 0.5000 | 0.5000 | 0.4000 | 0.6000 | 0.3500 | 20 | 1000.0 | 10 | 1 |" in md
    # ... and the quality level of the timed model (S8)
    assert "trained checkpoint (seed 1, held-out loss 6.500000)" in md
    # the prefill row carries its percentiles and n
    assert "| HE | 8 | 0.01000 | 0.00900 | 0.01100 | 0.00900 | 20 | 10.00 |" in md
    # the decode row separates the first token from the steady state (S4)
    assert "| HE | 8 | 0.003000 | 0.001900 | 0.002000 | 16 |" in md
    # the sublayer row is labelled analytic and prints its formula
    assert "| HE | 128 | 0.002000 |" in md and "180 |" in md
    assert "(l - 1) * d * 4 bytes" in md
    # the traffic block
    assert "| HE | 10 | 40 | 120 | 40 |" in md
    assert "3 * n_params * 4 bytes" in md
    # the A/A null and the drift floor (S2)
    assert "Run-order null" in md and "0.0100" in md
    # the caveats travel with the artefact
    assert "the CPU memory rows are process-level" in md


def test_a_nan_is_printed_as_a_dash_and_never_as_the_word_nan():
    """A NaN is not a number: the tables must not print it as one.

    `float('nan')` formats as "nan" in Python, so a table could carry the
    literal string "nan" into the write-up, where a reader would take it for a
    measurement. The renderer shows an em dash instead, and the dash is the
    only thing a missing value may look like.
    """
    assert rrt._fmt(float("nan")) == "—"
    assert rrt._fmt(None) == "—"
    assert rrt._fmt(float("inf")) == "∞"
    assert rrt._fmt(1.5, 2) == "1.50"
    md = rrt._fmt(float("nan")) + " " + rrt._fmt(None)
    assert "nan" not in md.lower()


def test_the_conditions_table_carries_everything_protocol_rule_1_requires(tmp_path):
    """Protocol §7 rule 1: a headline number carries the seeds, the steps, the
    tokens processed, the nominal epochs, the corpus identity, the held-out
    definition, the eval window count and target tokens, and the device."""
    p = tmp_path / "q.json"
    d = _quality()
    d["protocol"].update({
        "epochs_of_our_corpus": 1.813,
        "corpus_tokens_train_region": 9_038_476,
        "corpus_tokens_heldout_region": 475_710,
    })
    d["config"]["batch_size"] = 64
    d["config"]["l"] = 128
    p.write_text(json.dumps(d), encoding="utf-8")
    md = rrt.render_quality_conditions(p)
    for needle in ("nominal epochs of the train region", "1.813",
                   "eval windows / target tokens per eval point",
                   "512 windows / 65536 target tokens",
                   "token budget vs the paper", "held-out definition",
                   "corpus_id", "tokenizer sha256", "train seed",
                   "tokens processed per run"):
        assert needle in md, needle
    # the token budget is stated as a share of the paper's, with the paper's own
    # budget printed so the ratio can be checked
    assert "491,520,000" in md
