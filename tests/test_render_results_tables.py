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
                        "python": "3.13.15", "torch_threads": 1},
        "config": {"d": 128, "l": 128, "n_layers": 18, "ffn_width": 512, "vocab": 5000,
                   "batch_size": 64, "n_heads": 32, "trials": 5, "warmup": 2,
                   "gen_tokens": 16},
        "training_step": [{"arm": "HE", "step_seconds": {"median": 0.5, "mean": 0.5,
                                                          "p10": 0.4, "p90": 0.6},
                           "tokens_per_second": 1000.0, "n_params": 10,
                           "memory": {"peak_rss_bytes": 1}}],
        "prefill": [{"arm": "HE", "context_tokens": 8, "latency": {"median": 0.01},
                     "tokens_per_second": 10.0, "memory": {"peak_rss_bytes": 1}}],
        "incremental_decode": [{"arm": "HE", "context_tokens": 8,
                                "per_token_seconds": {"median": 0.002, "samples": 4},
                                "tokens_per_second": 500.0, "cache_bytes_at_prompt": 64}],
        "sublayer_scaling": [{"arm": "HE", "context_tokens": 128,
                              "per_token_seconds": {"median": 0.002, "samples": 4},
                              "state_bytes_per_layer": 10,
                              "state_bytes_all_layers": 180}],
        "weight_traffic": {"HE": {"n_params": 10, "parameter_bytes_fp32": 40,
                                  "training_step_parameter_bytes_read": 120,
                                  "decode_step_parameter_bytes_read": 40}},
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
    assert "| HE | 0.5000 |" in md
    assert "| HE | 8 | 0.01000 |" in md
    assert "| HE | 8 | 0.002000 | 4 |" in md
    assert "| HE | 128 | 0.002000 | 10 | 180 |" in md
    assert "| HE | 10 | 40 | 120 | 40 |" in md
