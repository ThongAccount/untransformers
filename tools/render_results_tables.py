#!/usr/bin/env python3
"""Renders markdown tables from a measured results file.

The write-ups must carry the measured numbers, not retyped numbers. This
script prints the tables that go into `docs/milestone-2/QUALITY.md` and
`docs/milestone-3/SYSTEMS.md`, straight from the JSON artefact that the run
produced — so a number in the docs is by construction the number in the file.

Usage:
    python tools/render_results_tables.py quality results/quality_gpu_full.json
    python tools/render_results_tables.py systems results/systems.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _fmt(x, nd=4):
    """Format one number for a markdown cell.

    A missing value and a NaN are both shown as an em dash: a table must never
    print "nan" (it is not a number, it is a bug in the arithmetic upstream) and
    it must never print "None" (that is a Python object leaking into prose).
    """
    if x is None:
        return "—"
    if isinstance(x, float):
        if x != x:  # NaN
            return "—"
        if x in (float("inf"), float("-inf")):
            return "∞" if x > 0 else "−∞"
        return f"{x:.{nd}f}"
    return str(x)


def render_quality_conditions(path: Path) -> str:
    """The conditions table of a quality run, on its own.

    Protocol §7 rule 1: every headline number carries arm, seeds, steps, tokens
    processed, nominal epochs, corpus_id + tokenizer digest, the held-out
    definition, the eval window count and target tokens, device/torch, trials
    and warmup. This table is where those conditions live.
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    cfg = d.get("config", {})
    proto = d.get("protocol", {})
    env = d.get("environment", {})
    corp = d.get("corpus", {})
    out: list[str] = []

    out.append("### Conditions\n")
    out.append("| field | value |\n|---|---|")
    out.append(f"| label | {d.get('label')} |")
    out.append(f"| commit | {d.get('commit', '—')} |")
    out.append(f"| corpus_id | `{corp.get('corpus_id', '—')}` |")
    out.append(f"| tokenizer sha256 | `{corp.get('tokenizer_sha256', '—')}` |")
    out.append(f"| books | {corp.get('books', '—')} |")
    train_tok = proto.get("corpus_tokens_train_region", "—")
    held_tok = proto.get("corpus_tokens_heldout_region", "—")
    out.append(f"| tokens: total / train region / held-out region | "
               f"{corp.get('n_tokens', '—')} / {train_tok} / {held_tok} |")
    out.append(f"| held-out definition | the last 5% of the frozen token array "
               f"({held_tok} tokens); never trained on, never used for selection |")
    out.append(f"| model shape | d={cfg.get('d')} l={cfg.get('l')} "
               f"layers={cfg.get('n_layers', cfg.get('layers'))} "
               f"ffn={cfg.get('ffn_width', cfg.get('ffn'))} "
               f"vocab={cfg.get('vocab')} "
               f"heads={cfg.get('n_heads', cfg.get('heads', 32))} |")
    out.append(f"| steps | {proto.get('steps')} |")
    out.append(f"| tokens per step | {proto.get('tokens_per_step')} |")
    out.append(f"| tokens processed per run | {proto.get('tokens_processed_per_run')} |")
    epochs = proto.get("epochs_of_our_corpus")
    out.append(f"| nominal epochs of the train region | "
               f"{_fmt(epochs, 3) if epochs is not None else '—'} "
               f"(windows are sampled with replacement: repeats included) |")
    tokens_run = proto.get("tokens_processed_per_run")
    if isinstance(tokens_run, (int, float)):
        out.append(f"| token budget vs the paper | {_fmt(tokens_run / 491_520_000 * 100, 3)}% "
                   f"of the paper's 491,520,000 tokens (60,000 batches x 8,192) |")
    out.append(f"| seeds | {proto.get('seeds')} |")
    out.append(f"| eval every / eval batches | {proto.get('eval_every')} / {proto.get('eval_batches')} |")
    batch = cfg.get("batch_size")
    l_ctx = cfg.get("l")
    eval_batches = proto.get("eval_batches")
    if batch and l_ctx and eval_batches:
        windows = batch * eval_batches
        out.append(f"| eval windows / target tokens per eval point | "
                   f"{windows} windows / {windows * l_ctx} target tokens |")
    out.append(f"| train seed | {proto.get('train_seed')} |")
    out.append(f"| device | {env.get('device')} |")
    out.append(f"| hardware | {env.get('gpu', env.get('platform', '—'))} |")
    out.append(f"| torch | {env.get('torch')} |")
    out.append(f"| python | {env.get('python')} |")
    out.append(f"| threads | {env.get('torch_threads')} |")
    out.append("")

    return "\n".join(out)


def render_quality(path: Path) -> str:
    d = json.loads(path.read_text(encoding="utf-8"))
    out: list[str] = []

    # the conditions block is rendered in one place only
    out.append(render_quality_conditions(path))

    out.append("### Held-out loss per seed (final eval point)\n")
    out.append("| arm | seed | steps | final held-out loss | perplexity = exp(loss) |")
    out.append("|---|---|---|---|---|")
    for arm, res in (d.get("results") or {}).items():
        for r in res.get("runs", []):
            loss = r.get("final_held_out_loss")
            import math
            ppl = math.exp(loss) if isinstance(loss, (int, float)) else None
            out.append(f"| {arm} | {r.get('seed')} | {r.get('steps')} | "
                       f"{_fmt(loss, 6)} | {_fmt(ppl, 2)} |")
    out.append("")

    paired = d.get("paired_final") or {}
    if paired:
        out.append("### Paired per-seed differences (HE − attention-32h; positive = HE worse)\n")
        out.append("| seed | HE | attention-32h | difference |\n|---|---|---|---|")
        fa = paired.get("arm_a_final_held_out_loss", {})
        fb = paired.get("arm_b_final_held_out_loss", {})
        diffs = paired.get("differences_arm_a_minus_b", {})
        for s in paired.get("seeds", []):
            out.append(f"| {s} | {_fmt(fa.get(str(s)), 6)} | {_fmt(fb.get(str(s)), 6)} | "
                       f"{_fmt(diffs.get(str(s)), 6)} |")
        out.append(f"| **mean** | | | {_fmt(paired.get('mean_difference'), 6)} |")
        # the harness's own spread is a POPULATION SD (ddof=0). Protocol §7 rule 3
        # forbids quoting it as the dispersion: the pre-registered statistic is
        # the sample SD (ddof=1) with its CI and the MDD, and those are in the
        # statistics table of the write-up, not here.
        out.append(f"| population SD of the differences (ddof=0, descriptive only) "
                   f"| | | {_fmt(paired.get('std_difference'), 6)} |")
        out.append(f"| n | | | {paired.get('n')} |")
        out.append("")

    out.append("### Dispersion across seeds (per eval point)\n")
    for arm, res in (d.get("results") or {}).items():
        disp = res.get("dispersion") or {}
        if not isinstance(disp, dict) or not disp:
            continue
        out.append(f"**{arm}**\n")
        out.append("| step | n | mean held-out loss | std |\n|---|---|---|---|")
        for step, p in sorted(disp.items(), key=lambda kv: int(kv[0])):
            out.append(f"| {step} | {p.get('n')} | {_fmt(p.get('mean'), 6)} | "
                       f"{_fmt(p.get('std'), 6)} |")
        out.append("")
    return "\n".join(out)


def _quality_cell(q: dict) -> str:
    """The quality level of a timed row, in one cell."""
    if not q:
        return "—"
    if q.get("loaded"):
        return (f"trained checkpoint (seed {q.get('seed')}, "
                f"held-out loss {_fmt(q.get('final_held_out_loss'), 6)})")
    return "architecture at initialisation (no quality level)"


def render_systems(path: Path) -> str:
    d = json.loads(path.read_text(encoding="utf-8"))
    env = d.get("environment", {})
    cfg = d.get("config", {})
    num = env.get("numerics") or {}
    out: list[str] = []

    out.append("### Conditions\n")
    out.append("| field | value |\n|---|---|")
    out.append(f"| device | {env.get('device')} |")
    out.append(f"| hardware | {env.get('gpu', env.get('platform', '—'))} |")
    out.append(f"| torch | {env.get('torch')} |")
    out.append(f"| python | {env.get('python')} |")
    out.append(f"| threads | {env.get('torch_threads')} |")
    out.append(f"| commit | `{env.get('git_commit', '—')}` (branch `{env.get('git_branch', '—')}`, "
               f"dirty: {env.get('git_dirty')}) |")
    out.append(f"| arms | {env.get('arms')} |")
    out.append(f"| shape | d={cfg.get('d')} l={cfg.get('l')} layers={cfg.get('n_layers')} "
               f"ffn={cfg.get('ffn_width')} vocab={cfg.get('vocab')} heads={cfg.get('n_heads')} |")
    out.append(f"| batch | {cfg.get('batch_size')} |")
    out.append(f"| trials / warmup / decode trials | {cfg.get('trials')} / {cfg.get('warmup')} "
               f"/ {cfg.get('decode_trials')} |")
    out.append(f"| generated tokens | {cfg.get('gen_tokens')} |")
    out.append(f"| model contexts | {cfg.get('contexts')} |")
    out.append(f"| sublayer-1 contexts | {cfg.get('sublayer_contexts')} |")
    out.append(f"| measurement order | {cfg.get('order_schedule')} "
               f"({cfg.get('order_schedule_kind')}) |")
    out.append(f"| dtype | {num.get('dtype')} |")
    out.append(f"| allow_tf32 (matmul / cuDNN) | {num.get('allow_tf32_matmul')} / "
               f"{num.get('allow_tf32_cudnn')} |")
    out.append(f"| float32_matmul_precision | {num.get('float32_matmul_precision')} |")
    out.append(f"| cuDNN (benchmark / deterministic) | {num.get('cudnn_benchmark')} / "
               f"{num.get('cudnn_deterministic')} |")
    out.append(f"| deterministic_algorithms | {num.get('deterministic_algorithms')} |")
    out.append("")

    # --- the order null and the drift floor --------------------------------
    oe = d.get("order_effect") or {}
    if oe:
        out.append("### Run-order null (A/A) and the drift floor\n")
        out.append("Every arm is measured twice, in a palindromic (ABBA) order, so the "
                   "same arm appears at the start and at the end of the order. The "
                   "spread of that null is the run-order drift floor: an arm-to-arm "
                   "difference smaller than the floor is not resolvable against "
                   "clock/thermal drift.\n")
        out.append("| block | arm | first slot (index) | last slot (index) | "
                   "first median s | last median s | difference s | relative |")
        out.append("|---|---|---|---|---|---|---|---|")
        for block, entry in oe.items():
            for arm, row in (entry.get("aa_null") or {}).items():
                out.append(f"| {block} | {arm} | {row.get('first_slot_order_index')} | "
                           f"{row.get('last_slot_order_index')} | "
                           f"{_fmt(row.get('first_slot_median_seconds'), 6)} | "
                           f"{_fmt(row.get('last_slot_median_seconds'), 6)} | "
                           f"{_fmt(row.get('difference_seconds'), 6)} | "
                           f"{_fmt(row.get('relative_difference'), 4)} |")
        out.append("")
        out.append("| block | drift floor (s) | drift floor (relative) |\n|---|---|---|")
        for block, entry in oe.items():
            out.append(f"| {block} | {_fmt(entry.get('drift_floor_seconds'), 6)} | "
                       f"{_fmt(entry.get('drift_floor_relative'), 4)} |")
        out.append("")

    out.append("### Training step (forward + backward + optimizer step)\n")
    out.append("| arm | median s/step | mean | p10 | p90 | min | n | tok/s | params | "
               "peak memory bytes | memory scope | quality level |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in d.get("training_step", []):
        s = r.get("step_seconds", {})
        mem = r.get("memory") or {}
        out.append(f"| {r.get('arm')} | {_fmt(s.get('median'), 4)} | {_fmt(s.get('mean'), 4)} | "
                   f"{_fmt(s.get('p10'), 4)} | {_fmt(s.get('p90'), 4)} | "
                   f"{_fmt(s.get('min'), 4)} | {s.get('n')} | "
                   f"{_fmt(r.get('tokens_per_second'), 1)} | {r.get('n_params')} | "
                   f"{mem.get('peak_rss_bytes')} | {mem.get('scope', '—')} | "
                   f"{_quality_cell(r.get('quality'))} |")
    out.append("")
    seen_caveats = []
    for r in d.get("training_step", []):
        c = r.get("kernel_caveat")
        if c and c not in seen_caveats:
            seen_caveats.append(c)
    for c in seen_caveats:
        out.append(f"- {c}")
    out.append("")

    out.append("### Prefill latency (one full forward over the prompt)\n")
    out.append("| arm | context | median s | p10 | p90 | min | n | tok/s | "
               "peak memory bytes | quality level |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in d.get("prefill", []):
        lat = r.get("latency") or {}
        mem = r.get("memory") or {}
        out.append(f"| {r.get('arm')} | {r.get('context_tokens')} | "
                   f"{_fmt(lat.get('median'), 5)} | {_fmt(lat.get('p10'), 5)} | "
                   f"{_fmt(lat.get('p90'), 5)} | {_fmt(lat.get('min'), 5)} | "
                   f"{lat.get('n')} | {_fmt(r.get('tokens_per_second'), 2)} | "
                   f"{mem.get('peak_rss_bytes')} | {_quality_cell(r.get('quality'))} |")
    out.append("")

    out.append("### Incremental decoding (per generated token)\n")
    out.append("The first generated token is reported separately from the steady state; "
               "the pooled row keeps every sample.\n")
    out.append("| arm | context | first-token median s | steady-state median s | "
               "pooled median s | pooled n | steady tok/s | tok/s incl. first | "
               "cache bytes at prompt | per layer | quality level |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in d.get("incremental_decode", []):
        per = r.get("per_token_seconds") or {}
        first = r.get("first_token_seconds") or {}
        steady = r.get("steady_state_per_token_seconds") or {}
        out.append(f"| {r.get('arm')} | {r.get('context_tokens')} | "
                   f"{_fmt(first.get('median'), 6)} | {_fmt(steady.get('median'), 6)} | "
                   f"{_fmt(per.get('median'), 6)} | {per.get('n')} | "
                   f"{_fmt(r.get('tokens_per_second'), 2)} | "
                   f"{_fmt(r.get('tokens_per_second_including_first_token'), 2)} | "
                   f"{r.get('cache_bytes_at_prompt')} | {r.get('cache_bytes_per_layer')} | "
                   f"{_quality_cell(r.get('quality'))} |")
    out.append("")

    out.append("### Sublayer-1 scaling (one incremental token, one sublayer)\n")
    out.append("The state bytes are **analytic** (the formula is printed), not measured; "
               "the latencies are measured.\n")
    out.append("| arm | context | median s/token | p10 | p90 | n | state bytes/layer | "
               "state bytes all layers | formula |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in d.get("sublayer_scaling", []):
        per = r.get("per_token_seconds") or {}
        out.append(f"| {r.get('arm')} | {r.get('context_tokens')} | "
                   f"{_fmt(per.get('median'), 6)} | {_fmt(per.get('p10'), 6)} | "
                   f"{_fmt(per.get('p90'), 6)} | {per.get('n')} | "
                   f"{r.get('state_bytes_per_layer')} | {r.get('state_bytes_all_layers')} | "
                   f"`{_fmt(r.get('state_bytes_formula'))}` |")
    out.append("")

    out.append("### Per-token state traffic (analytic)\n")
    st = d.get("state_traffic") or {}
    out.append("| arm | state bytes/layer/token | all layers/token | formula | "
               "grows with context |")
    out.append("|---|---|---|---|---|")
    for arm, r in st.items():
        grows = r.get("grows_with_context")
        per_layer = r.get("state_bytes_per_layer_per_token")
        all_layers = r.get("state_bytes_all_layers_per_token")
        if per_layer is None and grows:
            # the attention state depends on the context length: one number
            # would be wrong, and "None" would leak Python into the table
            per_layer = "context-dependent (see the sublayer-1 table)"
            all_layers = "context-dependent (see the sublayer-1 table)"
        out.append(f"| {arm} | {_fmt(per_layer)} | "
                   f"{_fmt(all_layers)} | "
                   f"`{_fmt(r.get('state_bytes_per_layer_formula'))}` | "
                   f"{grows} |")
    out.append("")
    for arm, r in st.items():
        if r.get("note"):
            out.append(f"- *{arm}*: {r['note']}")
    out.append("")

    out.append("### Weight traffic (analytic)\n")
    wt = d.get("weight_traffic") or {}
    out.append("| arm | params | parameter bytes (fp32) | training-step bytes read | "
               "decode-step bytes read |")
    out.append("|---|---|---|---|---|")
    for arm, r in wt.items():
        out.append(f"| {arm} | {r.get('n_params')} | {r.get('parameter_bytes_fp32')} | "
                   f"{r.get('training_step_parameter_bytes_read')} | "
                   f"{r.get('decode_step_parameter_bytes_read')} |")
    out.append("")
    for arm, r in wt.items():
        if r.get("training_step_parameter_bytes_formula"):
            out.append(f"- *{arm}* training read: `{r['training_step_parameter_bytes_formula']}`; "
                       f"decode read: `{r.get('decode_step_parameter_bytes_formula')}`")
        if r.get("note"):
            out.append(f"- *{arm}* note: {r['note']}")
    out.append("")

    paired = d.get("paired_quality") or {}
    if paired:
        out.append("### Paired quality attached to the systems rows\n")
        out.append("| arm | seed | steps | final held-out loss |\n|---|---|---|---|")
        for arm, seeds in paired.items():
            for seed, row in sorted(seeds.items()):
                out.append(f"| {arm} | {seed} | {row.get('steps')} | "
                           f"{_fmt(row.get('final_held_out_loss'), 6)} |")
        out.append("")

    caveats = d.get("caveats") or []
    if caveats:
        out.append("### Caveats carried by the artefact\n")
        for c in caveats:
            out.append(f"- {c}")
        out.append("")
    return "\n".join(out)


def summary_conditions(d: dict) -> str:
    cond = d.get("conditions") or {}
    out: list[str] = []
    out.append("### Conditions of the comparison\n")
    out.append("| field | value |\n|---|---|")
    for key in ("arms", "seeds", "steps", "tokens_per_step", "tokens_processed_per_run",
                "nominal_epochs_of_train_region", "train_seed", "eval_every",
                "eval_batches", "corpus_id", "tokenizer_sha256", "device", "torch",
                "python", "commit"):
        if key in cond:
            out.append(f"| {key} | {cond[key]} |")
    out.append("")
    return "\n".join(out)


def summary_endpoints(d: dict) -> str:
    out: list[str] = []
    for endpoint in ("final", "last4"):
        block = (d.get("endpoints") or {}).get(endpoint)
        if not block:
            continue
        out.append(f"### Endpoint: {block.get('definition')}\n")
        out.append(f"Sign convention: {block.get('sign_convention')}\n")
        out.append("| seed | HE | attention-32h | d = HE - attention |\n|---|---|---|---|")
        fa = block.get("per_seed_arm_a") or {}
        fb = block.get("per_seed_arm_b") or {}
        diffs = block.get("differences") or {}
        for s in block.get("seeds") or []:
            out.append(f"| {s} | {_fmt(fa.get(str(s)), 6)} | {_fmt(fb.get(str(s)), 6)} | "
                       f"{_fmt(diffs.get(str(s)), 6)} |")
        out.append(f"| **mean** | {_fmt((block.get('arm_a') or {}).get('mean'), 6)} | "
                   f"{_fmt((block.get('arm_b') or {}).get('mean'), 6)} | "
                   f"{_fmt(block.get('mean_difference'), 6)} |")
        out.append(f"| **sample SD (ddof=1)** | "
                   f"{_fmt((block.get('arm_a') or {}).get('std_ddof1'), 6)} | "
                   f"{_fmt((block.get('arm_b') or {}).get('std_ddof1'), 6)} | "
                   f"{_fmt(block.get('std_difference_ddof1'), 6)} |")
        out.append("")
        out.append("| statistic | value |\n|---|---|")
        out.append(f"| standard error | {_fmt(block.get('standard_error'), 6)} |")
        ci = block.get("ci95") or [None, None]
        out.append(f"| 95% CI for d-bar | [{_fmt(ci[0], 6)}, {_fmt(ci[1], 6)}] |")
        out.append(f"| CI excludes 0 | {block.get('ci95_excludes_zero')} |")
        out.append(f"| MDD at 80% power | {_fmt(block.get('mdd_80_power'), 6)} |")
        out.append(f"| perplexity ratio exp(d-bar) | {_fmt(block.get('perplexity_ratio'), 6)} |")
        pr = block.get("perplexity_ratio_ci95") or [None, None]
        out.append(f"| perplexity ratio, 95% CI | [{_fmt(pr[0], 6)}, {_fmt(pr[1], 6)}] |")
        out.append(f"| verdict | {block.get('verdict')} |")
        needed = block.get("seeds_needed_for_80_power_at_observed_effect")
        if needed is not None:
            out.append(f"| seeds needed for 80% power at the observed effect | {needed} |")
        out.append("")
    return "\n".join(out)


def summary_bootstrap(d: dict) -> str:
    boot = d.get("eval_set_bootstrap") or {}
    per_seed = boot.get("per_seed") or {}
    if not per_seed:
        return ""
    out: list[str] = []
    out.append("### Eval-set bootstrap (1,000 resamples of the 512 eval windows)\n")
    out.append("| seed | bootstrap SE, HE | bootstrap SE, attention-32h | "
               "bootstrap SE, paired difference |\n|---|---|---|---|")
    for s, b in sorted(per_seed.items()):
        out.append(f"| {s} | {_fmt(b.get('bootstrap_se_arm_a'), 6)} | "
                   f"{_fmt(b.get('bootstrap_se_arm_b'), 6)} | "
                   f"{_fmt(b.get('bootstrap_se_paired_difference'), 6)} |")
    out.append("")
    out.append(f"_{boot.get('note')}_\n")
    return "\n".join(out)


def summary_sensitivity(d: dict) -> str:
    sens = d.get("book_aligned_sensitivity") or {}
    if not sens:
        return ""
    out: list[str] = []
    out.append("### Book-aligned sensitivity analysis (post-hoc, 3 unseen books)\n")
    out.append(f"_{sens.get('definition')}_\n")
    out.append("| seed | HE | attention-32h | d = HE - attention |\n|---|---|---|---|")
    fa = sens.get("per_seed_arm_a") or {}
    fb = sens.get("per_seed_arm_b") or {}
    diffs = sens.get("differences") or {}
    for s in sens.get("seeds") or []:
        out.append(f"| {s} | {_fmt(fa.get(str(s)), 6)} | {_fmt(fb.get(str(s)), 6)} | "
                   f"{_fmt(diffs.get(str(s)), 6)} |")
    ci = sens.get("ci95") or [None, None]
    out.append(f"| **mean** | | | {_fmt(sens.get('mean_difference'), 6)} |")
    out.append(f"| **95% CI** | | | [{_fmt(ci[0], 6)}, {_fmt(ci[1], 6)}] |")
    out.append(f"| **verdict** | | | {sens.get('verdict')} |")
    out.append("")
    return "\n".join(out)


def summary_curves(d: dict) -> str:
    curves = d.get("curves") or {}
    if not curves:
        return ""
    out: list[str] = []
    out.append("### The eval curves (every eval point, every seed)\n")
    arms = list(curves.keys())
    steps = []
    for arm in arms:
        for seed_rows in curves[arm].values():
            for e in seed_rows.get("evals", []):
                if e["step"] not in steps:
                    steps.append(e["step"])
    steps.sort()
    header = "| eval step |"
    sep = "|---|"
    for arm in arms:
        for seed in sorted(curves[arm].keys()):
            header += f" {arm} seed {seed} |"
            sep += "---|"
    out.append(header)
    out.append(sep)
    for step in steps:
        row = f"| {step} |"
        for arm in arms:
            for seed in sorted(curves[arm].keys()):
                evals = {e["step"]: e["held_out_loss"]
                         for e in curves[arm][seed].get("evals", [])}
                row += f" {_fmt(evals.get(step), 6)} |"
        out.append(row)
    out.append("")
    return "\n".join(out)


def render_summary(path: Path) -> str:
    """The pre-registered statistics block (protocol 3.2) from the summary."""
    d = json.loads(path.read_text(encoding="utf-8"))
    parts = [summary_conditions(d), summary_endpoints(d), summary_bootstrap(d),
             summary_sensitivity(d), summary_curves(d)]
    return "\n".join(p for p in parts if p)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    kind, path = sys.argv[1], Path(sys.argv[2])
    if kind == "quality":
        print(render_quality(path))
    elif kind == "systems":
        print(render_systems(path))
    elif kind == "summary":
        print(render_summary(path))
    else:
        print(f"unknown kind: {kind}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
