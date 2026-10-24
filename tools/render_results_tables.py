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
        diffs = (paired.get("differences_arm_a_minus_arm_b")
                 or paired.get("differences_arm_a_minus_b") or {})
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
        out.append("| step | n | mean held-out loss | std (population, ddof=0 — "
                   "descriptive only) |\n|---|---|---|---|")
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


def _ctx_sort_key(key):
    """Sort the context keys numerically ("8" before "112"), model shape first."""
    if key == "None":
        return (0, 0.0, "")
    try:
        return (1, float(key), key)
    except (TypeError, ValueError):
        return (2, 0.0, str(key))


def _slot_median(row: dict):
    """The median of one measurement slot, whatever block it came from.

    The slot rows carry the timed statistics under `latency_stats`; the
    fallback keys are the block-specific ones.
    """
    for key in ("latency_stats", "latency", "step_seconds",
                "steady_state_per_token_seconds"):
        block = row.get(key)
        if isinstance(block, dict) and block.get("median") is not None:
            return float(block["median"])
    return None


def _aa_spread_by_context(slots: list[dict]) -> dict:
    """(arm, context) -> the A/A null's spread, in seconds.

    The null is the same arm measured at the start and at the end of the
    palindromic order **at the same context**. Its repetitions are those two
    slot medians, so the spread is the max-min over them.

    It has to be same-context: the prefill and incremental-decode blocks carry
    one row per context length per slot, so an arm-level comparison across the
    whole block would measure the growth of the latency with the context, not
    the run-order drift.
    """
    by: dict = {}
    for s in slots:
        med = _slot_median(s)
        if med is None:
            continue
        key = (s.get("arm"), s.get("context_tokens"))
        by.setdefault(key, []).append((s.get("order_index", 0), med))
    out: dict = {}
    for key, rows in by.items():
        if len(rows) < 2:
            continue
        rows.sort()
        out[key] = abs(rows[-1][1] - rows[0][1])
    return out


def _floor_by_context(slots: list[dict]) -> dict:
    """context -> the drift floor: the largest A/A spread over the arms.

    The floor is the conservative one (the largest spread at that context), so
    a difference has to beat the worst arm's own drift to count.
    """
    floors: dict = {}
    for (_arm, ctx), spread in _aa_spread_by_context(slots).items():
        if ctx not in floors or spread > floors[ctx]:
            floors[ctx] = spread
    return floors


def _block_medians(block: str, rows: list[dict]) -> dict:
    """(arm, context) -> the pooled median of that block."""
    out: dict = {}
    for r in rows:
        if block == "training_step":
            med = (r.get("step_seconds") or {}).get("median")
            ctx = None
        elif block == "prefill":
            med = (r.get("latency") or {}).get("median")
            ctx = r.get("context_tokens")
        else:
            med = (r.get("steady_state_per_token_seconds") or {}).get("median")
            ctx = r.get("context_tokens")
        if med is None:
            continue
        out[(r.get("arm"), ctx)] = med
    return out


def render_systems_differences(d: dict) -> str:
    """Every arm-to-arm difference, with the drift floor next to it.

    The rule was fixed before the systems numbers existed (the reviewer's
    pre-registration, `docs/milestone-3/SCOPE.md` §3.2): a difference is
    resolvable iff its absolute value exceeds the A/A null's spread for the
    same block and context; below the floor it is not resolvable, whatever its
    sign.
    """
    blocks = (
        ("training_step", d.get("training_step") or [], d.get("training_step_slots") or []),
        ("prefill", d.get("prefill") or [], d.get("prefill_slots") or []),
        ("incremental_decode", d.get("incremental_decode") or [],
         d.get("incremental_decode_slots") or []),
    )
    out: list[str] = []
    out.append("### Arm-to-arm differences against the drift floor\n")
    out.append("The floor for a block and a context is the **max−min spread across "
               "the A/A null's repetitions** (the same arm at the start and at the "
               "end of the palindromic order), taken as the **largest** such spread "
               "over the arms at that context. A difference is **resolvable** iff "
               "its absolute value exceeds the floor; below the floor it is **not "
               "resolvable, whatever its sign**. The rule was fixed before the "
               "systems numbers existed (`docs/milestone-3/SCOPE.md` §3.2). "
               "Exceeding the floor is necessary, not sufficient, for a claim.\n")
    out.append("| block | context | arm A | arm B | median A (s) | median B (s) | "
               "difference (s) | relative | floor (s) | resolvable? |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    any_row = False
    for block, rows, slots in blocks:
        meds = _block_medians(block, rows)
        floors = _floor_by_context(slots)
        arms = sorted({arm for arm, _ctx in meds})
        contexts = sorted({ctx for _arm, ctx in meds}, key=lambda c: (c is None, c))
        for ctx in contexts:
            present = [a for a in arms if (a, ctx) in meds]
            for i, a in enumerate(present):
                for b in present[i + 1:]:
                    ma, mb = meds[(a, ctx)], meds[(b, ctx)]
                    diff = ma - mb
                    floor = floors.get(ctx)
                    rel = (abs(diff) / mb) if mb else None
                    if floor is None:
                        verdict = "no floor measured"
                    elif abs(diff) > floor:
                        verdict = "**resolvable**"
                    else:
                        verdict = "not resolvable"
                    any_row = True
                    ctx_label = "model shape" if ctx is None else str(ctx)
                    out.append(f"| {block} | {ctx_label} | {a} | {b} | "
                               f"{_fmt(ma, 6)} | {_fmt(mb, 6)} | {_fmt(diff, 6)} | "
                               f"{_fmt(rel, 4)} | {_fmt(floor, 6)} | {verdict} |")
    if not any_row:
        return ""
    out.append("")
    return "\n".join(out)


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
                   "same arm appears at the start and at the end of the order **at the "
                   "same context**. The spread of that null is the run-order drift "
                   "floor: an arm-to-arm difference smaller than the floor of its "
                   "context is not resolvable against clock/thermal drift. The floor "
                   "of a context is the largest such spread over the arms.\n")
        out.append("| block | context | arm | first slot (index) | last slot (index) | "
                   "first median s | last median s | difference s | relative | "
                   "floor at this context (s) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for block, entry in oe.items():
            by_ctx = entry.get("aa_null_by_context") or {}
            floors = entry.get("floor_by_context_seconds") or {}
            for ctx_key in sorted(by_ctx, key=_ctx_sort_key):
                arms = by_ctx[ctx_key] or {}
                floor = floors.get(ctx_key)
                for arm in sorted(arms):
                    row = arms[arm] or {}
                    ctx_label = "model shape" if ctx_key == "None" else ctx_key
                    out.append(f"| {block} | {ctx_label} | {arm} | "
                               f"{row.get('first_slot_order_index')} | "
                               f"{row.get('last_slot_order_index')} | "
                               f"{_fmt(row.get('first_slot_median_seconds'), 6)} | "
                               f"{_fmt(row.get('last_slot_median_seconds'), 6)} | "
                               f"{_fmt(row.get('difference_seconds'), 6)} | "
                               f"{_fmt(row.get('relative_difference'), 4)} | "
                               f"{_fmt(floor, 6)} |")
        out.append("")
        out.append("| block | the artefact's block-level A/A difference (s) | "
                   "relative |\n|---|---|---|")
        for block, entry in oe.items():
            out.append(f"| {block} | {_fmt(entry.get('drift_floor_seconds'), 6)} | "
                       f"{_fmt(entry.get('drift_floor_relative'), 4)} |")
        out.append("")
        out.append("The artefact's block-level difference is the **largest** of the "
                   "per-context floors above — the same-context null, never a "
                   "cross-context one. The verdicts in the next table use the floor "
                   "of the context they are printed in.\n")

    out.append(render_systems_differences(d))

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
                "eval_batches", "context_length_l", "batch_size", "eval_windows",
                "eval_target_tokens", "corpus_id", "tokenizer_sha256", "device",
                "torch", "python", "commit"):
        if key in cond:
            out.append(f"| {key} | {_fmt(cond[key])} |")
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
        out.append(f"| CI excludes 0 | {_fmt(block.get('ci95_excludes_zero'))} |")
        out.append(f"| MDD at 80% power | {_fmt(block.get('mdd_80_power'), 6)} |")
        out.append(f"| perplexity ratio exp(d-bar) | {_fmt(block.get('perplexity_ratio'), 6)} |")
        pr = block.get("perplexity_ratio_ci95") or [None, None]
        out.append(f"| perplexity ratio, 95% CI | [{_fmt(pr[0], 6)}, {_fmt(pr[1], 6)}] |")
        out.append(f"| verdict | {_fmt(block.get('verdict'))} |")
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
    # the resample count and the window count are the artefact's, not literals:
    # a header that says "1,000 resamples of the 512 eval windows" while the
    # artefact says otherwise is a drift the reader cannot see.
    first = next(iter(per_seed.values()), {}) or {}
    n_resamples = first.get("n_resamples")
    n_windows = first.get("n_windows")
    header = "### Eval-set bootstrap"
    if n_resamples is not None and n_windows is not None:
        header += f" ({int(n_resamples):,} resamples of the {int(n_windows):,} eval windows)"
    out.append(header + "\n")
    out.append("| seed | bootstrap SE, HE | bootstrap SE, attention-32h | "
               "bootstrap SE, paired difference |\n|---|---|---|---|")
    for s, b in sorted(per_seed.items()):
        out.append(f"| {s} | {_fmt(b.get('bootstrap_se_arm_a'), 6)} | "
                   f"{_fmt(b.get('bootstrap_se_arm_b'), 6)} | "
                   f"{_fmt(b.get('bootstrap_se_paired_difference'), 6)} |")
    out.append("")
    # the note lives on the per-seed entries (it is the per-seed statement that
    # the SE is not part of the training-noise CI); the block carries the
    # definition of the resampling. Either is fine, but the renderer must
    # never print the word "None" into a committed write-up.
    note = boot.get("note")
    if not note:
        for b in per_seed.values():
            note = (b or {}).get("note")
            if note:
                break
    if not note:
        note = boot.get("definition")
    if note:
        out.append(f"_{note}_\n")
    return "\n".join(out)


def summary_sensitivity(d: dict) -> str:
    sens = d.get("book_aligned_sensitivity") or {}
    if not sens:
        return ""
    out: list[str] = []
    out.append("### Book-aligned sensitivity analysis (post-hoc, 3 unseen books)\n")
    if sens.get("definition"):
        out.append(f"_{sens['definition']}_\n")
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

    # the coverage, stated honestly: with stride 1 the windows overlap, so the
    # count of target positions is a count of TOKEN-PREDICTIONS, not of
    # distinct tokens, and the first token of each book is never a target.
    per_book = sens.get("per_book") or {}
    if per_book:
        l_ctx = ((d.get("conditions") or {}).get("context_length_l")
                 or (d.get("conditions") or {}).get("l"))
        out.append("### Book-aligned coverage (how much was actually scored)\n")
        out.append("The stride-1 sweep places one window start at every token "
                   "position inside each unseen book, so the windows **overlap**: "
                   "the target count below is a count of **token-predictions** "
                   "(windows x the context length), **not** of distinct tokens, "
                   "and the first token of each book is never a target (a window "
                   "needs `l + 1` tokens to predict its last one).\n")
        out.append("| checkpoint | book | book tokens | windows | stride | "
                   "token-predictions | mean loss |")
        out.append("|---|---|---|---|---|---|---|")
        for ckpt in sorted(per_book):
            books = per_book[ckpt] or {}
            for ebook_id in sorted(books, key=lambda k: int(k)):
                b = books[ebook_id] or {}
                windows = b.get("n_windows")
                preds = (windows * l_ctx) if (windows and l_ctx) else None
                out.append(f"| {ckpt} | {ebook_id} | {_fmt(b.get('n_tokens'))} | "
                           f"{_fmt(windows)} | {_fmt(b.get('stride'))} | "
                           f"{_fmt(preds)} | {_fmt(b.get('mean_loss'), 6)} |")
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
