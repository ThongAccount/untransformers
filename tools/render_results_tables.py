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
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_quality(path: Path) -> str:
    d = json.loads(path.read_text(encoding="utf-8"))
    out: list[str] = []
    cfg = d.get("config", {})
    proto = d.get("protocol", {})
    env = d.get("environment", {})
    corp = d.get("corpus", {})

    out.append("### Conditions\n")
    out.append("| field | value |\n|---|---|")
    out.append(f"| label | {d.get('label')} |")
    out.append(f"| commit | {d.get('commit', '—')} |")
    out.append(f"| corpus_id | `{corp.get('corpus_id', '—')}` |")
    out.append(f"| tokenizer sha256 | `{corp.get('tokenizer_sha256', '—')}` |")
    out.append(f"| tokens (train / held-out) | {corp.get('n_tokens', '—')} |")
    out.append(f"| model shape | d={cfg.get('d')} l={cfg.get('l')} layers={cfg.get('layers')} "
               f"ffn={cfg.get('ffn')} vocab={cfg.get('vocab')} heads={cfg.get('heads')} |")
    out.append(f"| steps | {proto.get('steps')} |")
    out.append(f"| tokens per step | {proto.get('tokens_per_step')} |")
    out.append(f"| tokens processed per run | {proto.get('tokens_processed_per_run')} |")
    out.append(f"| seeds | {proto.get('seeds')} |")
    out.append(f"| eval every / eval batches | {proto.get('eval_every')} / {proto.get('eval_batches')} |")
    out.append(f"| train seed | {proto.get('train_seed')} |")
    out.append(f"| device | {env.get('device')} |")
    out.append(f"| hardware | {env.get('gpu', env.get('platform', '—'))} |")
    out.append(f"| torch | {env.get('torch')} |")
    out.append(f"| python | {env.get('python')} |")
    out.append(f"| threads | {env.get('torch_threads')} |")
    out.append("")

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
        out.append(f"| **std** | | | {_fmt(paired.get('std_difference'), 6)} |")
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


def render_systems(path: Path) -> str:
    d = json.loads(path.read_text(encoding="utf-8"))
    env = d.get("environment", {})
    cfg = d.get("config", {})
    out: list[str] = []

    out.append("### Conditions\n")
    out.append("| field | value |\n|---|---|")
    out.append(f"| device | {env.get('device')} |")
    out.append(f"| hardware | {env.get('gpu', env.get('platform', '—'))} |")
    out.append(f"| torch | {env.get('torch')} |")
    out.append(f"| python | {env.get('python')} |")
    out.append(f"| threads | {env.get('torch_threads')} |")
    out.append(f"| shape | d={cfg.get('d')} l={cfg.get('l')} layers={cfg.get('n_layers')} "
               f"ffn={cfg.get('ffn_width')} vocab={cfg.get('vocab')} heads={cfg.get('n_heads')} |")
    out.append(f"| batch | {cfg.get('batch_size')} |")
    out.append(f"| trials / warmup | {cfg.get('trials')} / {cfg.get('warmup')} |")
    out.append(f"| generated tokens | {cfg.get('gen_tokens')} |")
    out.append("")

    out.append("### Training step (forward + backward + optimizer)\n")
    out.append("| arm | median s/step | mean | p10 | p90 | tok/s | params | peak RSS bytes |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in d.get("training_step", []):
        s = r.get("step_seconds", {})
        mem = (r.get("memory") or {}).get("peak_rss_bytes")
        out.append(f"| {r.get('arm')} | {_fmt(s.get('median'), 4)} | {_fmt(s.get('mean'), 4)} | "
                   f"{_fmt(s.get('p10'), 4)} | {_fmt(s.get('p90'), 4)} | "
                   f"{_fmt(r.get('tokens_per_second'), 1)} | {r.get('n_params')} | {mem} |")
    out.append("")

    out.append("### Prefill latency (full forward over the prompt)\n")
    out.append("| arm | context | median s | tok/s | peak RSS bytes |\n|---|---|---|---|---|")
    for r in d.get("prefill", []):
        mem = (r.get("memory") or {}).get("peak_rss_bytes")
        out.append(f"| {r.get('arm')} | {r.get('context_tokens')} | "
                   f"{_fmt((r.get('latency') or {}).get('median'), 5)} | "
                   f"{_fmt(r.get('tokens_per_second'), 2)} | {mem} |")
    out.append("")

    out.append("### Incremental decoding (per generated token)\n")
    out.append("| arm | context | median s/token | samples | tok/s | cache bytes at prompt |")
    out.append("|---|---|---|---|---|---|")
    for r in d.get("incremental_decode", []):
        per = r.get("per_token_seconds") or {}
        out.append(f"| {r.get('arm')} | {r.get('context_tokens')} | "
                   f"{_fmt(per.get('median'), 6)} | {per.get('samples')} | "
                   f"{_fmt(r.get('tokens_per_second'), 2)} | "
                   f"{r.get('cache_bytes_at_prompt')} |")
    out.append("")

    out.append("### Sublayer-1 scaling (per-token time and state size)\n")
    out.append("| arm | context | median s/token | state bytes/layer | state bytes all layers |")
    out.append("|---|---|---|---|---|")
    for r in d.get("sublayer_scaling", []):
        per = r.get("per_token_seconds") or {}
        out.append(f"| {r.get('arm')} | {r.get('context_tokens')} | "
                   f"{_fmt(per.get('median'), 6)} | {r.get('state_bytes_per_layer')} | "
                   f"{r.get('state_bytes_all_layers')} |")
    out.append("")

    out.append("### Weight traffic\n")
    out.append("| arm | params | parameter bytes (fp32) | training-step bytes read | decode-step bytes read |")
    out.append("|---|---|---|---|---|")
    wt = d.get("weight_traffic") or {}
    for arm, r in wt.items():
        out.append(f"| {arm} | {r.get('n_params')} | {r.get('parameter_bytes_fp32')} | "
                   f"{r.get('training_step_parameter_bytes_read')} | "
                   f"{r.get('decode_step_parameter_bytes_read')} |")
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
    return "\n".join(out)


def render_summary(path: Path) -> str:
    """The pre-registered statistics block (protocol §3.2) from the summary."""
    d = json.loads(path.read_text(encoding="utf-8"))
    out: list[str] = []
    cond = d.get("conditions") or {}
    out.append("### Conditions of the comparison\n")
    out.append("| field | value |\n|---|---|")
    for key in ("arms", "seeds", "steps", "tokens_per_step", "tokens_processed_per_run",
                "nominal_epochs_of_train_region", "train_seed", "eval_every",
                "eval_batches", "corpus_id", "tokenizer_sha256", "device", "torch",
                "python", "commit"):
        if key in cond:
            out.append(f"| {key} | {cond[key]} |")
    out.append("")

    for endpoint in ("final", "last4"):
        block = (d.get("endpoints") or {}).get(endpoint)
        if not block:
            continue
        out.append(f"### Endpoint: {block.get('definition')}\n")
        out.append(f"Sign convention: {block.get('sign_convention')}\n")
        out.append("| seed | HE | attention-32h | d = HE − attention |\n|---|---|---|---|")
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
        out.append(f"| 95% CI for d̄ | [{_fmt(ci[0], 6)}, {_fmt(ci[1], 6)}] |")
        out.append(f"| CI excludes 0 | {block.get('ci95_excludes_zero')} |")
        out.append(f"| MDD at 80% power | {_fmt(block.get('mdd_80_power'), 6)} |")
        out.append(f"| perplexity ratio exp(d̄) | {_fmt(block.get('perplexity_ratio'), 6)} |")
        pr = block.get("perplexity_ratio_ci95") or [None, None]
        out.append(f"| perplexity ratio, 95% CI | [{_fmt(pr[0], 6)}, {_fmt(pr[1], 6)}] |")
        out.append(f"| verdict | {block.get('verdict')} |")
        needed = block.get("seeds_needed_for_80_power_at_observed_effect")
        if needed is not None:
            out.append(f"| seeds needed for 80% power at the observed effect | {needed} |")
        out.append("")

    boot = d.get("eval_set_bootstrap") or {}
    per_seed = boot.get("per_seed") or {}
    if per_seed:
        out.append("### Eval-set bootstrap (1,000 resamples of the 512 eval windows)\n")
        out.append("| seed | bootstrap SE, HE | bootstrap SE, attention-32h | "
                   "bootstrap SE, paired difference |\n|---|---|---|---|")
        for s, b in sorted(per_seed.items()):
            out.append(f"| {s} | {_fmt(b.get('bootstrap_se_arm_a'), 6)} | "
                       f"{_fmt(b.get('bootstrap_se_arm_b'), 6)} | "
                       f"{_fmt(b.get('bootstrap_se_paired_difference'), 6)} |")
        out.append("")
        out.append(f"_{boot.get('note')}_\n")

    sens = d.get("book_aligned_sensitivity") or {}
    if sens:
        out.append("### Book-aligned sensitivity analysis (post-hoc, 3 unseen books)\n")
        out.append(f"_{sens.get('definition')}_\n")
        out.append("| seed | HE | attention-32h | d = HE − attention |\n|---|---|---|---|")
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
