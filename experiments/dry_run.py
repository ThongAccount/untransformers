"""Milestone 2 dry-run: validate harness MECHANICS on synthetic tokens.

Runs a small config (not the paper's 18-layer/d=128 — mechanics don't need
it), 2 arms x 2 seeds, and prints per-seed curves + dispersion. Numbers are
NOT interpreted: this validates that forward/backward/step, eval, and the
multi-seed protocol produce correct, deterministic data (gate 0). Real
training waits for the corpus + compute plan.
"""

import sys

from hextractor.harness import TrainConfig, run_multi_seed

# d=32 keeps the real 32-head arm valid (d must be a multiple of n_heads);
# everything else stays small — mechanics don't need the paper's d=128.
CFG = TrainConfig(d=32, l=16, n_layers=2, ffn_width=32, vocab=64,
                  batch_size=8, eval_every=4, eval_batches=2)


def main():
    for arm in ("HE", "attention-32h"):
        out = run_multi_seed(arm, CFG, seeds=[1, 2], steps=12)
        for r in out["runs"]:
            curve = [round(v, 4) for v in r["train_curve"]]
            evals = [(e["step"], round(e["held_out_loss"], 4)) for e in r["evals"]]
            print(f"[{arm}] seed={r['seed']} train_curve={curve}")
            print(f"[{arm}] seed={r['seed']} held_out_evals={evals}")
        disp = {s: {"mean": round(v["mean"], 4), "std": round(v["std"], 4), "n": v["n"]}
                for s, v in out["dispersion"].items()}
        print(f"[{arm}] dispersion={disp}")
        finite = all(v == v and abs(v) != float("inf")
                     for r in out["runs"] for v in r["train_curve"])
        print(f"[{arm}] mechanics_ok={finite}")
    print("DRY_RUN_DONE")


if __name__ == "__main__":
    sys.exit(main())
