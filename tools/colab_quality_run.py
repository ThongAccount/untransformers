#!/usr/bin/env python3
"""Runs INSIDE a Colab session: verify the suite, then run the quality sweep.

Clones the public repo (the corpus snapshot comes with the clone — no upload
of 50 MB), runs the full test suite (T14 included: the corpus checksums are
verified in the session that will train on them), then runs the Milestone-2
quality sweep and writes `/content/results.json` + `/content/run.log`.

Configuration comes from environment variables set by the launcher:
  STEPS, SEEDS, ARMS, LABEL, OUT, EVAL_EVERY, EVAL_BATCHES, TRAIN_SEED,
  DEVICE, SAVE_CHECKPOINTS
"""

import json
import os
import platform
import subprocess
import sys
import time

REPO_URL = "https://github.com/ThongAccount/untransformers.git"
BRANCH = os.environ.get("BRANCH", "research/milestone-1-extractor")
DEST = "/content/repo"
LOG = "/content/run.log"


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def main() -> int:
    t_start = time.time()
    open(LOG, "w").close()
    log(f"[env] python={platform.python_version()} platform={platform.platform()}")

    r = subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, DEST],
                       text=True, capture_output=True)
    log(f"[clone] rc={r.returncode} {r.stdout[-500:]} {r.stderr[-800:]}")
    if r.returncode != 0:
        return 2
    os.chdir(DEST)
    log(f"[clone] HEAD={subprocess.run(['git','rev-parse','HEAD'], capture_output=True, text=True).stdout.strip()}")

    missing = []
    for mod, pkg in (("pytest", "pytest"), ("tokenizers", "tokenizers"), ("numpy", "numpy")):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            missing.append(pkg)
    if missing:
        # Install only what is absent: upgrading numpy underneath torch breaks it.
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing],
                           text=True, capture_output=True)
        log(f"[pip] installed={missing} rc={r.returncode} {r.stderr[-400:]}")
    else:
        log("[pip] all deps present; nothing installed")

    try:
        import torch
        log(f"[env] torch={torch.__version__} cuda={torch.version.cuda} "
            f"cuda_available={torch.cuda.is_available()} "
            f"threads={torch.get_num_threads()}")
        if torch.cuda.is_available():
            log(f"[env] gpu={torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        log(f"[env] torch probe failed: {exc}")

    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
                       text=True, capture_output=True)
    log(f"[suite] pytest_rc={r.returncode}")
    log(r.stdout[-3000:])
    if r.stderr:
        log(r.stderr[-1000:])

    steps = os.environ.get("STEPS", "200")
    seeds = os.environ.get("SEEDS", "1,2").split(",")
    arms = os.environ.get("ARMS", "HE,attention-32h").split(",")
    label = os.environ.get("LABEL", "pilot")
    out = os.environ.get("OUT", "/content/results.json")
    eval_every = os.environ.get("EVAL_EVERY", "50")
    eval_batches = os.environ.get("EVAL_BATCHES", "8")
    train_seed = os.environ.get("TRAIN_SEED", "0")
    device = os.environ.get("DEVICE", "auto")
    save_ckpt = os.environ.get("SAVE_CHECKPOINTS", "0") == "1"

    cmd = [sys.executable, "experiments/run_quality.py",
           "--arms", *arms, "--seeds", *seeds, "--steps", steps,
           "--eval-every", eval_every, "--eval-batches", eval_batches,
           "--train-seed", train_seed, "--label", label, "--out", out,
           "--device", device]
    if save_ckpt:
        cmd.append("--save-checkpoints")

    t0 = time.time()
    r = subprocess.run(cmd, text=True, capture_output=True)
    log(f"[sweep] rc={r.returncode} wall={time.time()-t0:.1f}s")
    log(r.stdout[-6000:])
    if r.stderr:
        log(r.stderr[-2000:])
    log(f"[done] total wall {time.time()-t_start:.1f}s")
    return 0 if r.returncode == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
