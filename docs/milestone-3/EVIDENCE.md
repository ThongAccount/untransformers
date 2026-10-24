# EVIDENCE — Milestone 2 runs → Milestone 3 systems → scope decision

Gate ledger for `research/milestone-1-extractor` (repo `ThongAccount/untransformers`).
The gate definitions live in [`GATES.md`](../../GATES.md); this file records what
actually ran, with its conditions, so the write-up can cite evidence instead of
memory.

`/data/skills/unlazy` ships only `SKILL.md` in this installation (no
`templates/`, `references/` or `scripts/`), so the automatic gate-checker is
unavailable. Every gate below was therefore executed by hand with the command,
the exit code and the matched token recorded. Nothing here is a re-run of
somebody else's number: each row is the output of the command in that row.

## Conditions of the evidence run

| item | value |
|---|---|
| host | Modal container, 1 CPU core, 676 GB RAM |
| working dir | `/workspace/untransformers` |
| python | `/home/node/.venv/bin/python` (CPython 3.12.15) |
| torch | 2.14.1+cpu (local host) |
| threads | 1 (`torch.set_num_threads(1)` in the drivers) |
| git | `f207b7a` (the commit each row was executed at, unless stated) |
| date | 2026-10-23 (UTC) |

The Colab runs carry their own conditions in the artefacts themselves
(`environment` block of each results JSON) and in the session logs committed
alongside them.

---

## Gate ledger

| gate | status | command | exit | matched token | wall |
|---|---|---|---|---|---|
| G1 | MET | `/home/node/.venv/bin/python -m pytest tests/test_t14_data_pipeline.py -q` | 0 | `12 passed` | 3.33 s |
| G2 | MET | `/home/node/.venv/bin/python -m pytest -q` | 0 | `91 passed, 1 skipped` | 5.61 s |
| G3 | MET | `… check_quality_results.py --file results/quality_pilot.json --label pilot --arms HE attention-32h --seeds 1 2` | 0 | `PILOT_OK` | 2026-10-24 02:24 UTC (pilot artefact from session `he-pilot-cpu2`) |
| G3a | MET | `… verify_pairing_evidence.py --results results/quality_pilot.json --out results/pairing_evidence_pilot.json --merge-into results/quality_pilot.json` | 0 | `verified=True` (both claims) | 2026-10-24 02:24 UTC |
| G4 | IN FLIGHT | `… check_quality_results.py --file results/quality_gpu_full.json --label full --arms HE attention-32h --seeds 1 2 3` | — | — | GPU sweep running (Colab session `he-gpu3`, Tesla T4, commit `5b296ca`) |
| G4a | PENDING | `… verify_pairing_evidence.py --results results/quality_gpu_full.json --checkpoint-dir results/checkpoints --out results/pairing_evidence_gpu_full.json --merge-into results/quality_gpu_full.json` | — | — | runs on the merged artefact |
| G4b | PENDING | `… summarise_quality.py --file results/quality_gpu_full.json --eval-windows results/eval_windows.json --out results/quality_summary.json` | — | — | the pre-registered statistics + bootstrap + sensitivity |
| G5 | MET | `/home/node/.venv/bin/python -m pytest tests/test_m3_incremental.py -q` | 0 | `11 passed` | 0.13 s |
| G6 | PENDING | `… check_systems_results.py --file results/systems.json` | — | — | queued in the same T4 session, after the sweep |
| G7 | PENDING | `… check_systems_results.py --file results/systems.json --require-paired-quality` | — | — | paired checkpoints come from the sweep's `SAVE_CHECKPOINTS=1` |
| G8 | PENDING | `… check_docs.py --systems docs/milestone-3/SYSTEMS.md --scope docs/milestone-3/SCOPE.md` | — | — | written after the numbers land |
| G9 | PENDING | `git rev-parse HEAD` == `git rev-parse origin/research/milestone-1-extractor` and clean tree | — | — | final push |

### G2 note — the suite has grown

`88 passed, 1 skipped` was the state after the Milestone-3 code landed
(`a6f9fde`). The three gate-checker self-tests (`tests/test_verify_scripts.py`)
took it to `91 passed, 1 skipped`. The skip is T15, which requires CUDA; it
runs on the GPU session instead (see below).

---

## T15 — the GPU smoke test had never run before

Until a T4 slot opened on 2026-10-23 at 15:17:21 UTC, every Colab session
this project used was CPU-only, so `tests/test_t15_gpu_smoke.py` had never
executed once. On the first GPU it ran, it failed — and both defects were in
the test, not in the model:

1. **dtype mismatch.** The layer was constructed in fp32 while the oracle
   input was fp64, and `copy_` casts into the destination dtype, so the
   forward raised
   `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: double != float`.
   Fixed by building an fp64 layer for the equation gate and a separate fp32
   layer for the dtype the training actually uses.
2. **off-by-one causality boundary.** The assertion compared rows `[:65]`
   after perturbing row 64, but row 64 is itself allowed to change — the
   rows that must stay bit-identical are `[:64]`. Fixed, and a negative
   control (`row 64` MUST change) was added so the check cannot pass
   vacuously.

Measured on the local host at d=l=t=128 before the fix:

| comparison | max abs diff | max rel diff |
|---|---|---|
| fp64 layer vs fp64 oracle | 9.094947017729282e-13 | — |
| fp32 layer vs fp64 oracle | 0.0006435961917645727 | 0.0018941273901099621 |

The fp32 tolerance in the rewritten test (atol 2e-3, rtol 5e-3) follows from
those measurements; the fp64 gate keeps the tight 1e-5 absolute tolerance.

Fixed in `1754738`, pushed, and the corrected file was uploaded into the live
T4 session before the sweep reached its T15 step, so the in-session run
executes the corrected test.

## GPU session — conditions of the Milestone-2 sweep and Milestone-3 systems run

| item | value |
|---|---|
| session | `he-gpu-main` |
| hardware | Tesla T4 (free tier), Standard shape, GPU variant |
| allocated | 2026-10-23 15:17:21 UTC |
| torch | 2.11.0+cu130, CUDA 13.0, cudnn per the session |
| python | 3.13.15, Linux 6.6.122 |
| threads | 1 |
| clone | `research/milestone-1-extractor` @ `a6f9fde` (the fixed T15 and the fixed `run_systems.py` were uploaded over it afterwards) |
| sweep | `STEPS=2000 SEEDS=1,2,3 ARMS=HE,attention-32h LABEL=full EVAL_EVERY=250 EVAL_BATCHES=8 TRAIN_SEED=0 DEVICE=cuda SAVE_CHECKPOINTS=1` |
| systems | `d=128 l=128 layers=18 ffn=512 vocab=5000 batch=64 heads=32 trials=5 warmup=2 gen_tokens=16 device=cuda`, same session, same GPU |

Both legs run in the same session so the quality numbers and the systems
numbers share one hardware/software environment.

## CPU pilot — conditions

| item | value |
|---|---|
| session | `he-pilot-cpu` |
| hardware | CPU runtime, Standard shape |
| torch | 2.11.0+cpu, python 3.13.16, threads 1 |
| clone | `research/milestone-1-extractor` @ `554e537` |
| suite inside the session | `77 passed, 1 skipped in 9.49 s` |
| pilot | `STEPS=50 SEEDS=1,2 ARMS=HE,attention-32h LABEL=pilot EVAL_EVERY=25 EVAL_BATCHES=4 TRAIN_SEED=0 DEVICE=cpu` |

The pilot is a reduced-budget run (50 steps per seed): its purpose is to prove
the protocol end-to-end on the pinned corpus, not to settle the quality
question. The paper-budget comparison is the GPU sweep.

---

## 2026-10-24 — the GPU leg, twice: one pruned session and one silent bug

Two things went wrong between the first sweep attempt and the sweep that
counts. Neither produced a measured number that anyone could misread, and
both are recorded here because they change how the artefacts must be read.

### 1. The first T4 session was pruned mid-sweep

The session `he-gpu-main` (allocated 2026-10-23 15:17:21 UTC) was pruned by
Colab at 16:29 UTC, while the sweep cell was still executing. The CLI client
had been killed by a container restart at ~16:20, and the runtime followed
nine minutes later. **No artefact of that session was recovered** — not one
eval point. The CPU pilot (`he-pilot-cpu2`) survived the same restart because
its cell had already finished (it completed at 16:33 UTC and its artefacts
were downloaded).

The sweep was therefore re-designed so that a second prune cannot cost the
whole run:

* every `(arm, seed)` is an **independent invocation** of the same runner, so
  the six runs are six independently-downloadable artefacts. The split changes
  nothing numerically: `corpus_stream_factory` returns a *factory*, and
  `train_one_seed` calls it again for every seed, so each seed replays the
  same batches from the start and each arm sees the same batches in the same
  order — exactly as in one combined invocation (protocol §3.4 states the same
  fact);
* the local supervisor recreates the session and resumes if the runtime dies,
  re-uploading the runs that already finished.

### 2. `--device cuda` never reached the training loop

The re-launched sweep (`he-gpu2`, commit `700c3e9`) produced **no progress
line for 22 minutes**, while an in-session probe of the same work measured
**1.17 s/step** for the HE arm. The cause: `harness.train_one_seed` built the
model on the CPU and never moved it — the runner wrote `device=cuda` into its
environment manifest, but the training loop ran on the CPU, at **14.3 s/step**.
(The probe's numbers matched the CPU pilot's numbers bit-for-bit, which is how
the discrepancy was spotted: the "GPU" run was reproducing the CPU run.)

Fixed in `5b296ca`: the model is moved after the seeded initialisation, the
optimizer is built after the move, and the batches follow the model. The
in-session verification after the fix: **20 steps in 25.3 s (1.27 s/step)**.
The regression test `tests/test_t15_gpu_smoke.py::test_quality_training_loop_really_trains_on_the_gpu`
watches the device the forward actually sees, so the device cannot be dropped
again silently. Only throw-away probes (20 steps) were produced before the
fix; they were discarded and are not part of any result.

Recorded as deviation **D-23** in `docs/milestone-1/DEVIATIONS.md`.

### The session that counts

| item | value |
|---|---|
| session | `he-gpu3` |
| hardware | Tesla T4 (free tier), Standard shape, GPU variant |
| allocated | 2026-10-24 02:30 UTC |
| torch | 2.11.0+cu130, CUDA 13.0 |
| clone | `research/milestone-1-extractor` @ `5b296ca` |
| suite inside the session | `112 passed in 13.73 s` (the GPU tests included) |
| sweep | one invocation per `(arm, seed)`: `--steps 2000 --eval-every 250 --eval-batches 8 --train-seed 0 --device cuda --save-checkpoints` |
| measured step time | HE ≈ 1.2 s/step, attention-32h ≈ 0.37 s/step (in-session probe) |
| artefacts | `/content/runs/<arm>_seed<seed>.json`, `/content/checkpoints/<arm>_seed<seed>.pt`, `/content/sweep.log` — downloaded as they land |
