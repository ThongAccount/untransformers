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
| G3 | IN FLIGHT | `… check_quality_results.py --file results/quality_pilot.json --label pilot --arms HE attention-32h --seeds 1 2` | — | — | CPU pilot running (Colab session `he-pilot-cpu`) |
| G4 | IN FLIGHT | `… check_quality_results.py --file results/quality_gpu_full.json --label full --arms HE attention-32h --seeds 1 2 3` | — | — | GPU sweep running (Colab session `he-gpu-main`, Tesla T4) |
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
