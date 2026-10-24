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
| git | `6448165` (the commit each row was executed at, unless stated) |
| date | 2026-10-24 (UTC) |

The Colab runs carry their own conditions in the artefacts themselves
(`environment` block of each results JSON) and in the session logs committed
alongside them.

---

## Gate ledger

| gate | status | command | exit | matched token | wall |
|---|---|---|---|---|---|
| G1 | MET | `/home/node/.venv/bin/python -m pytest tests/test_t14_data_pipeline.py -q` | 0 | `12 passed` | 3.33 s |
| G2 | MET | `/home/node/.venv/bin/python -m pytest -q` | 0 | `153 passed, 1 skipped` | 16.53 s |
| G3 | MET | `… check_quality_results.py --file results/quality_pilot.json --label pilot --arms HE attention-32h --seeds 1 2` | 0 | `PILOT_OK` | 2026-10-24 02:24 UTC (pilot artefact from session `he-pilot-cpu2`) |
| G3a | MET | `… verify_pairing_evidence.py --results results/quality_pilot.json --out results/pairing_evidence_pilot.json --merge-into results/quality_pilot.json` | 0 | `verified=True` (both claims) | 2026-10-24 02:24 UTC |
| G4 | MET | `… check_quality_results.py --file results/quality_gpu_full.json --label full --arms HE attention-32h --seeds 1 2 3` | 0 | `FULL_OK` | 0.06 s |
| G4a | MET | `… verify_pairing_evidence.py --results results/quality_gpu_full.json --checkpoint-dir results/checkpoints --out results/pairing_evidence_gpu_full.json --merge-into results/quality_gpu_full.json` | 0 | `verified=True` (both claims) | 2.74 s |
| G4b | PENDING | `… summarise_quality.py --file results/quality_gpu_full.json --eval-windows results/eval_windows.json --out results/quality_summary.json` | — | — | the pre-registered statistics + bootstrap + sensitivity; the eval-window re-scoring is queued on a T4 (see below) |
| G5 | MET | `/home/node/.venv/bin/python -m pytest tests/test_m3_incremental.py -q` | 0 | `11 passed` | 0.13 s |
| G6 | MET | `… check_systems_results.py --file results/systems.json` | 0 | `SYSTEMS_OK` | 0.08 s |
| G7 | MET | `… check_systems_results.py --file results/systems.json --require-paired-quality` | 0 | `PAIRED_QUALITY_OK` + `SYSTEMS_OK` | 0.10 s |
| G8 | PENDING | `… check_docs.py --systems docs/milestone-3/SYSTEMS.md --scope docs/milestone-3/SCOPE.md` | — | — | the scope decision is the reviewer's (H2); the checker now *fails* until it exists |
| G8b | PENDING | `… check_docs.py --systems … --scope … --quality docs/milestone-2/QUALITY.md` | — | — | after the tables are rendered |
| G8c | PENDING | `… check_doc_numbers.py --doc docs/milestone-2/QUALITY.md --results results/quality_gpu_full.json --eval-windows results/eval_windows.json --manifest data/manifest.json` | — | — | the prose numbers against the artefacts |
| G9 | PENDING | `test "$(git rev-parse HEAD)" = "$(git rev-parse origin/research/milestone-1-extractor)" && test -z "$(git status --porcelain)" && echo PUSHED` | — | — | final push (the command was repaired — see below) |

### G2 note — the suite has grown

`88 passed, 1 skipped` was the state after the Milestone-3 code landed
(`a6f9fde`). The three gate-checker self-tests (`tests/test_verify_scripts.py`)
took it to `91 passed, 1 skipped`. The skip is T15, which requires CUDA; it
runs on the GPU session instead (see below).

At the end of the write-up pass the suite is **153 passed, 1 skipped**: the
gate-hygiene self-tests below, the renderer self-tests and the doc-number
checker's self-tests are part of that count. The skip is T15, which needs
CUDA; it runs inside the GPU session instead.

---

## The gates that could not fail (the reviewer's Job-B pass, 2026-10-24)

The reviewer read the gates and the write-ups while the GPU sweep was still
running, and found four defects in the *gate machinery* itself. All four were
verified by running them, and all four were fixed before any systems or
quality number existed — so nothing here was adjusted after seeing a result.

| # | defect | how it was verified | the fix | the test that now pins it |
|---|---|---|---|---|
| 1 | **G8 passed with the scope decision still empty.** The placeholder list was `TODO/TBD/FIXME/XXX/PLACEHOLDER/Lorem ipsum`, so the write-up's own "_filled in below_" was invisible, and the section markers ("decision", "rationale", "evidence", "falsifier") were all satisfied by the surrounding prose | `check_docs.py` printed `DOCS_OK` with exit 0 while `SCOPE.md` §4 was a 67-character stub | `tools/verify/check_docs.py` now treats "filled in" / "to be written" as placeholders, requires the decision section to name one of the three options, to carry a rationale, and to be at least 400 characters | `tests/test_verify_scripts.py::test_docs_checker_refuses_a_scope_document_that_decides_nothing` |
| 2 | **G9 could never be MET.** The command ended in `\|\| true`, so it exited 0 with a dirty tree and exited 0 with an unpushed commit, and it printed nothing — under the rule "exits zero AND the EXPECT matches", it could not pass | run with a dirty tree (exit 0) and with an unpushed commit (exit 0) | `GATES.md` now uses `test … = … && test -z "$(git status --porcelain)" && echo PUSHED` | the command itself (it now prints the token and fails in both directions) |
| 3 | **The limitation IDs collided.** `SCOPE.md` §5 called its limitations D-16…D-20, and the deviation ledger already used those numbers for other things — D-19 meant two different things in two documents | read both ledgers side by side | the scope limitations are **L-01…L-05**, each naming the `PROTOCOL.md` §8.D item it comes from; the cross-references in `QUALITY.md` and `SYSTEMS.md` follow | the doc checker (the ids are now unique across the repository) |
| 4 | **The drift-floor rule was not operationalised.** "Above the drift floor" had no definition, so it could have been defined after seeing the numbers | — | the reviewer pre-registered it and it is implemented in the renderer: the floor for a block and a context is the **max−min spread across the A/A null's repetitions**, the **largest** such spread over the arms at that context; a difference is **resolvable** iff `\|difference\|` exceeds it, and below it it is **not resolvable, whatever its sign**. The rule and the floor are printed next to every arm-to-arm difference | `tests/test_render_results_tables.py::test_every_arm_to_arm_difference_carries_the_drift_floor_and_a_verdict` |

A second reviewer pass — the reviewer writing its own independent
re-computation, which it then ran against the pilot artefacts — found four
more defects of the same kind. Each was fixed **before** the GPU sweep landed
and before the systems leg ran, so no measured number was ever produced by the
broken code.

| # | defect | how it was verified | the fix | the test that now pins it |
|---|---|---|---|---|
| 5 | **The renderer read a key the merge never writes.** `render_results_tables.py` read `paired.differences_arm_a_minus_b`, while `merge_quality_runs.py` writes `differences_arm_a_minus_arm_b` — so after the fill, QUALITY.md's paired-difference column would have been **empty on every row**, and the gate that reads the correct key would have passed | the reviewer's tool printed the empty column; the renderer's own output showed it | the renderer reads the key the merge writes (and falls back to the other spelling) | `tests/test_render_results_tables.py::test_the_paired_difference_column_is_never_empty` |
| 6 | **The 80% t-quantile table was wrong for 29 of its 30 rows.** It listed 1.282 at df = 1 where the exact quantile is 1.376, and 0.466 at df = 30 where it is 0.838 — the "seeds needed for 80% power" loop used that column and understated the requirement by 10–24% (12 seeds instead of 15 at an effect of 0.8·s_d) | the reviewer's tool inverted the Student-t CDF numerically (validated against the Cauchy case df = 1, whose quantile is `tan(0.3π)`) and compared | the builder now **computes** the quantiles from the Student-t CDF (no table), and `verify_protocol_constants()` fails the run if they stop reproducing the protocol's stated constants (t₀.₉₇₅,₂ = 4.3027, t₀.₈₀,₂ = 1.0607, CI 2.484·s_d, MDD 3.097·s_d) | `tests/test_quality_tools.py::test_the_t_quantiles_reproduce_the_protocols_constants`, `::test_the_seeds_needed_loop_uses_the_computed_80_percent_quantile` |
| 7 | **The A/A null was not a same-context null.** `run_systems.py` compared an arm's first and last slot across the whole block, but the prefill and decode blocks carry one row per context per slot — so the "drift floor" was mostly the growth of the latency with the context | on the CPU rehearsal artefact: the prefill floor was 0.052005 s (a context gap) and is 0.003183 s as a same-context null; the decode floor was 0.024263 s and is 0.001187 s | the null is computed **per (arm, context)**; the artefact carries `aa_null_by_context` and `floor_by_context_seconds`; the renderer judges each difference against the floor of its own context; the G6 checker recomputes the null from the slot rows and fails if the artefact's null is not the same-context one | `tests/test_systems_null.py`, `tests/test_verify_scripts.py::test_systems_checker_rejects_a_cross_context_null` |
| 8 | **The per-eval-point dispersion table printed the population SD under a bare `std` header** — the exact statistic protocol §7 rule 3 forbids quoting as the dispersion (the paired table below it was labelled correctly, so the two tables disagreed) | read the two headers side by side | the column is now `std (population, ddof=0 — descriptive only)` | the renderer self-test (no cell may be `None`/`nan`) and the paired table's label |

Two more defects were found by the rehearsal (the write-up rendered from the
pilot-scale artefacts before the real ones existed):

* the quality summary wrote **NaN** — not valid JSON — when it had a single
  seed, and crashed mapping the perplexity interval through `exp()`; both are
  now null-safe, and the null renders as an em dash, never as the word "nan"
  (`e024981`, `c30fdd4`);
* a table section whose artefact was missing rendered as an **empty string**,
  which would have silently deleted a required section of a write-up while the
  doc gate still passed. The filler now refuses an empty section (`c30fdd4`).

And one more, found by the reviewer while it was writing its own independent
re-computation: the **book-aligned metric's coverage** was described as "every
token position inside each book is a target". With stride 1 the windows
overlap, so the count is a count of **token-predictions** (windows × `l`), not
of distinct tokens, and the first token of each book is never a target. The
coverage table now prints the books' token counts, the window counts, the
stride and the token-prediction counts, and says so in words.

And one more, found by the rehearsal of the write-up fill (the pilot-scale
artefacts rendered into a copy of the write-up before the real ones existed):
the **renderer could put a Python object into a committed write-up**. The
bootstrap block's note lives on the *per-seed* entries of the summary artefact,
while the renderer read it from the block — so the rendered note was the
literal `_None_`; the verdict and the "CI excludes 0" cells were interpolated
raw, so a missing value printed `None` instead of an em dash; and the
bootstrap header carried the window count and the resample count as literals,
so the header could disagree with the artefact it claims to describe. The note
is now taken from the per-seed entries (falling back to the block's
definition), a missing value is an em dash everywhere, and the header states
the artefact's own numbers (`844aaca`, pinned by
`tests/test_render_results_tables.py::test_the_bootstrap_note_is_rendered_and_never_the_word_none`).

---

## The systems leg, twice: a CUDA OOM and a re-measurement

The first GPU systems leg died **before it wrote an artefact**. The
sublayer-1 scaling block warms a cache of `T` positions and then times a
single new token at position `T-1`; the eager attention path materialises the
`[H, m, T]` score matrix for the chunk it is handed, and at `T = 8192` with 32
heads that is 8.6 GB per tensor — three of them alive at once, on a GPU with
14.56 GB. The runner died with:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.00 GiB.
GPU 0 has a total capacity of 14.56 GiB of which 6.11 GiB is free.
```

The fix is in the runner's **warm-up**, not in the measured implementation: the
cache is filled in pieces of 256 positions. The warm-up is **not timed** (the
measured call is a single new token at position `T-1`), and the cache it
leaves is bit-identical — a token's K and V do not depend on how the warm-up
was cut, and HE's rolling state keeps the last `l-1` vectors either way. Both
facts are pinned by `tests/test_systems_warmup.py`, including that the warm-up
is outside the timed region. Recorded as deviation **D-26**.

No number was produced by the broken code: the leg died before writing its
artefact, so there is no first-leg artefact to be read. The committed systems
artefact is the re-measurement, taken at the tooling commit, and the log of
**both** legs — the OOM traceback and the successful run — is committed as
`docs/milestone-3/logs/systems.txt`.

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

## 2026-10-24 — the GPU leg, three times: a pruned session, a silent bug and an OOM

Three things went wrong between the first sweep attempt and the artefacts that
count. None of them produced a measured number that anyone could misread, and
all three are recorded here because they change how the artefacts must be read.

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

### 3. The session that counts was pruned mid re-scoring

The session `he-gpu3` ran the six-run sweep, the T15 gate and both systems
legs. The first systems leg died with the CUDA OOM described above; the
re-measurement at the tooling commit `b2a3221` wrote `results/systems.json`.
The session was then used for the **evaluation-only re-scoring** — the
book-aligned stride-1 sweep and the eval-set re-scoring of the six
checkpoints — and **Colab pruned it at the four-hour limit** (≈ 06:17 UTC).

The prune cost the sweep of the three HE checkpoints that had already been
scored (≈ 820 s each): they existed only inside the session's filesystem, so
no artefact was recovered from them. Every checkpoint itself was already on
the local disk (`results/checkpoints/`, with the digests in
`results/checkpoints/MANIFEST.json`), and every other artefact of the round
had already been downloaded, so only the *scoring* had to be redone.

The CPU fallback was measured and rejected rather than assumed: on this
1-core host the book-aligned path (the exact `region_window_losses` call the
GPU leg makes — stride 1, batch 64, one thread) costs **19.8 ms per window
for the HE arm** and **65.3 ms per window for the attention arm**, i.e.
**2.23 h** and **7.37 h** per checkpoint, or ≈ 29 h for the six checkpoints
serially on the single core this container has. The eval-set half of the same
re-scoring (512 windows per checkpoint) costs 10.2 s per HE checkpoint and
33.0 s per attention checkpoint. The sweep therefore waits for a T4 — where
one checkpoint takes ≈ 820 s — rather than running on the CPU; Colab's T4
pool was saturated while this was written (the allocation endpoint answers
`Service Unavailable`), and the leg sits in a persistent retry loop.

The re-scoring leg was rebuilt so that the next prune cannot cost it again:

* **one checkpoint per `colab exec`**, and the artefact is downloaded
  immediately after that exec returns — the granularity that was missing
  before, where two arms ran inside one exec and the download happened only
  after it returned;
* **a checkpoint already on the local disk is skipped**, so a leg that dies
  half way resumes where it stopped instead of starting over;
* the measurement is unchanged: the same script, the same flags, the same
  window sets (protocol §2.3). Only the unit of work per exec differs.

The cross-check the re-scoring carries is part of the artefact: the
recomputed eval-set mean must reproduce the final held-out loss the run
reported. On the CPU rehearsal of the eval-set half (a rehearsal — the
committed artefact is the T4 one) the six recomputed means reproduce the
reported losses to within 3 × 10⁻⁷, which is the float32 CPU-vs-GPU
difference and not a modelling difference.
