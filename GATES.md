# GATES — untransformers: Milestone 2 runs → Milestone 3 systems → scope decision

Ledger for the remaining queue on `research/milestone-1-extractor`
(owner: MLForge; reviewer: Researcher; repo: `ThongAccount/untransformers`).

Rules that stay fixed: every number carries its conditions; no result is
interpreted before it lands; a gate is met only when its command exits zero AND
its `EXPECT:` matches the combined output. The `unlazy` checker scripts are not
shipped in this skill installation (`/data/skills/unlazy` contains only
`SKILL.md`), so the gates below are executed by hand and their evidence
(command, exit code, matched token, wall time) is recorded in
`docs/milestone-3/EVIDENCE.md`.

All commands run with `CWD: /workspace/untransformers` and
`python = /home/node/.venv/bin/python`.

---

## G1 — T14 corpus pipeline is green and deterministic (MET, commit 39829be)

  CHECK: /home/node/.venv/bin/python -m pytest tests/test_t14_data_pipeline.py -q
  EXPECT: passed

## G2 — full test suite green (GPU test skipped on CPU)

  CHECK: /home/node/.venv/bin/python -m pytest -q
  EXPECT: passed,.*1 skipped

## G3 — CPU pilot landed: both arms × seeds 1,2 on the pinned corpus

  CHECK: /home/node/.venv/bin/python tools/verify/check_quality_results.py --file results/quality_pilot.json --label pilot --arms HE attention-32h --seeds 1 2
  EXPECT: PILOT_OK

## G4 — GPU session sweep landed (full multi-seed) or capacity formally abandoned

  CHECK: /home/node/.venv/bin/python tools/verify/check_quality_results.py --file results/quality_gpu_full.json --label full --arms HE attention-32h --seeds 1 2 3
  EXPECT: FULL_OK

  (If no free-tier GPU slot ever opens, this gate is recorded with
  `ABANDON: G4 <reason>` — abandonment is handoff, never success.)

## G5 — incremental decoding is exact against the full forward pass (both arms)

  CHECK: /home/node/.venv/bin/python -m pytest tests/test_m3_incremental.py -q
  EXPECT: passed

## G6 — systems comparison measured at every context length, both arms

  CHECK: /home/node/.venv/bin/python tools/verify/check_systems_results.py --file results/systems.json
  EXPECT: SYSTEMS_OK

## G7 — paired quality checkpoints carried into the systems comparison

  CHECK: /home/node/.venv/bin/python tools/verify/check_systems_results.py --file results/systems.json --require-paired-quality
  EXPECT: PAIRED_QUALITY_OK

## G8 — write-up and scope decision exist with their required sections

  CHECK: /home/node/.venv/bin/python tools/verify/check_docs.py --systems docs/milestone-3/SYSTEMS.md --scope docs/milestone-3/SCOPE.md
  EXPECT: DOCS_OK

## G9 — everything is committed and on the Hub

  CHECK: bash -lc 'test "$(git rev-parse HEAD)" = "$(git rev-parse origin/research/milestone-1-extractor)" && git status --porcelain | grep -q . && exit 1 || true'
  EXPECT: PUSHED

  (Run after the final commit; `git status --porcelain` must be empty.)

---

## Handoff (not gates — these belong to people, not commands)

- H1 — Researcher's Job A protocol memo (dispersion statistics, systems-comparison
  requirements, misleading-number risks, corpus blessing) must be read before the
  long GPU sweep is interpreted.
- H2 — Researcher's Job B review (overclaiming check + Milestone-3 write-up +
  scope decision) is the final word on the scope call.
- H3 — The user owns the decision to publish anything outside this repo.
