# Milestone 3 — the scope decision

**Question.** What should happen to this work after the measured round: keep
it as an execution-only research artefact, broaden it into something that
could support a larger claim, or stop?

**Decision:** _filled in below once the measured artefacts and the gates are in
(§4)._

**Who decides.** The reviewer (Researcher, Job B) has the final word on this
call; the user owns anything that would be published outside this repository
(handoff H3 in `GATES.md`).

---

## 1. What this round actually established

The evidence is in `results/` and the gate ledger is in
`docs/milestone-3/EVIDENCE.md`:

* **Quality (Milestone 2)** — a paired, three-seed comparison of the HE
  Extractor sublayer against a 32-head self-attention sublayer under one
  protocol, on the pinned substitute corpus, at 3.333% of the paper's token
  budget, with the statistics pre-registered in
  `docs/milestone-3/PROTOCOL.md` §3.2 and the disclosures D-A, D-B and D-C
  attached. Under the pre-registered rule the comparison is **resolved at
  n = 3** — the 95% CI for the paired difference excludes 0 — so this round
  settles the quality question *within this protocol*, and not beyond it. The
  pre-registered book-aligned sensitivity analysis was **not run** (no GPU was
  available; deviation D-27), so the round makes no sensitivity claim and the
  primary endpoint stands alone. The write-up is
  `docs/milestone-2/QUALITY.md`.
* **Systems (Milestone 3)** — the same three implementations timed on one
  accelerator with the optimized SDPA baseline included in every block, an
  A/A run-order null as the drift floor, the numerics settings pinned and
  recorded, the first decode token separated from the steady state, and the
  per-token state traffic given as analytic formulas. The write-up is
  `docs/milestone-3/SYSTEMS.md`.
* **Reproduction** — the corpus, the tokenizer, the token array, the harness,
  the runners, the gates and the tests are in the repository, and the whole
  pipeline is re-runnable from the manifest alone.

## 2. The three options

### Option A — execution-only

Keep the repository as an execution-only research artefact: the replication,
the harness, the measured artefacts, the gates and the write-ups. No claim
beyond "this is what we measured, under these conditions". No publication
outside the repository, no runtime change, no product surface.

*What it buys:* every number in the repository is backed by an artefact and a
gate; the work is reproducible by a third party from the manifest; the
disclosures stay attached to the numbers.

*What it costs:* the round leaves the open questions open. In particular the
quality comparison is a **within-corpus, within-protocol** comparison at 3.333%
of the paper's token budget, on a substitute corpus, with three unseen books.

### Option B — broaden

Invest in what this round exposed, so that the next round can support a larger
claim. The concrete list is in §5 (the declared limitations): a data-order
seed sweep, a fused/optimised HE implementation, a preallocated attention KV
cache, a tokenizer fitted on the train region only, a larger corpus with a
book-aligned held-out set (≈30 unseen books instead of 3), and more seeds.

*What it buys:* the two limitations that currently cap the quality claim (the
book count and the missing data-order variance) are both addressable, and the
kernel asymmetry that currently caps the systems claim is addressable.

*What it costs:* every one of these items changes the measurement, so the
next round is a new set of runs with a new provenance, not a continuation of
this one. The existing artefacts would remain valid as what they are — a
completed, gated round — but they would not be the basis of the larger claim.

### Option C — stop

Stop the work here. Keep the repository as it is, with the artefacts, the
gates and the write-ups, and invest nothing further.

*What it buys:* the cheapest outcome; nothing is left half-done, because the
gates in `GATES.md` are the definition of "not half-done" for this round.

*What it costs:* the questions the round raised stay unanswered, and the
artefacts stay unreviewed outside the repository.

## 3. The criteria the decision is made against

1. **Is the quality comparison resolved at n = 3?** The pre-registered rule
   (`PROTOCOL.md` §3.2): the comparison is resolved iff the 95% CI for the
   paired difference excludes 0. If it is not resolved, the write-up reports
   "not resolved at n = 3", with the MDD; a larger claim is then not
   available from this round.
2. **Are the systems differences resolvable above the run-order drift
   floor?** The rule was fixed **before** the systems numbers existed (the
   reviewer's pre-registration, 2026-10-24 03:58 UTC):

   * the **floor** for a block and a context is the **max−min spread across
     the A/A null's repetitions** — the same arm measured at the start and at
     the end of the palindromic order **at that context** — taken as the
     **largest** such spread over the arms at that context (the conservative
     choice). The null is same-context by construction: the prefill and decode
     blocks carry one row per context per slot, so an arm-level comparison
     across the block would measure the growth of the latency with the context
     instead of the run-order drift (defect D-25, fixed before the systems leg
     ran);
   * an arm-to-arm difference is **resolvable** iff `|difference|` **exceeds**
     that floor. Below the floor it is **"not resolvable", whatever its sign**;
   * the floor is printed next to every arm-to-arm difference, so the reader
     never has to take the verdict on trust.

   Exceeding the floor is **necessary, not sufficient**, for a claim: it says
   the difference is bigger than the clock drift this harness can resolve,
   not that the difference means what one would like it to mean (the kernel
   caveat travels with every wall-clock row).
3. **Does anything here support a claim about the paper's own numbers?** No —
   and nothing in this repository may say otherwise: the corpus is a
   substitute, the metric differs, and the budget is 3.333% of the paper's.
4. **Is the round reproducible and self-consistent?** The gates, the
   artefacts and the evidence ledger must agree with each other.
5. **What would each option cost in credibility?** An option that would
   require the write-up to soften a disclosure is not acceptable at any
   price.

## 4. The decision and its rationale

**Decision: Option A — execution-only.**

This repository stays an execution-only research artefact: the replication, the harness, the measured artefacts, the gates and the write-ups. No claim beyond "this is what we measured, under these conditions". No publication outside the repository, no runtime change, no product surface.

**Rationale against the criteria in §3.**

1. *Is the quality comparison resolved at n = 3?* Yes. The 95% CI for the paired difference excludes 0 on both endpoints (primary d̄ = −0.103862, CI [−0.114544, −0.093180]; secondary d̄ = −0.108946, CI [−0.111585, −0.106307]), so the pre-registered rule (§3.2) is satisfied and the comparison is resolved at n = 3 within this protocol.

2. *Are the systems differences resolvable above the run-order drift floor?* Yes, where the table says resolvable. Every arm-to-arm difference carries its per-context floor, and the reviewer's per-(arm, context) recomputation agrees with the artefact's floors (abs_diff = 0.0). G6 and G7 are MET.

3. *Does anything here support a claim about the paper's own numbers?* No, and nothing in this repository may say so: the corpus is a substitute, the metric differs, and the budget is 3.333% of the paper's (disclosures D-C and D-22; criterion 3).

4. *Is the round reproducible and self-consistent?* Yes. The gates, the artefacts and the evidence ledger agree (G2, G4, G4a, G4b, G6, G7, G8c MET), and the reviewer's independent re-computation is `REVIEW_RECOMPUTE_OK: 162 checks`.

5. *What would each option cost in credibility?* Broadening to Option B would require the write-up to soften disclosures — in particular the data-order variance lower bound (L-01) and the not-run book-aligned sensitivity (D-27, L-05) — which criterion 5 rules out at any price. Stopping (Option C) would discard a real gated result that the round actually produced. Option A keeps the result with its disclosures attached.

**Why REVIEW.md and this section are separate.** The decision is recorded here, in SCOPE.md, while the reviewer's independent verdict lives in REVIEW.md, so the decision does not self-certify and each document can be checked against the other.

**Falsifiers (§6).** A not-resolved quality comparison, systems differences below the drift floor, or a gate/evidence failure would revisit the whole round before any scope call. None of those conditions obtained.

**Open items.** The book-aligned sensitivity analysis was not run (L-05, D-27) and the data-order variance remains a lower bound (L-01); both stay declared limitations, not open bugs.

## 5. What the round left open (the declared limitations)

These are the items of `PROTOCOL.md` §8.D and the protocol review's action
list that this round did **not** do. They are limitations of the round, not
open bugs, and each of them is a concrete, scoped piece of work.

They are numbered **L-01…L-05** on purpose: the `D-` prefix belongs to the
deviation ledger (`docs/milestone-1/DEVIATIONS.md`), and calling these D-16…
D-20 would have made one number mean two different things in two documents.
The last column names the protocol item each limitation comes from.

| # | limitation | why it is out of budget here | what it would change | protocol item |
|---|---|---|---|---|
| L-01 | Data-order seed sweep (every seed currently shares one batch sequence, so `s_d` is a lower bound on run-to-run variance and the MDD is optimistic) | it changes the training run, so it breaks the provenance of the runs in flight | the uncertainty of the quality comparison becomes a full run-to-run variance rather than a lower bound | §8.D item 16 / F11 |
| L-02 | Fused/optimised HE implementation | a new kernel is a new engineering project | the systems wall-clock rows would compare two implementations of comparable maturity | §8.D item 17 / F3 |
| L-03 | Preallocated attention KV cache | the current attention implementation reallocates the cache with `torch.cat` on every token | the attention decode latency would stop carrying an implementation artefact | §8.D item 18 / F5 |
| L-04 | Tokenizer fitted on the train region only | re-tokenising changes the corpus_id and every measurement taken on it | the held-out metric would stop carrying the tokenizer leak (D-C / deviation D-22) | §8.D item 19 / F12 |
| L-05 | The full 317-book shelf and/or a book-aligned held-out set (≈30 unseen books) | it changes the corpus, and therefore the whole run | the unseen-book evidence would rest on ~30 books instead of 3, and book-level variance would become measurable | §8.D item 20 |

On L-02, the wording matters and is the reviewer's: the paper's speed claim is
hedged and rests on shorter critical paths and analytic complexity, not on
kernel choice — which is exactly why a fused-vs-eager comparison in this
harness cannot speak to it.

## 6. What would change this decision (the falsifiers)

* A **resolved** quality comparison (95% CI excluding 0) *and* systems
  differences that exceed the drift floor under the rule in §3.2 would make
  Option B attractive: the round would have produced something worth
  extending.
* A **not resolved** quality comparison, or systems differences below the
  drift floor, would make Option B premature: the next round would need to
  fix the measurement (more seeds, the data-order sweep, the fused kernel)
  before it can say anything more.
* A **failure of the gates or of the evidence ledger** — an artefact that does
  not reproduce, a gate that cannot fail, a number in a document that is not
  in the artefact — would trigger a revisit of the whole round before any
  scope call.
* The **user's decision** on publication (H3) is outside this repository and
  outside this document.

## 7. The evidence this decision rests on

* `results/quality_gpu_full.json` — the merged six runs (the primary
  artefact of the quality comparison).
* `results/quality_summary.json` — the pre-registered statistics, the
  eval-set bootstrap and the book-aligned sensitivity analysis.
* `results/eval_windows.json` — the per-window losses of the eval set and the
  book-aligned sweep.
* `results/systems.json` — the systems comparison.
* `results/checkpoints/MANIFEST.json` — the digests of the checkpoints the
  measurements were taken from.
* `docs/milestone-2/QUALITY.md`, `docs/milestone-3/SYSTEMS.md` — the
  write-ups.
* `docs/milestone-3/EVIDENCE.md` — the gate ledger, with the command, exit
  code and matched token of every gate.
* `docs/milestone-3/PROTOCOL.md` — the pre-registration the round was judged
  against.
