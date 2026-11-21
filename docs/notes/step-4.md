# Step 4 — conformal cause sets: notes for merging

Branch `step-4` (from `step-3`, worktree `~/code/iqs-step5`), 2026-11-20. Shared docs are not
edited on this branch; this file and `docs/notes/step-3.md` collect what goes into them.
Reference: blueprint Revision 2, §2.8 (conformal maths), §5 Step 4 (gate).

> **Merged 2026-11-21.** The text meant for the shared docs is now in CHANGELOG, KNOWN_ISSUES,
> TECHNICAL_DECISIONS (ADR-007), TESTING, EXPERIMENTS, ARCHITECTURE and IMPLEMENTATION_PLAN.
> This file stays as the detailed record.

> **Simulator coverage is not real-world coverage.** All numbers below are on
> `dx_corpus_v2` (simulated line, injected faults). The guarantee is marginal, under
> exchangeability with simulator calibration cases; it does not transfer to NHTSA or
> plant cases, which need their own calibration run (gate 1).

## Gate status

| Gate (§5 Step 4) | Status |
| --- | --- |
| **1. Coverage and set size on the real test split (LAC, APS, α = 0.1, Clopper–Pearson 95 %)** | **OPEN.** Needs the hand-labelled NHTSA set (step 3f, labelling guide NHTSA section, about 2–3 h of user labelling). No real cases exist yet. |
| 2. Statistical validity on ≥ 500 simulator cases: mean coverage over R = 100 resplits ≥ 0.9 − 3 SE | **PASSED** for LAC (0.8956 ≥ 0.8934; **margin only 0.002**) and APS (0.9602 ≥ 0.8985), pool 684 |
| 3. New-problem coverage reported as "stress test, no guarantee" | **DONE** (below) |
| Universal gate (CI green on main, docs, tag) | Pending merge; docs text below |

Decisions taken with the user (2026-11-20): no MAPIE (cross-check against hand-computed
reference sets in `test_conformal.py`); migration 0003 approved; gates 2 and 3 closed now,
gate 1 left open.

## What was built

| Sub-step | Commit | Content |
| --- | --- | --- |
| 4a | `18a72aa` | `conformal/split.py` (k = ⌈(n+1)(1−α)⌉, error when k > n; (α, δ) threshold from Beta(k, n+1−k)); `conformal/scores.py` (LAC, non-randomised APS); Clopper–Pearson |
| 4b | `8be9a82` | `scoring/combine.py` (refit, extended grid); `conformal/calibration.py` (hash split, disjointness check, frozen `CalibrationRun`); `scripts/calibrate_causes.py`; `configs/diagnosis/calibration/cal_dx_corpus_v2_dev_a010.json`; `configs/diagnosis/ranker_params_v2.json`; migration 0003 (`diagnosis.calibration_runs`, `case_causes.calibration_run_id`) |
| 4c | `e03fb2b` | agent node `conformal_set`; 8D D4 lists the set (empty LAC set → "escalate"); `/cause-set` returns `kind: "conformal"` with α, q̂, run ID and the stated guarantee; capability flag |
| 4d | this commit | `evaluation/conformal_eval.py`, `scripts/evaluate_conformal.py`, report `reports/diagnosis/conformal_dx_corpus_v2_test.json` |

## Data, disjointness, refit

- **Library:** v2 train (306 cases, 8 causes). **Scorer fit:** v2 dev *fit* half (289 cases).
  **Calibration:** v2 dev *cal* half, cases with a library primary cause (n = 230). The
  halves come from SHA-256 of `"dx_corpus_v2/dev" + case_id`. **Test:** v2 test, evaluated
  once. `check_disjoint` raises if a calibration case is a library or fit case; the
  evaluation script refuses to run if a test case was used for any of them; a test
  checks the committed run (calibration ∩ fit = ∅).
- **Vote smoothing refit** on the fit half with the grid extended from
  {0.001 … 0.3} to {1e-7 … 0.3} (14 values): optimum **alpha = 0.001, n_neighbours = 10**,
  now in the interior of the grid (a test asserts this), NLL 0.294. The step-3 edge
  value was therefore not an artefact of the grid's lower bound.
- Refit novelty threshold 0.553 (fit half: false novelty 0.046, detection 0.786; before:
  0.708 from v1 dev).
- Thresholds (α = 0.1, δ = 0.1, n = 230): LAC q̂ = 0.4992 (k = 208), LAC (α, δ) q̂ = 0.6026
  (k = 214), APS q̂ = 0.9911 (k = 208 and 214: same value, inside a tie block).

## Ties make coverage conservative

Vote probabilities take few distinct values: of the 230 calibration scores only 72 are
distinct, and 158 are the same value (cases where every neighbour votes for the true
cause: p = (1 + α)/(1 + 10α) = 1.001/1.01 ≈ 0.9911 with α = 0.001, LAC score 0.0089). With ties:
- every cause whose score equals q̂ is included, so P(true cause ∈ set) ≥ 1 − α still holds;
- the upper bound 1 − α + 1/(n+1) and the Beta(k, n+1−k) law of realised coverage do
  **not** hold; coverage can sit well above target (APS: 0.989 at a 0.9 target);
- the (α, δ) threshold can coincide with the marginal one (APS: both k = 208 and k = 214
  land on 0.9911).
`test_conformal.py::test_ties_make_coverage_conservative` demonstrates this on a
three-valued score.

## Results on the v2 test split (α = 0.1; Clopper–Pearson 95 %)

582 test cases: 454 with a library primary cause (349 single-fault, 105 two-fault),
128 new problems (held-out primary cause).

| Variant | Coverage, known (n = 454) | Mean set size | Empty sets | Sizes |
| --- | --- | --- | --- | --- |
| LAC | **0.894** [0.862, 0.921] | 0.96 | 0.035 [0.020, 0.057] | 0: 16, 1: 438 |
| LAC (α, δ = 0.1) | 0.914 [0.884, 0.938] | 1.01 | 0.009 [0.002, 0.022] | 0: 4, 1: 442, 2: 8 |
| APS | **0.989** [0.974, 0.996] | 1.56 | 0 [0, 0.008] | 1: 305, 2: 80, 3: 38, 4: 25, 5: 5, 6: 1 |
| APS (α, δ = 0.1) | 0.989 [0.974, 0.996] | 1.56 | 0 | same as APS |

| Variant | Single-fault coverage (n = 349) | Two-fault: primary covered (n = 105) | Two-fault: every injected cause in set | Two-fault size / empty |
| --- | --- | --- | --- | --- |
| LAC | 0.991 [0.975, 0.998] | **0.571** [0.471, 0.668] | 0.000 [0, 0.035] | 0.85 / 0.152 |
| LAC (α, δ) | 0.997 | 0.638 [0.539, 0.730] | 0.019 | 0.99 / 0.038 |
| APS | 0.997 | **0.962** [0.905, 0.990] | 0.562 [0.462, 0.659] | 2.82 / 0 |

**New problems (stress test, no guarantee; n = 128):** coverage 0.000 [0, 0.028] for every
variant, as expected: the true cause has no library cases, so it only gets smoothing mass.
What matters is the signal to the engineer: **LAC returns an empty set for 57.8 %**
[48.8 %, 66.5 %] of new problems (shown as "escalate"), against 3.5 % of known cases;
APS returns large sets instead (mean 4.46 causes vs 1.56). The separate novelty flag
(step 3) remains the main new-problem signal.

**Gate 2 (R = 100 resplits, pool = 230 calibration + 454 test known cases = 684,
n_calibration = 230):** LAC mean coverage 0.8956 (sd 0.0219, SE 0.0022) ≥ 0.8934 → pass;
mean size 0.96. APS 0.9602 (sd 0.0049) ≥ 0.8985 → pass; mean size 1.57. The resplits
overlap, so the SE is optimistic; LAC passes with a margin of only 0.002.

Reading the results:
- **Marginal, not conditional.** LAC meets 90 % overall by being near-perfect on
  single-fault cases (0.991) and poor on two-fault cases (0.571). That is exactly what
  §2.8 warns about: no guarantee per case type. A Mondrian (group-wise) calibration
  by case type would need a group label that is observable at prediction time (the agent
  does not know that a case has two faults), so it is not a fix here.
- **LAC sets have at most one cause** at this α: q̂ = 0.4992 means p ≥ 0.5008, and only
  one cause can exceed 0.5. LAC therefore behaves as "top-1 or escalate". For two-fault
  cases APS is the useful set (primary covered 96 %, both causes 56 %, 2.8 causes).
- **Decision (user, 2026-11-20):** the dashboard shows **APS** sets, with the LAC-empty
  flag as an "uncertain / possible new problem" signal. `IQS_DIAGNOSIS__CAUSE_SET_VARIANT`
  now defaults to `aps`; every cause set also carries `lac_empty` (LAC set at the same α
  is empty), and the 8D D4 text states it.
- **The LAC gate-2 margin is only 0.002** (0.8956 vs 0.8934). It passes, but a slightly
  different pool or seed could fail it; LAC coverage on test (0.894 [0.862, 0.921]) is
  also just under target. Treat LAC's 90 % as borderline on this corpus; APS is far above.

## Tests (all at `nice -n 19`, changed files only; no Docker)

- `test_conformal.py` (14): index rule (n = 8 errors, n = 9 works; blueprint examples),
  10,000-trial coverage band + KS test against Beta(k, n+1−k), (α, δ) bound, ties,
  hand-computed LAC/APS reference sets, score/set consistency, Clopper–Pearson.
- `test_calibration_run.py` (4): hash split, disjointness errors, thresholds equal order
  statistics, (α, δ) needs n ≥ 22, committed run consistent and leak-free, smoothing not
  on the grid edge.
- `test_conformal_eval.py` (5): set metrics by hand, resplit check passes on calibrated
  synthetic cases, bad input rejected.
- Agent / API: conformal node, empty set → escalate, no calibration → no set,
  `/cause-set` kind conformal.
- Integration (written, **not run**): calibration-run rows and cause-set persistence in
  `tests/integration/test_diagnosis_persistence.py`; run with `make test-integration`.

## ADR

Moved to `docs/TECHNICAL_DECISIONS.md` as **ADR-007** at the merge into `main` (2026-11-21).
