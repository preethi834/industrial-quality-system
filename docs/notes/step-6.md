# Step 6 — cost-aware decisions (P2 signature move B): plan

Status: **done locally** (2026-11-22, branch `step-6`). Blueprint §5 Step 6, §2.5 P2-C.
This plan was written on 2026-11-20, before step 2. What was built follows ADR-011, which
supersedes this note where they differ: no score export (step 2's `scores.json` is used); the
guaranteed (constrained) policy is the served one; the Platt fit uses sup_train ∪ sup_val only
(not the calib images); figures are hand-written SVG (matplotlib is not a direct dependency).
Results: EXPERIMENTS EXP-6-01.

Goal: for every inspected part choose **accept / reject / human review** by minimising
expected cost in euros, and report the expected cost per 10,000 parts against a fixed
threshold, the share sent to review, and how the result changes with the cost
assumptions. Every euro value is an **assumption**, never presented as measured.

## Where Step 6 starts from

- Step 6 depends on steps 1–2 (vision). Their state today:
  - **Step 1** is on branch `step-1` (not merged; the official PatchCore models exist for
    some categories, and the training queue is still running).
  - **Step 2** (conformal threshold, MVTec AD 2, VisA) has not started.
- `step-1` already has what Step 6 needs: split manifests with `calib` (good only) and
  `sup_train / sup_val / sup_test` (labelled good and defect), the `deploy` protocol
  (calibration images never seen by the detector) and the inspection route with a binary
  decision against a provisional threshold (the largest calibration score).
- Per-image scores are **not saved** today (only aggregate `metrics.json`). Step 6 needs
  them, so 6a adds a score export. This costs one inference pass per category, not a
  retrain.

## Sub-steps

**6a — Score export.** `scripts/export_scores.py` writes one row per image to
`reports/vision/scores/<dataset>/<category>/<protocol>.parquet`. Each row has the
manifest path, sha256, split, sup_split, label, defect_type and score. It uses the
saved `deploy` models only and runs inference only, at `nice -n 19`, after the training
queue has finished.

**6b — Probability calibration** (`vision/decision/calibrate_proba.py`).
- Map scores to p(defect) with Platt scaling (2 parameters, own NumPy/SciPy code). The
  comparison is isotonic regression (scikit-learn, already a vision dependency).
- Fit per category on `sup_train ∪ sup_val`, plus the good-only `calib` images. Never use
  `sup_test`.
- **Prior-shift correction:** the MVTec test sets contain far more defects than a plant
  does. After fitting, p is shifted to an assumed production defect rate π by an odds
  correction. π is a stated assumption.
- Report calibration quality on `sup_test`: Brier score, ECE and a reliability curve,
  with bootstrap intervals. Isotonic is likely to overfit at these sample sizes (tens of
  defects per category); this will be reported, not hidden.

**6c — Decision policy and costs** (`vision/decision/{costs.py,policy.py}`, `configs/costs/example_costs.yaml`).
- `CostConfig` (pydantic, frozen) holds C_false_reject, C_escape, C_review, the human
  error rates in review (ε_miss, ε_false_reject), π and currency. Every value carries a
  `note: assumption`.
- Expected costs per part:
  - accept = p·C_escape;
  - reject = (1−p)·C_false_reject;
  - review = C_review + ε_miss·p·C_escape + ε_fr·(1−p)·C_false_reject.
- Because all three are linear in p, the rule reduces to two cut-offs on p. These are
  computed in closed form, and review is dropped automatically when it never wins.
- Pure domain code: no I/O.

**6d — Evaluation** (`scripts/evaluate_costs.py`, output `reports/vision/costs_<dataset>.json`).
- Evaluate on `sup_test` per category, plus a pooled result.
- Expected cost per 10,000 parts = 10,000·[π·E(cost | defect) + (1−π)·E(cost | good)].
  It uses the class-conditional action rates measured on `sup_test`, so the test-set
  defect share does not distort the result. Report the review share as well.
- Baselines, both with accept/reject only:
  - the provisional step-1 threshold;
  - the step-2 conformal threshold (α = 2 %) once it exists;
  - a cost-optimal single threshold on p, without review.
- Intervals come from a stratified bootstrap over `sup_test` images.
- Sensitivity:
  - a grid over C_escape / C_false_reject and C_review / C_false_reject (log scale)
    showing the cost saving and the review share;
  - a separate sweep over π.
- The plot is a heat map plus a line chart (matplotlib, already in the lock). It goes into
  `reports/vision/` and the EXPERIMENTS.md entry EXP-6-01, with the assumptions written
  next to every number.

**6e — Serving and UI.**
- `POST /v1/inspections` additionally returns p_defect, the action (accept / reject /
  review), the expected cost of each action and the cost config id. It only does this
  when a calibrator and a cost config are loaded; otherwise those fields are absent,
  never faked.
- The dashboard inspection page shows the decision and the three expected costs.

## Tests (changed files only, `nice -n 19`)

- **Policy on hand-computed cases**, including the boundary at each cut-off and the case
  where review never wins.
- **Invariants** (hypothesis property tests):
  - raising C_review never increases the review share;
  - raising C_escape never increases the accept share;
  - raising C_false_reject never increases the reject share;
  - with ε = 0 and C_review = 0, nothing is accepted or rejected unless p is 0 or 1.
- **Calibrators:**
  - p stays monotone in the score;
  - the prior-shift correction is exact on a synthetic example;
  - `sup_test` is never used to fit (a leak test on the manifest split).
- **Evaluation arithmetic** on a tiny synthetic score table.

## Done when (blueprint)

The cost table and the sensitivity plot are in EXPERIMENTS.md, with the assumptions next
to every number. The headline result is reported per category and pooled, with intervals.
The results are on MVTec images with **assumed** costs and an **assumed** defect rate,
not plant savings.

## Needs your approval / decision

1. **Branch base.** I recommend `step-6` from `step-1`, because the vision code lives only
   there; the diagnosis branches are not needed. You said earlier not to merge `step-1`,
   so I'm asking before I branch from it.
2. **Order versus Step 2.** The blueprint says Step 6 depends on steps 1–2. My proposal:
   build 6a–6c and the 6e code now, and use the provisional threshold as the baseline.
   Then add the conformal-threshold baseline once Step 2 exists, and only then write the
   final EXP-6-01 entry. The alternative is to do Step 2 first.
3. **Compute.**
   - The score export (6a) and the evaluation need the CPU after the training queue has
     finished (inference only, roughly minutes per category).
   - Which categories? Everything with a `deploy` model at that time, or a fixed subset.
4. **Euro assumptions** (blueprint checklist item; example values for your approval):

   | Assumption | Proposed value |
   |---|---|
   | C_false_reject (a good part scrapped) | €5 |
   | C_escape (a defective part shipped: complaint, sorting, freight) | €200 |
   | C_review (one human look) | €0.50 |
   | ε_miss (reviewer misses a defect) | 0.05 |
   | ε_false_reject (reviewer rejects a good part) | 0.02 |
   | π (production defect rate) | 1 % |

   All values are labelled "assumption, not measured". The sensitivity grid covers ratios
   two orders of magnitude around these values.
5. **`vision.cost_configs` table** (blueprint §2.4). I recommend YAML only in Step 6 and
   adding the table in Step 7 after the branches are merged. Migration numbers already
   diverge: `step-1` ends at 0001 and the diagnosis branches at 0004.
6. **No new dependencies expected.** NumPy, SciPy, scikit-learn, matplotlib and
   hypothesis are already in `uv.lock`. If that turns out to be wrong, I will stop and ask.

## Timing risk

The deadline is Mon 2026-11-23. Steps 1 and 2 are both incomplete, and steps 7 and 9 are
still to come. Step 6's code is small (about a day), but its numbers wait on the training
queue and on Step 2.
