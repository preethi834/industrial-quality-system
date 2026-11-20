# Step 5b — scientist agent: plan and notes

Branch `step-5b` (from `step-4`), 2026-11-20. Blueprint Rev 2 §2.2 (`scientist` component),
§2.5 P1-D, §5 Step 5 (gate). Step 5a (simulator) is done; this is the scientist half.

Rules for this step (user): `nice -n 19`, changed-file tests only, no Docker, no push, no
downloads, no paid APIs, no `uv.lock` changes; stop and ask if any of these is needed.

## Plan

**Method.** One test template per taxonomy cause (the `test_template` names already in
`configs/taxonomy/causes.yaml`). A template = fixed, parameterised SQL on the read-only
DuckDB store over `observed/` → a pre-registered test → one-sided p-value, effect size,
evidence plot. Templates that scan several targets (machines, batches, shifts) report the
Bonferroni-adjusted minimum over targets, so each template's p-value is valid on its own.

| Cause | Template | Data and test |
| --- | --- | --- |
| Tool wear | `defect_rate_by_tool_wear_level` | per machine, cuts with wear ≥ 0.8 × limit vs below: reject rate, Fisher exact (greater) |
| Spindle overload | `spindle_load_defective_vs_good` | per machine, spindle load of rejected vs passed parts (affected period): Mann–Whitney U (greater) |
| Insufficient cooling | `temperature_delta_defective_vs_good` | per machine, temperature − ambient, rejected vs passed: MWU (greater) |
| Drive fault | `power_command_residual_defective_vs_good` | per machine, \|actual/commanded − 1\|, rejected vs passed: MWU (greater) |
| Fixture misalignment | `station_defect_rate_change_point` | per machine, measured dimension before (reference) vs after the alarm-window start (fixed split, no search): MWU two-sided on \|dim − nominal\| (greater) |
| Bad batch | `defect_rate_by_batch` | per batch (≥ 30 parts), batch vs other parts: Fisher (greater) |
| Wrong parameters | `defect_rate_by_parameter_set` | per machine, parts cut with a parameter set unseen in the reference period or speed outside the validated window vs standard: Fisher (greater) |
| Sensor drift | `sensor_reference_residual_trend` | master-part gauge checks over time: OLS slope t-test (two-sided) |
| Ambient | `ambient_temperature_defective_vs_good` | ambient at the cut, rejected vs passed: MWU (greater) |
| Handling error | `defect_rate_by_shift` | per shift code, shift vs other shifts: Fisher (greater) on dimension rejects |

**Verdicts.** Hypotheses tested for a case are corrected together with Benjamini–Hochberg
(statsmodels, already locked) at q = 0.05: *supported* if adjusted p ≤ 0.05, *inconclusive*
if the template lacks data (minimum group sizes not met), else *rejected* ("no evidence in
the data", not proof of absence). Every result carries the SQL text and parameters, the
statistic, raw and adjusted p, effect size, group sizes and a plot.

**Evidence plots** are plotly-compatible figure dictionaries (JSON); the dashboard
(plotly is its dependency) renders them. No plotting library is added to diagnosis.

**Agent.** New node `test_hypotheses` after `score_causes`, testing the top-3 causes.
The scientist does **not** change the probabilities or the conformal set: the step-4
calibration is frozen for the scorer as calibrated, and feeding test outcomes into the
probabilities would void it. Its output is evidence plus a "data-supported cause" (the
highest-ranked supported hypothesis), shown next to the set in D4 and in the API.

**Files.** `diagnosis/scientist/{templates.py, tests.py, runner.py, plots.py}`,
`scripts/{check_scientist_null.py, evaluate_scientist.py}`, agent node, 8D/API fields,
tests.

**Gates (§5 Step 5).**
1. Determinism and access tests pass (5a simulator tests + scientist store cannot read
   `ground_truth/`).
2. **Null check:** each template's false-positive rate over **1,000 fault-free simulations**
   ≤ 0.05 + 3 SE (SE = √(0.05·0.95/1000) ≈ 0.0069 → ≤ 0.0707). Pseudo alarm window from
   hour 36 (reference period: first 12 h), same as a real case.
3. **Evaluation family** `sci_eval_v1` (50 scenarios per fault, standard magnitudes, new
   seed) generated and its manifest hash **committed before the first evaluation run**.

**Recorded.** Root-cause accuracy per fault type (agent top-1; scientist "data-supported"
pick; templates alone); share of correct causes supported and of wrong causes rejected
(all 10 templates per case, BH over 10; and agent mode, top-3); template power per effect
size (magnitude grid per fault, 20 runs per level).

**Not in this step / needs approval later:** a `diagnosis.hypothesis_tests` table
(blueprint §2.4) would be migration 0004; evidence is persisted inside the 8D report
content until that is approved. Dashboard evidence view and production-monitoring page
belong to the dashboard step.

## Sub-steps

| Sub-step | Content |
| --- | --- |
| 5b-0 | this plan |
| 5b-1 | templates + plots + synthetic null/effect unit tests |
| 5b-2 | runner (BH, verdicts) + agent node + 8D/API evidence |
| 5b-3 | gate 2 null check (1,000 sims) + power per effect size |
| 5b-4 | evaluation family manifest committed (gate 3) |
| 5b-5 | evaluation run once + results |

## Results (2026-11-20)

> **Simulator results, not real-plant results.** The quality model was designed so that
> each injected fault is observable; these numbers show the method works on it.

### Gate status

| Gate (§5 Step 5) | Status |
| --- | --- |
| 1. Determinism and access tests pass | **PASSED**: `packages/simulator/tests/test_export.py` (identical file hashes per seed, DuckDB allow-list cannot read `ground_truth/`), `test_rng_and_alignment.py`, `test_simulator_profile.py::test_process_store_cannot_read_ground_truth`, AST leak guard (32 passed) |
| 2. Null data: FP rate per template over 1,000 simulations ≤ 0.05 + 3 SE (0.0707) | **PASSED** for all 10 templates (table below) |
| 3. Evaluation family never used in development; manifest hash committed before the first evaluation run | **PASSED**: `sci_eval_v1` manifest committed in `aa6152c`, evaluation run afterwards (the script checks the hash) |
| Universal gate (CI on main, docs, tag) | Pending merge |

### Gate 2: 1,000 fault-free simulations (`reports/diagnosis/scientist_null_power.json`)

72 h runs, seeds 900000–900999, pseudo alarm window (reference = first 12 h, affected
from hour 36). Inconclusive = not positive; the rate among conclusive runs is shown too.

| Template | FP rate (95 % CP) | Inconclusive | FP among conclusive |
| --- | --- | --- | --- |
| spindle_load_defective_vs_good | 0.053 [0.040, 0.069] | 0 | 0.053 |
| sensor_reference_residual_trend | 0.053 [0.040, 0.069] | 0 | 0.053 |
| temperature_delta_defective_vs_good | 0.045 [0.033, 0.060] | 0 | 0.045 |
| station_defect_rate_change_point | 0.027 [0.018, 0.039] | 0 | 0.027 |
| power_command_residual_defective_vs_good | 0.021 [0.013, 0.032] | 270 | 0.029 |
| defect_rate_by_batch | 0.001 [0.000, 0.006] | 0 | 0.001 (intersection-union is conservative) |
| defect_rate_by_shift | 0.000 | 992 | 0.000 (n = 8) |
| defect_rate_by_tool_wear_level | 0.000 | 1000 | n/a |
| defect_rate_by_parameter_set | 0.000 | 1000 | n/a |
| ambient_temperature_defective_vs_good | 0.000 | 1000 | n/a |

**The null check is vacuous for four templates**: their exposed group (wear ≥ 80 % of the
limit, changed parameter set, hall out of spec, and mostly enough dimension rejects per
shift) does not exist in fault-free data. Their validity rests on the Fisher exact test,
and `test_scientist_checks.py` adds synthetic null tests where the groups exist
(Bonferroni over 3 targets: 2,000 draws; intersection-union: 3,000 draws): FP ≤ α + 3 SE.

Run time: 22 min 17 s wall clock per run of the script (null + power), `nice -n 19`,
2 workers. The script was run twice: the first power run showed weak fixture power
(0.30–0.65 at standard magnitudes), traced to a design bug (deviation from a pooled
median, which moves towards the faulty values when most parts are affected). The
template now uses the nominal size from `run.json`; null results were identical in
both runs, fixture power rose to 1.00 from 0.0125 mm. The evaluation family was not
touched before the fix.

### Power per effect size (20 runs per level; power = own template p ≤ 0.05, among runs with a case)

| Fault | Magnitudes (floor → top of standard range) | Alarm rate | Power given a case |
| --- | --- | --- | --- |
| tool wear (× rate) | 1.2, 1.78, 2.35, 2.93, 3.5 | 0.05, 0.10, 0.95, 1, 1 | 0, 0, 1, 1, 1 |
| spindle overload (× load) | 1.1, 1.25, 1.4, 1.55, 1.7 | 0, 0.20, 0.65, 1, 1 | –, 1, 1, 1, 1 |
| coolant loss (fraction) | 0.1, 0.2, 0.3, 0.4, 0.5 | 0, 0.35, 1, 1, 1 | –, 0.71, 0.95, 1, 1 |
| drive deviation | 0.05, 0.11, 0.18, 0.24, 0.3 | 0.15, 0.40, 0.90, 1, 1 | 0.33, 1, 1, 1, 1 |
| fixture offset (mm) | 0.005, 0.0125, 0.02, 0.0275, 0.035 | 0.05, 0.50, 1, 1, 1 | 0, 1, 1, 1, 1 |
| bad batch (HB) | 20, 30, 40, 50, 60 | 0, 0.20, 0.80, 0.95, 1 | –, 0, 0.69, 0.89, 0.90 |
| wrong parameters (speed) | 0.15, 0.2, 0.25, 0.3, 0.35 | 0.95, 0.90, 1, 1, 0.90 | 1, 0.94, 1, 1, 1 |
| gauge drift (mm/h) | 0.0002 … 0.0012 | 0.10, 0.95, 1, 1, 1 | 1, 1, 1, 1, 1 |
| ambient (°C) | 3, 4.5, 6, 7.5, 9 | 0.05, 0.90, 1, 1, 1 | 0, 1, 1, 1, 1 |
| handling (probability) | 0.02, 0.09, 0.16, 0.23, 0.3 | 0.20, 1, 1, 1, 1 | 0.50, 1, 1, 1, 1 |

Small cells (a few cases at the weakest levels) are noisy; the full table is in the report.

### Evaluation on `sci_eval_v1` (run once; `reports/diagnosis/scientist_sci_eval_v1.json`)

600 scenarios (60 per fault, standard magnitudes) → 577 cases; simulate 251 s, evaluate
466 s, 12 min wall clock. Agent mode uses the calibrated v2 ranker (library = v2 train,
8 causes): drive fault and handling error are **new problems** for the agent.

All-causes mode (every template per case, BH over 10), 95 % Wilson:

| Metric | Result |
| --- | --- |
| Correct cause supported | **0.957** [0.937, 0.970] (552/577); never inconclusive |
| Wrong causes rejected | **0.459** [0.445, 0.472] (2382/5193) |
| Wrong causes supported | 0.137 [0.127, 0.146] (709/5193) |
| Wrong causes inconclusive | 0.405 (2102/5193) |

| Fault | n | Correct supported | Wrong rejected | Wrong supported | Agent top-1 | Scientist pick correct | Templates alone |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tool wear | 57 | 1.000 | 0.374 | 0.255 | 1.000 | 1.000 | 0.070 |
| ambient | 60 | 1.000 | 0.339 | 0.294 | 1.000 | 1.000 | 1.000 |
| bad batch | 51 | 0.706 | 0.399 | 0.118 | 0.980 | 1.000 | 0.196 |
| coolant loss | 58 | 0.879 | 0.425 | 0.096 | 1.000 | 1.000 | 0.810 |
| drive fault (new) | 60 | 1.000 | 0.548 | 0.007 | 0.000 | 0.000 | 1.000 |
| fixture | 60 | 1.000 | 0.594 | 0.043 | 1.000 | 1.000 | 1.000 |
| handling (new) | 60 | 1.000 | 0.541 | 0.100 | 0.000 | 0.000 | 1.000 |
| wrong parameters | 59 | 1.000 | 0.623 | 0.021 | 1.000 | 1.000 | 1.000 |
| gauge drift | 60 | 1.000 | 0.326 | 0.302 | 1.000 | 1.000 | 0.000 |
| spindle overload | 52 | 0.942 | 0.397 | 0.128 | 0.962 | 0.959 | 0.538 |
| **all** | 577 | 0.957 | 0.459 | 0.137 | 0.787 | 0.922 (474 picks) | 0.672 |

Agent mode (top 3 tested): the scientist names a data-supported cause in 82 % of cases
(474/577); it is correct in 92 % of them. On the 457 cases with a known cause the true
cause was among the top 3 every time and supported 439 times; the combined answer
(scientist pick, else agent top-1) equals the agent's top-1 accuracy (454/457).

Reading the results:
- **The data check confirms the right cause reliably** (95.7 %), weakest for bad batches
  near the standard range's low end (0.706) and coolant loss (0.879).
- **It does not single out the cause on its own.** Wrong causes are rejected only 46 % of
  the time and supported 14 %: faults have real physical side effects in the simulator
  (a worn tool raises load and temperature; a hot hall makes high-load cuts hotter; gauge
  drift moves measured dimensions on every machine). "Supported" means "the data show
  this pattern", not "this is the cause". Picking the template with the smallest p
  (templates alone) is right only 67 % of the time, and fails for tool wear (7 %) and
  gauge drift (0 %), where a side-effect template has an even smaller p.
- **On new problems the check can mislead**: for drive and handling cases the true cause
  is not in the agent's top 3, and the scientist then supports a wrong candidate
  (pick correct 0 %), although the right template, if asked, would be supported 100 %.
  The agent's novelty flag must stay visible next to any "data-supported" label, and a
  natural extension is to test all 10 causes when the case is flagged novel.
- Within known causes the scientist adds evidence, not accuracy (top-1 is already
  454/457). Its value is the audit trail (SQL, test, p, figure) for the engineer and
  the 8D report.

### Report field changed after the evaluation

The report first repeated the family SHA-256, which the secret scan flags. The script now
writes `family_manifest` (path of the committed manifest, which holds the hash) instead,
and the existing report got the same one-field change; the hash was checked to be equal
before it was removed. No result changed and the evaluation was not rerun.

### Code moved after the evaluation

`scientist/checks.py` → `evaluation/scientist_checks.py`: the AST leak guard (gate-1
evidence run) flagged that it imported `iqs_diagnosis.evaluation` from the agent path. No
behaviour change; the reports were produced by identical code.

### Proposed ADR-008 (number may need renumbering at merge)

**ADR-008 — Scientist agent: one pre-registered template per cause, BH per case,
evidence only (step 5b).**
1. One template per taxonomy cause, fixed SQL on the locked DuckDB store, pre-registered
   one-sided test; Bonferroni over targets inside a template; BH (q = 0.05) over the
   hypotheses of one case.
2. Verdicts supported / rejected / inconclusive; "rejected" means no evidence, not absence.
3. The scientist does not change probabilities or conformal sets (calibration stays valid);
   it adds evidence and a data-supported cause.
4. Evidence figures are plotly-compatible JSON; no plotting dependency in diagnosis.
5. Evidence is stored inside the 8D report content until a `diagnosis.hypothesis_tests`
   table (migration 0004) is approved.
