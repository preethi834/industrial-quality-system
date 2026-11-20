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
