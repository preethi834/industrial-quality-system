# Diagnosis labelling guide

Every labelled case gets exactly **one primary cause** from
`configs/taxonomy/causes.yaml` (taxonomy v0.1.0). There is no OTHER or UNKNOWN
label: a case that fits no cause is left unlabelled and treated as a novelty case.

NHTSA recalls (step 3f) have their own section below. They add two **statuses** that
are not labels: `outside_taxonomy` (a stated cause that is not one of the ten; used as a
real new-problem case) and `not_determinable` (the text does not say why the part was
made defective; excluded and counted). NHTSA complaints are not labelled (they carry no
cause).

## Mapping AI4I 2020 rows to one primary cause

AI4I 2020 flags five failure types (TWF, HDF, PWF, OSF, RNF). Each is defined by a
physical condition, not by a root cause, so the mapping below is a **documented
interpretation**, recorded in `docs/TECHNICAL_DECISIONS.md`.

| AI4I flag | Definition in the dataset | Primary cause |
| --- | --- | --- |
| TWF | Tool wear reaches its replacement window | `MACHINE.TOOL_WEAR` |
| HDF | Air–process temperature difference too small at low rotational speed | `MACHINE.INSUFFICIENT_COOLING` |
| PWF | Operating power (torque × rotational speed) outside 3500–9000 W | `METHOD.WRONG_PROCESS_PARAMETERS` |
| OSF | Tool wear × torque above a product-type threshold | `MACHINE.SPINDLE_OVERLOAD` |
| RNF | Random failure, independent of process values | none |

Notes on two mappings:

- **PWF → `METHOD.WRONG_PROCESS_PARAMETERS`.** AI4I records only the operating
  values, with no commanded-vs-actual signal. By the taxonomy's boundary, operating
  outside the validated power window is a parameter (method) problem;
  `MACHINE.DRIVE_FAULT` requires actual power to deviate from the commanded power and
  is therefore simulator-only.
- **OSF → `MACHINE.SPINDLE_OVERLOAD` is an interpretation.** OSF is defined by the
  product tool wear × torque, so it can fire at normal torque when tool wear is high.
  Such rows are labelled `MACHINE.SPINDLE_OVERLOAD` by this rule, and the ambiguity is
  reported with the results rather than hidden.

### Rules

1. **RNF carries no cause.** A row whose only flag is RNF gets no label and is
   excluded from cause-labelled data. If RNF occurs together with another flag,
   ignore RNF.
2. **Several cause flags: use the fixed priority HDF > PWF > OSF > TWF.** HDF and PWF
   depend on conditions unrelated to tool wear, so when present they are the
   distinct, process-level cause. OSF ranks above TWF because the excess torque is
   the controllable factor; TWF is the age-based fallback.
3. **Inconsistent rows are excluded and counted.** This covers `Machine failure = 1`
   with no type flagged, and a type flagged while `Machine failure = 0`. The counts are
   measured at ingestion and recorded in `data/README.md`; they are not assumed here.
4. **Robustness check.** Every AI4I-based result is also reported with all
   multi-flag rows excluded, so the effect of rule 2 is visible.
5. **Coverage limit.** AI4I provides data for four causes only: `MACHINE.TOOL_WEAR`,
   `MACHINE.SPINDLE_OVERLOAD`, `MACHINE.INSUFFICIENT_COOLING` and
   `METHOD.WRONG_PROCESS_PARAMETERS`. The other six causes get their Bayesian-network
   probabilities and test data from the simulator only, and results state this.

## Simulator cases (step 3)

Simulator labels are never assigned by hand: they come from the run's ground truth
(`ground_truth/case_labels.json`, read only by `iqs_diagnosis.evaluation.labels`).

- **One injected fault:** its taxonomy cause is the primary cause.
- **Several injected faults: primary = the fault that adds the most defects.** For each
  fault `f`, `excess(f) = (truly defective parts touched by f) − p0 × (parts touched by f)`,
  where `p0` is the defect rate of parts touched by no fault. The fault with the largest
  excess is the primary cause; ties go to the fault listed first in the scenario. The
  other faults are recorded as secondary causes (`secondary_causes`), and the excess
  values are stored with the label so every decision can be checked.
- Rationale: it is the Pareto view an 8D team takes (fix what produces most of the
  extra defects first). It uses ground truth only, so it does not depend on what the
  agent can observe.
- Results on two-fault cases are reported separately: top-1 against the primary cause,
  top-3 containing the primary cause, and top-3 containing every injected cause.

## NHTSA recalls (step 3f)

**Source.** NHTSA ODI recalls flat file, `FLAT_RCL_POST_2010.zip` (records with a
report date from 2010 on), tab-delimited, 29 fields (`RCL.txt`). One sheet row is one
**recall campaign** (`CAMPNO`); the flat file's rows per make/model/year are merged
(`n_rows`, `model_years`, `makes`). Sheet: `eval_sets/diagnosis/nhtsa/recalls_v1.csv`,
built by `scripts/prepare_nhtsa_labelling.py`; check it any time with
`scripts/check_nhtsa_labels.py`.

**The basic problem.** A recall describes a defect of a *vehicle component* and its
consequence on the road. Our taxonomy describes *why a production line made bad parts*.
Most recalls never say the second thing (design faults, software, unspecified "may have
been manufactured incorrectly"). So most rows will be `not_determinable` or
`outside_taxonomy`, and only a few causes can appear at all (see "Expected coverage").
That is expected and is reported, not worked around.

### What you fill in (last six columns)

| Column | Content |
| --- | --- |
| `label_status` | `labelled`, `outside_taxonomy` or `not_determinable`. Empty = not done yet. |
| `cause_code` | `labelled` only: exactly one taxonomy code (primary cause). Empty otherwise. |
| `secondary_cause_codes` | `labelled` only, optional: further codes the text names, separated by `;`. |
| `evidence` | `labelled` and `outside_taxonomy`: the shortest **verbatim** quote (copy and paste) from the three summaries that justifies the decision. Case and spacing may differ; words may not. |
| `confidence` | `labelled` only: `high` (the text names the mechanism literally) or `medium` (one step of interpretation from the table below). If it would be "low", use `not_determinable`. |
| `notes` | Optional free text (e.g. why two causes were close). |

Never edit the text columns, the row order of columns, `row_id` or `campno`; the check
script rejects the sheet if they change.

### Decision, in order

1. Read `defect_summary`, then `corrective_summary` and `consequence_summary`. Use only
   this text: no outside knowledge, no other NHTSA documents (keeps labels
   reproducible).
2. **Does the text say why the defective parts were produced** (a production, assembly,
   supplier-process or inspection cause)? If not → `not_determinable`. This includes
   "may have been manufactured incorrectly", "a manufacturing defect", "a quality issue"
   without the mechanism.
3. **Is the stated cause outside production?** Design, specification, software or
   calibration of the *vehicle* (not of production equipment), labels and manuals,
   noncompliance by design → `outside_taxonomy`, with the quote as evidence.
4. **Production cause that fits one row of the mapping table** → `labelled` with that
   code. If it fits none (e.g. parts damaged in transport, wrong part shipped by a
   supplier mix-up) → `outside_taxonomy`.
5. Two production causes named → the one the text presents as the origin is primary;
   the other goes to `secondary_cause_codes`.
6. **Label the production cause, never the vehicle failure.** Brakes overheating on the
   road is not `MACHINE.INSUFFICIENT_COOLING`; a vehicle sensor drifting is not
   `MEASUREMENT.SENSOR_DRIFT` (that cause is about the plant's measuring equipment).

### Mapping table (recall wording → taxonomy cause)

| Cause | Label when the text says, for example | Confidence |
| --- | --- | --- |
| `MATERIAL.BAD_BATCH` | raw material / resin / steel "did not meet specification"; a contaminated or out-of-spec **lot** of material; incorrect hardness or composition of a lot | high |
|  | a supplier lot of components was defective, mechanism not stated (**documented interpretation**: seen from the receiving plant, an incoming bad batch) | medium |
| `METHOD.WRONG_PROCESS_PARAMETERS` | incorrect weld, cure, heat-treatment, press or torque **settings / program / process parameters** of production equipment | high |
|  | "improperly torqued / welded / crimped" without saying whether a setting or a person caused it | medium |
|  | "improperly heat-treated" without naming the setting (**decision 2026-11-21**: heat treatment is a process step, so a bad result is read as a parameter problem) | medium |
| `PERSONNEL.PART_HANDLING_ERROR` | only an **intermittent, operator- or shift-linked handling mistake** the text states as such (the taxonomy cause is "intermittent loading or handling mistakes, associated with a shift"), e.g. parts damaged during handling by operators on one shift | high |
| none → `outside_taxonomy` | **assembly errors** (**decision 2026-11-21**): a part missing, the wrong part installed, a part not fully inserted / seated, mis-routed hoses or wires, wrong position or orientation. Recalls do not say these are shift-linked handling mistakes, so they are not mapped to `PERSONNEL.PART_HANDLING_ERROR` | – |
| `MACHINE.FIXTURE_MISALIGNMENT` | misaligned fixture, jig, die or tooling **position** producing a consistent dimensional offset | high |
| `MACHINE.TOOL_WEAR` | worn tooling, die or cutting tool | high |
| `MEASUREMENT.SENSOR_DRIFT` | plant inspection or test equipment out of calibration / malfunctioning, so defective parts passed (**documented interpretation**: the taxonomy cause covers decisions based on a drifted reading) | medium |
| `MACHINE.INSUFFICIENT_COOLING`, `MACHINE.SPINDLE_OVERLOAD`, `MACHINE.DRIVE_FAULT`, `ENVIRONMENT.AMBIENT_TEMPERATURE_DEVIATION` | only if the text states it for **production equipment or the plant** (e.g. "coolant system failure on the machining line", "plant humidity"). "Insufficient cooling **time**" is a parameter → `METHOD.WRONG_PROCESS_PARAMETERS`. | high |

### Who labelled the pilot (provenance)

The pilot sheet `recalls_pilot_v1.csv` (50 rows) was labelled on 2026-11-21 by an **AI
assistant (LLM)** following this guide, with a verbatim evidence quote on
every `labelled` and `outside_taxonomy` row, and then **reviewed by the project
author**. Every row's `notes` column starts with "AI label (LLM)", and the sheet
manifest records the same (`labelling`). Consequences:

- These are AI-assisted labels with human review, not independent expert labels.
  Every result on them says so.
- A component that itself uses the same LLM must not be scored against these labels
  without stating the shared-model bias risk (the agent's LLM path is Ollama only, so
  this does not apply today).
- The optional consistency check (working order, step 4) is best done by the human
  reviewer, blind, on 20 rows.

### Expected coverage (reported with every result)

Recall texts can only support the first five or six rows of the table in practice;
tool wear, cooling, spindle overload, drive faults, ambient temperature and sensor drift
are expected to be rare or absent. NHTSA results therefore cover fewer causes than the
simulator and are reported separately per cause, with counts. `outside_taxonomy` rows
are the real-data **new-problem** cases (novelty evaluation); `not_determinable` rows are
excluded, and their share is reported.

### How the sheet is sampled (so results can be read correctly)

- Vehicle recalls (`RCLTYPECD = V`) with report date ≥ 2015-01-01 and a non-empty
  defect summary, in the approved component groups (3–5, blueprint §8). A campaign
  touching several approved groups is assigned to the first group in the list.
- Per group: a **term** stratum (defect summary matches a manufacturing word list, in
  `evaluation/nhtsa_labels.py`, e.g. *manufactur-, assembl-, weld-, supplier, torque-,
  contaminat-, improperly, out of specification*) and a smaller random **control**
  stratum from the rest. The filter only raises the share of labellable rows; the
  control stratum measures what it misses.
- Row order is shuffled and the stratum is kept out of the CSV (it is in the
  manifest), so it cannot influence the label.
- Results are reported per stratum. Statements about "NHTSA recalls in general" are
  not possible from the term stratum alone.

### Working order

1. **Pilot:** label rows `r0001`–`r0050` (a random mix), then run the check script. It
   prints the done count, the yield of `labelled` rows by stratum and any errors; the
   pilot fixes the time per row and whether the sheet size should change.
2. Label the rest; run the check script whenever you like (it never changes the file).
3. Open the CSV in LibreOffice or Excel as UTF-8, keep `campno` as text, save as CSV
   (UTF-8). Do not sort and re-save in a different column order.
4. Optional consistency check: one week later, re-label 20 random rows blind; the
   agreement is reported.
5. Splits (dev / calibration / test, plus new-problem cases) are drawn **after**
   labelling, grouped by manufacturer so near-identical texts of one maker stay in one
   split, with a fixed seed; the split manifest is committed before any evaluation.

### Licence

NHTSA data is a work of the U.S. federal government (data.gov access level "public";
no licence field; treated as public domain under 17 U.S.C. § 105, an interpretation
recorded here). Recall text in the committed sheet is cited as "NHTSA Office of Defects
Investigation, Recalls flat file", with the file's download date and SHA-256 in the
manifest.
