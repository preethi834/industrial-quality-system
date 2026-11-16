# Diagnosis labelling guide

Every labelled case gets exactly **one primary cause** from
`configs/taxonomy/causes.yaml` (taxonomy v0.1.0). There is no OTHER or UNKNOWN
label: a case that fits no cause is left unlabelled and treated as a novelty case.

Sections for NHTSA recalls and complaints are added in step 3.

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
