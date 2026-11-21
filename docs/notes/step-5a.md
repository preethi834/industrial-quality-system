# Step 5a — factory simulator (`iqs_sim`): notes for merging

Written on branch `step-5` (worktree `~/code/iqs-step5`), 2026-11-19, while Step 1 was being
built in parallel on `step-1`. The shared docs (README, CHANGELOG, IMPLEMENTATION_PLAN,
KNOWN_ISSUES, TECHNICAL_DECISIONS, TESTING, ARCHITECTURE) were **not** edited here, to avoid
merge conflicts. This file collects everything that should go into them; §8 lists the exact
pieces to copy.

> **Merged 2026-11-21.** The text meant for the shared docs is now in CHANGELOG, KNOWN_ISSUES,
> TECHNICAL_DECISIONS (ADR-008), TESTING, EXPERIMENTS, ARCHITECTURE and IMPLEMENTATION_PLAN.
> This file stays as the detailed record.

Scope: Step 5a only (simulator, scenarios, export, ground truth, case generator, CLI, tests).
The scientist agent (`diagnosis/scientist`, templates, BH correction, evidence plots) is
Step 5b and needs the Step 3 agent.

Reference: blueprint Revision 2 (§2.2 simulator components, §2.4 DuckDB/Parquet, §5 Step 5).
Revision 2 is on `step-1` (commit `76debd5`); this branch still holds Revision 1, so it was
read with `git show 76debd5:TECHNICAL_BLUEPRINT.md`. The Step 5 text is the same in both
revisions apart from the gate wording.

---

## 1. What was built

| Module (`packages/simulator/src/iqs_sim/`) | Role |
| --- | --- |
| `machines.py` | Line L1 layout and master data: S1 turning (M1) → S2 milling (M2A ‖ M2B) → S3 drilling (M3) → S4 inspection (Q1). Validated speed window ±10 %, tolerance 50.000 ± 0.030 mm, ambient spec 18–26 °C. |
| `rng.py` | Named PCG64 streams from `(seed, sha256(name))`. |
| `faults.py` | `FaultType` (= the 10 `simulator_fault` names), `FAULT_SPECS` (target kind, magnitude unit/range, the observable columns each fault may change), `FaultConfig` (YAML model), `ActiveFaults` (runtime lookups), `cause_code_for` (fault → taxonomy code). |
| `scenarios.py` | `Scenario` / `LineConfig` (pydantic, `extra="forbid"`), YAML load/dump, scenario families + manifest hash. |
| `sensors.py` | Pure physics: sensor values and the hidden surface-defect logit from given inputs and given noise. |
| `line.py` | SimPy model: source, part flow, machine tokens, tool changes, gauge checks, inspection; returns row lists. |
| `export.py` | Explicit Arrow schemas; writes `observed/` and `ground_truth/`; file hashes in `run.json`. |
| `cases.py` | Case generator from `observed/` only (SPC-style alarm). |
| `pipeline.py` | `run_scenario()`: simulate → export → case → `ground_truth/case_labels.json`. |
| `cli.py` + `scripts/simulate.py` | `run`, `validate`, `family`. |

Scenario files: `configs/simulator/scenarios/` — `baseline.yaml` plus one reference scenario
per fault (11 files). All share seed 7, start 2026-01-05 06:00 UTC and 72 h, so each fault
scenario's fault-free twin *is* the baseline.

No new dependencies: `simpy`, `numpy`, `pandas`, `pyarrow`, `pyyaml` were already declared in
`packages/simulator/pyproject.toml` and locked. `uv.lock` is unchanged. DuckDB is used only in
one test, via `pytest.importorskip` (it is installed through the diagnosis package).

### Run directory layout

```text
<run>/observed/      parts, sensor_readings, station_events, batches, shifts, machines (.parquet), run.json
<run>/ground_truth/  faults, part_truth, batch_truth (.parquet), scenario.json, run.json, case_labels.json
<run>/cases/         case.json        (only when monitoring raises an alarm)
```

`observed/` is what the agent may read. It contains no scenario name, description, seed or
fault field, and its schemas are identical for every scenario. The seed is withheld on purpose:
with seed + simulator, a reader could re-run the fault-free twin and diff it.

### Usage

```bash
uv run python scripts/simulate.py validate configs/simulator/scenarios/*.yaml
uv run python scripts/simulate.py run --scenario configs/simulator/scenarios/tool_wear_m1.yaml \
    --out ~/iqs-data/sim/tool_wear_m1            # must be new or empty
uv run python scripts/simulate.py family --name eval_v1 --seed 2026 --n-per-fault 50 \
    --out ~/iqs-data/sim/families/eval_v1        # writes YAMLs + manifest.json (family_sha256)
```

---

## 2. Faults (one per taxonomy leaf)

"Signals" = the observable columns the fault may change besides the inspection outcome
(`parts.surface_defect_detected`, `final_quality`, `reject_reason`). The isolation test
checks that each fault changes **exactly** these columns, only at its target, and only after
its onset.

| `simulator_fault` | Cause code | Target | Magnitude | Signals | Reference scenario |
| --- | --- | --- | --- | --- | --- |
| `accelerated_tool_wear` | MACHINE.TOOL_WEAR | cutting machine | wear-rate × (≥ 1.2) | tool_wear, vibration, spindle_load, temperature | `tool_wear_m1` (M1, ×2.5) |
| `spindle_overload` | MACHINE.SPINDLE_OVERLOAD | cutting machine | load × (≥ 1.1) | spindle_load, temperature | `spindle_overload_m2a` (×1.5) |
| `coolant_flow_loss` | MACHINE.INSUFFICIENT_COOLING | cutting machine | flow fraction lost | temperature | `coolant_loss_m3` (0.35) |
| `drive_power_deviation` | MACHINE.DRIVE_FAULT | cutting machine | relative power error | power_actual | `drive_fault_m1` (+18 %) |
| `fixture_offset` | MACHINE.FIXTURE_MISALIGNMENT | cutting machine | mm | parts.measured_dimension | `fixture_offset_m2b` (0.025 mm) |
| `bad_material_batch` | MATERIAL.BAD_BATCH | batch `Bnnnn` | true hardness Δ HB (certificate unchanged) | spindle_load, temperature (that batch's parts) | `bad_batch_b0008` (+45 HB) |
| `parameter_setpoint_change` | METHOD.WRONG_PROCESS_PARAMETERS | cutting machine | relative speed change (\|x\| ≥ 0.15, i.e. outside the ±10 % window) | parameter_set_id, setpoint_speed, power_commanded, power_actual, spindle_load, temperature | `wrong_parameters_m3` (+25 %) |
| `sensor_calibration_drift` | MEASUREMENT.SENSOR_DRIFT | gauge Q1 | mm per hour | parts.measured_dimension, station_events.value (master-part gauge checks) | `gauge_drift_q1` (0.0008 mm/h) |
| `ambient_temperature_excursion` | ENVIRONMENT.AMBIENT_TEMPERATURE_DEVIATION | `HALL` | °C added | ambient_temperature, temperature (all machines) | `ambient_excursion` (+7 °C) |
| `handling_error_on_shift` | PERSONNEL.PART_HANDLING_ERROR | shift code A/B/C | probability per part | parts.measured_dimension | `handling_error_shift_b` (0.25) |

How the taxonomy's "Excludes" pairs stay distinguishable in the data:

- tool wear vs spindle overload: wear raises `tool_wear_mm` and vibration; overload raises load
  at normal wear.
- insufficient cooling vs ambient: cooling loss raises machine temperature with
  `ambient_temperature_c` in range; an excursion raises ambient on every machine.
- drive fault vs wrong parameters: a drive fault opens a gap between `power_actual_kw` and
  `power_commanded_kw`; wrong parameters change the setpoints and `parameter_set_id`, and
  commanded/actual move together.
- fixture offset vs handling error: fixture offset hits every part through one machine from an
  onset; handling errors hit some parts of one shift, on every route.
- sensor drift: the reference is observable (`station_events` rows `gauge_check`, value =
  master-part reading − certified size), never the hidden true dimension.

### Signal graph and quality model (hidden truth)

```text
power_commanded ← setpoints          power_actual ← power_commanded × (1 + drive deviation)
spindle_load ← setpoints, true hardness, tool wear, overload factor
temperature  ← ambient + (6 + 0.3·load) × (1 + 1.5·coolant loss)
vibration    ← tool wear             ambient ← daily cycle (21 ± 1.5 °C) + excursion
surface-defect logit = −4 + Σ over the 3 cuts [6·(wear/limit − 0.8)⁺ + 0.12·(load − 60)⁺
                       + 0.35·(temp − 46)⁺ + 10·(|power residual| − 0.04)⁺ + 1.5·out-of-window]
                       + 0.06·(|hardness − 200| − 15)⁺
true dimension = 50 + N(0, 0.008) + fixture offset + handling offset
measured dimension = true + N(0, 0.002) + gauge drift;  reject if |measured − 50| > 0.03 or surface defect detected (97 %)
```

All weights and thresholds are in `sensors.py`; they are modelling choices, not plant data.

---

## 3. Determinism and isolation (how it is achieved)

1. **Named random streams** (`rng.stream(seed, name)`): arrival, per-part quality, handling,
   batch, gauge checks, and per machine `cycle:`, `sensor:`, `wear:`. No global RNG, no
   Python `hash()`.
2. **Fixed draw counts**: every consumer draws the same number of values per event whatever
   the fault state (for example 5 sensor normals per cut, 3 handling uniforms per part).
3. **Faults never change timing or routing.** Every scenario with the same seed has the same
   parts, routes and timestamps. A fault changes values only, so a fault run and its twin line
   up row by row, and the isolation test can be exact rather than statistical.
4. **Byte-level determinism of files**: explicit Arrow schemas (no pandas metadata), zstd,
   no write timestamps, JSON with sorted keys. Test: the same scenario written twice gives
   identical SHA-256 for every file; a different seed changes every random table.

Known limit: bit-identity across machines and library versions is not promised (pyarrow
writes its version into the Parquet footer; float maths could differ on another CPU/BLAS).
Within one locked environment it holds.

---

## 4. Case generator

`generate_case(observed_dir)` behaves like line monitoring:

- baseline = first 12 h; p₀ = max(observed rate, 2 %) — the floor is the line's known
  long-run reject rate;
- alarm at the end of the first 6 h window (hourly steps) where rejects
  k > n·p₀ + 3.5·√(n·p₀(1−p₀)) and k/n ≥ 1.5·p₀;
- case = title, description, alarm time, reject rate before/after, reject-reason mix.
  It does **not** say where rejects concentrate (machine, batch, shift): that is the agent's
  job. The alarm time is not the fault onset.
- `case_labels.json` in `ground_truth/` links `case_id` → fault → cause code.

Tests: the case is generated after `ground_truth/` has been deleted; its JSON contains no
fault name, cause code, target, scenario name, "onset" or "magnitude".

---

## 5. Measured results (2026-11-19, WSL2 dev machine)

Reference scenarios, 72 h, onset hour 24 (bad batch: first use of B0008). Rates are reject
rates of exposed parts after onset vs the fault-free twin.

| Scenario | Reject rate fault / twin | Alarm lag after onset |
| --- | --- | --- |
| ambient_excursion | 0.178 / 0.015 | 5 h |
| bad_batch_b0008 | 0.107 / 0.027 | 4.3 h |
| baseline | 0.018 | no alarm |
| coolant_loss_m3 | 0.080 / 0.015 | 7 h |
| drive_fault_m1 | 0.064 / 0.015 | 6 h |
| fixture_offset_m2b | 0.160 / 0.015 | 3 h |
| gauge_drift_q1 | 0.243 / 0.015 | 24 h (slow drift) |
| handling_error_shift_b | 0.084 / 0.015 | 11 h |
| spindle_overload_m2a | 0.073 / 0.015 | 17 h |
| tool_wear_m1 | 0.112 / 0.015 | 5 h |
| wrong_parameters_m3 | 0.069 / 0.015 | 6 h |

- One 72 h run: ≈ 2 400 parts, 7 200 sensor readings, 19 000 events; 1.2–1.4 s to simulate
  and write; `observed/` ≈ 0.76 MB.
- **Alarm rule tuning.** First rule (4 h window, z = 3, 0.5 % floor): 13/30 fault-free runs
  raised a false alarm. Final rule: 0/30 on seeds 0–29 (used for tuning) and **1/30 on fresh
  seeds 100–129**. Detection on a sampled family (seed 99, 6 per fault): 54/60. The misses
  were mostly `drive_power_deviation` at small magnitudes (2/6). After raising its sampling
  range to 0.20–0.30: 12/12.
- Ambient excursions are **sampled warm only** in families: a cold hall has no modelled
  quality effect (negative magnitudes are still allowed in hand-written scenarios).
- Test suite: 66 simulator tests, ≈ 25 s; whole repo `pytest`: 187 passed.

---

## 6. ADR

Moved to `docs/TECHNICAL_DECISIONS.md` as **ADR-008** at the merge into `main` (2026-11-21).

---

## 7. Open questions (for the user)

1. **Symptom codes.** The case uses `reject_rate_increase` and `reject_reason_mix`. Should
   these map to taxonomy `Symptom` nodes (Step 3 knowledge graph), and with which names?
2. **Unreported concentration.** The case deliberately omits "rejects concentrate on M2B /
   batch B0008 / shift B". Agree that discovering this is the agent's task (the scientist
   templates `station_defect_rate_change_point`, `defect_rate_by_batch`,
   `defect_rate_by_shift` do exactly this)?
3. **Where should simulator runs live?** Suggested `~/iqs-data/sim/<run>` (outside the repo,
   blueprint §4). Nothing generated is committed; only scenario YAMLs.
4. **Golden hashes.** Should a small reference run's file hashes be pinned in a test so any
   change to the simulator's numbers across commits is a deliberate, reviewed change (like the
   taxonomy fingerprint)? Not done yet because pyarrow upgrades would also change them.
5. **Image linker** (§2.2, assign MVTec images to simulated parts) is not in 5a; it belongs
   to Step 7 integration. OK to defer?
6. **Multi-fault scenarios** work in the engine (several faults per scenario), but the
   reference set and families are single-fault because the taxonomy says one primary cause
   per case. Should Step 5b also evaluate two-fault cases?

---

## 8. Text to merge into shared docs

*Merged on 2026-11-21* (CHANGELOG, TESTING, KNOWN_ISSUES #25–26, IMPLEMENTATION_PLAN,
ARCHITECTURE). Kept below as written.

**CHANGELOG.md** (under Step 5):

> - Step 5a: factory simulator `iqs_sim` — SimPy line L1 (4 stations, 5 machines), batches,
>   shifts, sensors (temperature, ambient, spindle load, vibration, tool wear, commanded vs
>   actual power); 10 injectable faults named as the taxonomy `simulator_fault`s; scenario
>   YAMLs in `configs/simulator/scenarios/`; Parquet export with a separate `ground_truth/`;
>   case generator; `scripts/simulate.py` (run / validate / family).

**TESTING.md** (simulator section):

> `packages/simulator/tests`: determinism (same scenario ⇒ identical SHA-256 of every file),
> fault isolation (each fault vs its fault-free twin changes exactly its declared columns, at
> its target, after its onset, and raises the exposed reject rate), schema tests, no
> ground-truth tokens in `observed/` or in cases, observed bytes independent of scenario
> metadata, DuckDB allow-list cannot read `ground_truth/`, case generator works with
> `ground_truth/` deleted, CLI. Runs in ≈ 25 s (48 h horizon, session-cached simulations).

**KNOWN_ISSUES.md:**

> - Simulator file hashes are reproducible within one locked environment; a pyarrow upgrade
>   changes the Parquet footer and therefore the hashes.
> - Alarm rule: 1/30 false alarms on fresh fault-free runs; slow faults (gauge drift) alarm
>   about 24 h after onset.
> - Cold ambient excursions have no modelled quality effect.
> - Quality-model weights are modelling choices, not calibrated to a real plant.

**IMPLEMENTATION_PLAN.md:** Step 5a done (simulator); 5b (scientist agent, templates, BH,
evidence view, production-monitoring page, evaluation family + manifest) blocked on Step 3.

**ARCHITECTURE.md:** add the run-directory layout (§1) and the signal graph (§2).
