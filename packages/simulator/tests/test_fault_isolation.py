"""Each fault changes exactly its intended signals, only where and when it acts.

Every reference fault scenario is compared with its fault-free twin (same seed,
start and horizon). Because routing and timing never depend on faults and every
consumer draws from its own random stream, all rows line up one to one and any
difference is caused by the fault.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from iqs_sim.faults import FAULT_SPECS, MINUTES_PER_HOUR, OUTCOME_COLUMNS, TargetKind
from iqs_sim.line import Row, SimulationResult
from iqs_sim.scenarios import Scenario

Run = Callable[[Scenario], SimulationResult]
FAULT_SCENARIOS = [
    "accelerated_tool_wear",
    "spindle_overload",
    "coolant_flow_loss",
    "drive_power_deviation",
    "fixture_offset",
    "bad_material_batch",
    "parameter_setpoint_change",
    "sensor_calibration_drift",
    "ambient_temperature_excursion",
    "handling_error_on_shift",
]


def _scenario_for(fault: str, scenarios: dict[str, Scenario]) -> Scenario:
    (sc,) = [s for s in scenarios.values() if s.faults and s.faults[0].type.value == fault]
    return sc


def _diff_rows(a: list[Row], b: list[Row]) -> tuple[set[str], list[int]]:
    assert len(a) == len(b)
    columns: set[str] = set()
    rows: list[int] = []
    for i, (ra, rb) in enumerate(zip(a, b, strict=True)):
        changed = {k for k in ra if ra[k] != rb[k]}
        if changed:
            columns |= changed
            rows.append(i)
    return columns, rows


def _reject_rate(rows: list[Row]) -> float:
    return sum(r["final_quality"] == "reject" for r in rows) / len(rows)


@pytest.mark.parametrize("fault", FAULT_SCENARIOS)
def test_fault_changes_exactly_its_signals(
    fault: str, reference_scenarios: dict[str, Scenario], run_sim: Run
) -> None:
    sc = _scenario_for(fault, reference_scenarios)
    cfg = sc.faults[0]
    spec = FAULT_SPECS[cfg.type]
    faulty = run_sim(sc)
    clean = run_sim(sc.without_faults())

    changed: set[str] = set()
    diff_rows: dict[str, list[int]] = {}
    for table in clean.observed:
        cols, rows = _diff_rows(clean.observed[table], faulty.observed[table])
        changed |= {f"{table}.{c}" for c in cols}
        diff_rows[table] = rows

    assert changed - OUTCOME_COLUMNS == spec.signals
    assert "parts.final_quality" in changed

    # Where and when: differences only after onset, at the target.
    onset = cfg.onset_hours * MINUTES_PER_HOUR
    readings = faulty.observed["sensor_readings"]
    changed_readings = [readings[i] for i in diff_rows["sensor_readings"]]
    touches_readings = any(s.startswith("sensor_readings.") for s in spec.signals)
    if not touches_readings:
        assert not changed_readings
    elif spec.target_kind is TargetKind.CUTTING_MACHINE:
        assert changed_readings
        assert {r["machine_id"] for r in changed_readings} == {cfg.target}
        assert min(r["ts"] for r in changed_readings) >= onset
    elif spec.target_kind is TargetKind.BATCH:
        parts_of_batch = {
            p["part_id"] for p in faulty.observed["parts"] if p["batch_id"] == cfg.target
        }
        assert {r["part_id"] for r in changed_readings} == parts_of_batch
    elif spec.target_kind is TargetKind.HALL:
        assert {r["machine_id"] for r in changed_readings} == {"M1", "M2A", "M2B", "M3"}
        assert min(r["ts"] for r in changed_readings) >= onset

    parts = faulty.observed["parts"]
    changed_parts = [parts[i] for i in diff_rows["parts"]]
    assert changed_parts
    if spec.target_kind is TargetKind.CUTTING_MACHINE:
        assert all(cfg.target in p["station_path"].split(">") for p in changed_parts)
    if spec.target_kind is TargetKind.SHIFT:
        assert {p["shift_id"][-1] for p in changed_parts} == {cfg.target}
    if spec.has_time_window:
        assert min(p["completed_ts"] for p in changed_parts) >= onset

    # The fault has a visible effect: more rejects among exposed parts than in the twin.
    def exposed(p: Row) -> bool:
        if spec.target_kind is TargetKind.BATCH:
            return bool(p["batch_id"] == cfg.target)
        if p["completed_ts"] < onset:
            return False
        if spec.target_kind is TargetKind.CUTTING_MACHINE:
            return cfg.target in p["station_path"].split(">")
        if spec.target_kind is TargetKind.SHIFT:
            return bool(p["shift_id"].endswith(cfg.target))
        return True

    exposed_f = [p for p in parts if exposed(p)]
    exposed_c = [p for p in clean.observed["parts"] if exposed(p)]
    assert _reject_rate(exposed_f) > _reject_rate(exposed_c) + 0.01

    # Ground truth records the fault and the parts it touched.
    (truth,) = faulty.ground_truth["faults"]
    assert truth["simulator_fault"] == fault
    touched = {t["part_id"] for t in faulty.ground_truth["part_truth"] if t["fault_ids"]}
    assert touched
    assert {p["part_id"] for p in changed_parts} <= touched


def test_fault_free_twin_is_the_baseline(
    reference_scenarios: dict[str, Scenario], run_sim: Run
) -> None:
    """Reference fault scenarios share seed/start/horizon with baseline.yaml."""
    base = reference_scenarios["baseline"]
    for sc in reference_scenarios.values():
        assert (sc.seed, sc.start, sc.duration_hours, sc.line) == (
            base.seed,
            base.start,
            base.duration_hours,
            base.line,
        )
