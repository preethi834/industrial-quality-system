"""Behaviour of the fault-free line model."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime

from iqs_sim.line import SimulationResult, shift_of
from iqs_sim.machines import CUTTING_MACHINES, MACHINES
from iqs_sim.scenarios import Scenario

Run = Callable[[Scenario], SimulationResult]


def test_shift_calendar() -> None:
    assert shift_of(datetime(2026, 1, 5, 6, tzinfo=UTC))[:2] == ("20260105-A", "A")
    assert shift_of(datetime(2026, 1, 5, 13, 59, tzinfo=UTC))[1] == "A"
    assert shift_of(datetime(2026, 1, 5, 14, tzinfo=UTC))[:2] == ("20260105-B", "B")
    assert shift_of(datetime(2026, 1, 5, 23, tzinfo=UTC))[:2] == ("20260105-C", "C")
    # Night shift after midnight belongs to the shift that started the evening before.
    assert shift_of(datetime(2026, 1, 6, 5, 59, tzinfo=UTC))[:2] == ("20260105-C", "C")


def test_every_part_flows_through_all_stations(
    reference_scenarios: dict[str, Scenario], run_sim: Run
) -> None:
    res = run_sim(reference_scenarios["baseline"])
    parts = res.observed["parts"]
    assert len(parts) > 1000
    assert [p["part_id"] for p in parts] == sorted({p["part_id"] for p in parts})
    readings_per_part = Counter(r["part_id"] for r in res.observed["sensor_readings"])
    assert set(readings_per_part.values()) == {3}
    for p in parts:
        assert p["s1_machine_id"] == "M1" and p["s3_machine_id"] == "M3"
        assert p["s2_machine_id"] in ("M2A", "M2B")
        assert p["started_ts"] < p["completed_ts"]
    assert Counter(p["s2_machine_id"] for p in parts).keys() == {"M2A", "M2B"}
    ev = Counter(e["event_type"] for e in res.observed["station_events"])
    assert ev["part_start"] == ev["part_end"] == 4 * len(parts)


def test_events_and_readings_are_time_ordered(
    reference_scenarios: dict[str, Scenario], run_sim: Run
) -> None:
    res = run_sim(reference_scenarios["baseline"])
    ts = [e["ts"] for e in res.observed["station_events"]]
    assert ts == sorted(ts)
    assert [e["event_id"] for e in res.observed["station_events"]] == list(range(1, len(ts) + 1))


def test_tool_changes_follow_the_schedule_and_wear_stays_below_limit(
    reference_scenarios: dict[str, Scenario], run_sim: Run
) -> None:
    res = run_sim(reference_scenarios["baseline"])
    readings = res.observed["sensor_readings"]
    changes = Counter(
        e["machine_id"] for e in res.observed["station_events"] if e["event_type"] == "tool_change"
    )
    for m in CUTTING_MACHINES:
        spec = MACHINES[m]
        assert spec.tool_life_parts and spec.tool_wear_limit_mm
        cuts = sum(r["machine_id"] == m for r in readings)
        assert changes[m] == (cuts - 1) // spec.tool_life_parts
        wear = [r["tool_wear_mm"] for r in readings if r["machine_id"] == m]
        assert max(wear) < 0.8 * spec.tool_wear_limit_mm


def test_baseline_is_in_its_normal_operating_range(
    reference_scenarios: dict[str, Scenario], run_sim: Run
) -> None:
    res = run_sim(reference_scenarios["baseline"])
    parts = res.observed["parts"]
    reject_rate = sum(p["final_quality"] == "reject" for p in parts) / len(parts)
    assert 0.005 < reject_rate < 0.04
    readings = res.observed["sensor_readings"]
    assert all(18 <= r["ambient_temperature_c"] <= 26 for r in readings)
    assert all(r["parameter_set_id"] == "PS-STD" for r in readings)
    assert res.ground_truth["faults"] == []
    assert all(t["fault_ids"] == [] for t in res.ground_truth["part_truth"])


def test_gauge_checks_and_shifts_cover_the_run(
    reference_scenarios: dict[str, Scenario], run_sim: Run
) -> None:
    sc = reference_scenarios["baseline"]
    res = run_sim(sc)
    checks = [e for e in res.observed["station_events"] if e["event_type"] == "gauge_check"]
    interval_h = sc.line.gauge_check_interval_hours
    assert len(checks) == int(sc.duration_hours / interval_h - 1e-9)
    assert all(abs(e["value"]) < 0.01 for e in checks)
    shifts = res.observed["shifts"]
    assert shifts[0]["start_ts"] <= 0 and shifts[-1]["end_ts"] >= res.end_minutes
    assert all(a["end_ts"] == b["start_ts"] for a, b in zip(shifts, shifts[1:], strict=False))
    known = {s["shift_id"] for s in shifts}
    assert {p["shift_id"] for p in res.observed["parts"]} <= known
