"""SimPy model of line L1: parts flow S1 -> S2 -> S3 -> S4, faults act on the way.

``simulate(scenario)`` returns a ``SimulationResult`` holding plain row lists for the
observable tables and, separately, for the ground-truth tables. Writing them to
disk is ``iqs_sim.export``'s job.

Determinism and isolation rules followed here:

- all randomness comes from named streams (``iqs_sim.rng``); per-part draws happen
  when the part is created (in part order), per-cut draws on the machine's own
  stream (in that machine's part order);
- each consumer draws a fixed number of values whatever the fault state;
- processing and routing never depend on faults, so every scenario with the same
  seed has the same parts, routes and timestamps, and a fault changes values only.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import simpy

from iqs_sim.faults import MINUTES_PER_HOUR, ActiveFaults, batch_id
from iqs_sim.machines import (
    CHANGED_PARAMETER_SET,
    CUTTING_STATIONS,
    DIMENSION_TOLERANCE_MM,
    GAUGE_ID,
    INSPECTION_STATION,
    MACHINES,
    MATERIAL,
    NOMINAL_DIMENSION_MM,
    NOMINAL_HARDNESS_HB,
    STANDARD_PARAMETER_SET,
    STATIONS,
    MachineKind,
    ToolState,
    machines_at,
)
from iqs_sim.rng import stream
from iqs_sim.scenarios import Scenario
from iqs_sim.sensors import (
    N_CUT_NOISE,
    CutInputs,
    ambient_temperature,
    cut_defect_logit,
    cut_reading,
    hardness_defect_logit,
    sigmoid,
    wear_increment,
)

Row = dict[str, Any]

SUPPLIERS: tuple[str, ...] = ("SUP-01", "SUP-02", "SUP-03")
SD_TRUE_DIM_MM = 0.008
SD_GAUGE_MM = 0.002
SD_CERT_HB = 5.0
SD_TRUE_VS_CERT_HB = 3.0
SURFACE_DETECTION_RATE = 0.97
HANDLING_OFFSET_MM = (0.035, 0.06)
SHIFT_STARTS_H: tuple[tuple[int, str], ...] = ((6, "A"), (14, "B"), (22, "C"))

# Final-quality labels in the observable parts table.
PASS, REJECT = "pass", "reject"


@dataclass
class SimulationResult:
    scenario: Scenario
    observed: dict[str, list[Row]]
    ground_truth: dict[str, list[Row]]
    end_minutes: float


@dataclass
class _Batch:
    batch_id: str
    supplier_id: str
    hardness_cert_hb: float
    hardness_true_hb: float
    fault_ids: list[str]
    first_use_min: float
    quantity: int = 0


@dataclass
class _Part:
    index: int
    part_id: str
    batch: _Batch
    n_dim: float
    u_surface: float
    u_detect: float
    n_gauge: float
    u_handling: float
    u_sign: float
    u_size: float
    logit: float = 0.0
    dim_offset_mm: float = 0.0
    machines: list[str] = field(default_factory=list)
    fault_ids: set[str] = field(default_factory=set)
    shift_id: str = ""
    started_min: float = 0.0


def shift_of(moment: datetime) -> tuple[str, str, datetime]:
    """Return (shift_id, shift_code, shift start) for a UTC moment.

    Shifts: A 06-14, B 14-22, C 22-06 (UTC; the night shift keeps the date it started on).
    """
    day = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    if moment.hour < SHIFT_STARTS_H[0][0]:
        day -= timedelta(days=1)
        start_h, code = SHIFT_STARTS_H[-1]
    else:
        start_h, code = max((s for s in SHIFT_STARTS_H if s[0] <= moment.hour), key=lambda s: s[0])
    return f"{day:%Y%m%d}-{code}", code, day + timedelta(hours=start_h)


class _LineSimulation:
    def __init__(self, scenario: Scenario) -> None:
        self.sc = scenario
        self.cfg = scenario.line
        self.env = simpy.Environment()
        self.faults = ActiveFaults(scenario.faults)
        seed = scenario.seed
        self.horizon_min = scenario.duration_hours * MINUTES_PER_HOUR
        self.rng_arrival = stream(seed, "arrival")
        self.rng_part = stream(seed, "part-quality")
        self.rng_handling = stream(seed, "handling")
        self.rng_batch = stream(seed, "batch")
        self.rng_gauge_check = stream(seed, "gauge-check")
        self.rng_cycle = {m: stream(seed, f"cycle:{m}") for m in MACHINES}
        self.rng_sensor = {m: stream(seed, f"sensor:{m}") for m in MACHINES}
        self.rng_wear = {m: stream(seed, f"wear:{m}") for m in MACHINES}
        self.tools = {m: ToolState() for m, s in MACHINES.items() if s.kind is MachineKind.CUTTING}
        self.stores: dict[str, simpy.Store] = {}
        for station in STATIONS:
            store = simpy.Store(self.env)
            store.items.extend(machines_at(station))
            self.stores[station] = store
        self.batches: list[_Batch] = []
        self.parts: list[Row] = []
        self.readings: list[Row] = []
        self.events: list[tuple[float, int, Row]] = []
        self.part_truth: list[Row] = []
        self._event_seq = 0
        self.in_flight = 0

    # ----------------------------------------------------------------- helpers
    def moment(self, t_min: float) -> datetime:
        return self.sc.start + timedelta(minutes=t_min)

    def event(
        self,
        t: float,
        machine_id: str,
        event_type: str,
        part_id: str | None,
        value: float | None = None,
    ) -> None:
        row = {
            "ts": t,
            "station_id": MACHINES[machine_id].station_id,
            "machine_id": machine_id,
            "event_type": event_type,
            "part_id": part_id,
            "value": value,
        }
        self.events.append((t, self._event_seq, row))
        self._event_seq += 1

    def batch_for(self, index: int, t: float) -> _Batch:
        number = index // self.cfg.batch_size + 1
        if len(self.batches) < number:
            supplier = SUPPLIERS[int(self.rng_batch.integers(0, len(SUPPLIERS)))]
            n_cert, n_true = self.rng_batch.standard_normal(2)
            cert = NOMINAL_HARDNESS_HB + SD_CERT_HB * float(n_cert)
            bid = batch_id(number)
            offset, ids = self.faults.batch_hardness_offset(bid)
            true = cert + SD_TRUE_VS_CERT_HB * float(n_true) + offset
            self.batches.append(_Batch(bid, supplier, cert, true, ids, t))
        batch = self.batches[number - 1]
        batch.quantity += 1
        return batch

    # --------------------------------------------------------------- processes
    def source(self) -> Generator[simpy.Event, Any, None]:
        index = 0
        while True:
            jitter = float(self.rng_arrival.uniform(-0.15, 0.15))
            yield self.env.timeout(self.cfg.takt_minutes * (1.0 + jitter))
            if self.env.now >= self.horizon_min:
                return
            t = self.env.now
            n_dim, n_gauge = self.rng_part.standard_normal(2)
            u_surface, u_detect = self.rng_part.random(2)
            u_handling, u_sign, u_size = self.rng_handling.random(3)
            part = _Part(
                index=index,
                part_id=f"P{index + 1:06d}",
                batch=self.batch_for(index, t),
                n_dim=float(n_dim),
                u_surface=float(u_surface),
                u_detect=float(u_detect),
                n_gauge=float(n_gauge),
                u_handling=float(u_handling),
                u_sign=float(u_sign),
                u_size=float(u_size),
            )
            self.env.process(self.flow(part))
            index += 1

    def gauge_checks(self) -> Generator[simpy.Event, Any, None]:
        interval = self.cfg.gauge_check_interval_hours * MINUTES_PER_HOUR
        while True:
            yield self.env.timeout(interval)
            if self.env.now >= self.horizon_min:
                return
            noise = float(self.rng_gauge_check.standard_normal())
            drift = self.faults.gauge_drift_mm(self.env.now, set())
            # value = gauge reading of the master part minus its certified size
            self.event(self.env.now, GAUGE_ID, "gauge_check", None, drift + SD_GAUGE_MM * noise)

    def flow(self, part: _Part) -> Generator[simpy.Event, Any, None]:
        self.in_flight += 1
        part.logit = self.cfg.base_defect_logit + hardness_defect_logit(part.batch.hardness_true_hb)
        part.fault_ids.update(part.batch.fault_ids)
        for station in CUTTING_STATIONS:
            machine_id = yield self.stores[station].get()
            start = self.env.now
            if station == CUTTING_STATIONS[0]:
                self.load_part(part, start)
            cycle = MACHINES[machine_id].cycle_minutes * (
                1.0 + 0.05 * float(self.rng_cycle[machine_id].standard_normal())
            )
            self.event(start, machine_id, "part_start", part.part_id)
            self.cut(part, machine_id, start, start + cycle / 2)
            yield self.env.timeout(cycle)
            self.event(self.env.now, machine_id, "part_end", part.part_id)
            part.machines.append(machine_id)
            yield self.stores[station].put(machine_id)

        gauge = yield self.stores[INSPECTION_STATION].get()
        start = self.env.now
        cycle = MACHINES[gauge].cycle_minutes
        self.event(start, gauge, "part_start", part.part_id)
        yield self.env.timeout(cycle)
        self.inspect(part, start + cycle / 2)
        self.event(self.env.now, gauge, "part_end", part.part_id)
        yield self.stores[INSPECTION_STATION].put(gauge)
        self.in_flight -= 1

    def load_part(self, part: _Part, t: float) -> None:
        part.started_min = t
        part.shift_id, code, _ = shift_of(self.moment(t))
        p_error, ids = self.faults.handling_error_probability(code, t)
        if part.u_handling < p_error:
            lo, hi = HANDLING_OFFSET_MM
            size = lo + (hi - lo) * part.u_size
            part.dim_offset_mm += size if part.u_sign < 0.5 else -size
            part.fault_ids.update(ids)

    def cut(self, part: _Part, machine_id: str, start: float, t: float) -> None:
        spec = MACHINES[machine_id]
        assert spec.nominal_speed_rpm is not None and spec.nominal_feed_mm_rev is not None
        assert spec.tool_life_parts is not None
        tool = self.tools[machine_id]
        if tool.parts_cut >= spec.tool_life_parts:
            self.event(start, machine_id, "tool_change", None)
            tool.tool_number += 1
            tool.wear_mm = 0.0
            tool.parts_cut = 0

        hits: set[str] = set()
        wear_noise = float(self.rng_wear[machine_id].standard_normal())
        noise = tuple(float(v) for v in self.rng_sensor[machine_id].standard_normal(N_CUT_NOISE))
        tool.wear_mm += wear_increment(
            spec, self.faults.wear_multiplier(machine_id, t, hits), wear_noise
        )
        tool.parts_cut += 1

        speed_change = self.faults.speed_change(machine_id, t, hits)
        speed = spec.nominal_speed_rpm * (1.0 + speed_change)
        window = spec.speed_window
        assert window is not None
        out_of_window = not window[0] <= speed <= window[1]
        hour = self.moment(t).hour + self.moment(t).minute / 60.0
        ambient = ambient_temperature(hour, self.faults.ambient_offset_c(t, hits), noise[4])
        inputs = CutInputs(
            speed_rpm=speed,
            feed_mm_rev=spec.nominal_feed_mm_rev,
            hardness_hb=part.batch.hardness_true_hb,
            wear_mm=tool.wear_mm,
            ambient_c=ambient,
            overload_factor=self.faults.overload_factor(machine_id, t, hits),
            coolant_loss=self.faults.coolant_loss(machine_id, t, hits),
            drive_deviation=self.faults.drive_deviation(machine_id, t, hits),
        )
        reading = cut_reading(spec, inputs, noise)
        part.logit += cut_defect_logit(spec, inputs, reading, out_of_window=out_of_window)
        part.dim_offset_mm += self.faults.fixture_offset_mm(machine_id, t, hits)
        part.fault_ids.update(hits)
        self.readings.append(
            {
                "ts": t,
                "part_id": part.part_id,
                "station_id": spec.station_id,
                "machine_id": machine_id,
                "tool_id": f"{machine_id}-T{tool.tool_number:04d}",
                "parameter_set_id": (
                    CHANGED_PARAMETER_SET if speed_change != 0.0 else STANDARD_PARAMETER_SET
                ),
                "setpoint_speed_rpm": speed,
                "setpoint_feed_mm_rev": spec.nominal_feed_mm_rev,
                "power_commanded_kw": reading.power_commanded_kw,
                "power_actual_kw": reading.power_actual_kw,
                "spindle_load_pct": reading.spindle_load_pct,
                "temperature_c": reading.temperature_c,
                "ambient_temperature_c": ambient,
                "vibration_mm_s": reading.vibration_mm_s,
                "tool_wear_mm": tool.wear_mm,
            }
        )

    def inspect(self, part: _Part, t: float) -> None:
        hits: set[str] = set()
        true_dim = NOMINAL_DIMENSION_MM + SD_TRUE_DIM_MM * part.n_dim + part.dim_offset_mm
        measured = true_dim + SD_GAUGE_MM * part.n_gauge + self.faults.gauge_drift_mm(t, hits)
        true_surface = part.u_surface < sigmoid(part.logit)
        surface_detected = true_surface and part.u_detect < SURFACE_DETECTION_RATE
        dim_reject = abs(measured - NOMINAL_DIMENSION_MM) > DIMENSION_TOLERANCE_MM
        if dim_reject and surface_detected:
            reason: str | None = "dimension_and_surface"
        elif dim_reject:
            reason = "dimension"
        elif surface_detected:
            reason = "surface"
        else:
            reason = None
        self.parts.append(
            {
                "part_id": part.part_id,
                "batch_id": part.batch.batch_id,
                "shift_id": part.shift_id,
                "started_ts": part.started_min,
                "completed_ts": self.env.now + MACHINES[GAUGE_ID].cycle_minutes / 2,
                "s1_machine_id": part.machines[0],
                "s2_machine_id": part.machines[1],
                "s3_machine_id": part.machines[2],
                "station_path": ">".join([*part.machines, GAUGE_ID]),
                "measured_dimension_mm": measured,
                "surface_defect_detected": surface_detected,
                "final_quality": REJECT if reason else PASS,
                "reject_reason": reason,
            }
        )
        true_dim_bad = abs(true_dim - NOMINAL_DIMENSION_MM) > DIMENSION_TOLERANCE_MM
        self.part_truth.append(
            {
                "part_id": part.part_id,
                "true_dimension_mm": true_dim,
                "true_surface_defect": true_surface,
                "truly_defective": true_surface or true_dim_bad,
                "defect_probability": sigmoid(part.logit),
                "fault_ids": sorted(part.fault_ids | hits),
            }
        )

    # --------------------------------------------------------------------- run
    def run(self) -> SimulationResult:
        self.env.process(self.source())
        self.env.process(self.gauge_checks())
        self.env.run()
        if self.in_flight:
            raise RuntimeError(f"{self.in_flight} parts never finished")
        end = max([self.horizon_min, *(p["completed_ts"] for p in self.parts)])
        self._check_batch_targets()
        observed = {
            "parts": sorted(self.parts, key=lambda r: r["part_id"]),
            "sensor_readings": self.readings,
            "station_events": [row for _, _, row in sorted(self.events, key=lambda e: e[:2])],
            "batches": [
                {
                    "batch_id": b.batch_id,
                    "supplier_id": b.supplier_id,
                    "material": MATERIAL,
                    "hardness_cert_hb": b.hardness_cert_hb,
                    "quantity": b.quantity,
                    "first_use_ts": b.first_use_min,
                }
                for b in self.batches
            ],
            "shifts": self._shifts(end),
            "machines": _machine_rows(),
        }
        truth = {
            "faults": self._fault_rows(),
            "part_truth": sorted(self.part_truth, key=lambda r: r["part_id"]),
            "batch_truth": [
                {"batch_id": b.batch_id, "hardness_true_hb": b.hardness_true_hb}
                for b in self.batches
            ],
        }
        for i, row in enumerate(observed["sensor_readings"]):
            row["reading_id"] = i + 1
        for i, row in enumerate(observed["station_events"]):
            row["event_id"] = i + 1
        return SimulationResult(self.sc, observed, truth, end)

    def _check_batch_targets(self) -> None:
        used = {b.batch_id for b in self.batches}
        for f in self.sc.faults:
            if f.spec.target_kind.value == "batch" and f.target not in used:
                raise ValueError(
                    f"{f.type}: batch {f.target} is never used in this run "
                    f"({len(self.batches)} batches); pick an earlier batch or a longer run"
                )

    def _shifts(self, end_min: float) -> list[Row]:
        rows: list[Row] = []
        _, _, first = shift_of(self.sc.start)
        cursor = first
        end = self.moment(end_min)
        while cursor < end:
            shift_id, code, start = shift_of(cursor)
            nxt = start + timedelta(hours=8)
            rows.append(
                {
                    "shift_id": shift_id,
                    "shift_code": code,
                    "crew_id": f"CREW-{code}",
                    "start_ts": (start - self.sc.start).total_seconds() / 60.0,
                    "end_ts": (nxt - self.sc.start).total_seconds() / 60.0,
                }
            )
            cursor = nxt
        return rows

    def _fault_rows(self) -> list[Row]:
        rows: list[Row] = []
        for f in self.faults.faults:
            c = f.config
            if c.spec.has_time_window:
                start = c.onset_hours * MINUTES_PER_HOUR
                onset: float | None = start
                end = (
                    None
                    if c.duration_hours is None
                    else start + c.duration_hours * MINUTES_PER_HOUR
                )
            else:
                batch = next((b for b in self.batches if b.batch_id == c.target), None)
                onset, end = (batch.first_use_min if batch else None), None
            rows.append(
                {
                    "fault_id": f.fault_id,
                    "simulator_fault": c.type.value,
                    "target": c.target,
                    "magnitude": c.magnitude,
                    "magnitude_unit": c.spec.magnitude_unit,
                    "onset_ts": onset,
                    "end_ts": end,
                }
            )
        return rows


def _machine_rows() -> list[Row]:
    rows: list[Row] = []
    for m in MACHINES.values():
        window = m.speed_window
        rows.append(
            {
                "machine_id": m.machine_id,
                "station_id": m.station_id,
                "kind": m.kind.value,
                "process": m.process,
                "nominal_cycle_minutes": m.cycle_minutes,
                "nominal_speed_rpm": m.nominal_speed_rpm,
                "nominal_feed_mm_rev": m.nominal_feed_mm_rev,
                "speed_window_low_rpm": window[0] if window else None,
                "speed_window_high_rpm": window[1] if window else None,
                "tool_life_parts": m.tool_life_parts,
                "tool_wear_limit_mm": m.tool_wear_limit_mm,
            }
        )
    return rows


def simulate(scenario: Scenario) -> SimulationResult:
    """Run one scenario. Pure: the result depends only on ``scenario``."""
    return _LineSimulation(scenario).run()
