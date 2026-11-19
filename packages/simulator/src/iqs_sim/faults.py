"""Injectable faults: one per taxonomy leaf cause.

The ``FaultType`` values are exactly the ``simulator_fault`` names in
``configs/taxonomy/causes.yaml`` (a test enforces the equality). Each fault has a
``FaultSpec`` that fixes:

- what it targets (a cutting machine, the gauge, a batch, the hall or a shift);
- what its magnitude means and the allowed range;
- ``signals``: the observable ``table.column`` values it is allowed to change, in
  addition to the inspection outcome columns (``OUTCOME_COLUMNS``). The
  fault-isolation test checks that a fault changes exactly these columns.

``ActiveFaults`` answers, at simulation time, "how does fault X modify this
computation for this machine / part / time?". All faults are inactive outside
``[onset, end)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from iqs_common.taxonomy import Taxonomy
from iqs_sim.machines import CUTTING_MACHINES, GAUGE_ID, HALL_ID

MINUTES_PER_HOUR = 60.0


class FaultType(StrEnum):
    ACCELERATED_TOOL_WEAR = "accelerated_tool_wear"
    SPINDLE_OVERLOAD = "spindle_overload"
    COOLANT_FLOW_LOSS = "coolant_flow_loss"
    DRIVE_POWER_DEVIATION = "drive_power_deviation"
    FIXTURE_OFFSET = "fixture_offset"
    BAD_MATERIAL_BATCH = "bad_material_batch"
    PARAMETER_SETPOINT_CHANGE = "parameter_setpoint_change"
    SENSOR_CALIBRATION_DRIFT = "sensor_calibration_drift"
    AMBIENT_TEMPERATURE_EXCURSION = "ambient_temperature_excursion"
    HANDLING_ERROR_ON_SHIFT = "handling_error_on_shift"


class TargetKind(StrEnum):
    CUTTING_MACHINE = "cutting_machine"
    GAUGE = "gauge"
    BATCH = "batch"
    HALL = "hall"
    SHIFT = "shift"


SHIFT_CODES: tuple[str, ...] = ("A", "B", "C")
BATCH_ID_PREFIX = "B"

# Inspection outcome columns: every fault may change these (that is its effect).
OUTCOME_COLUMNS: frozenset[str] = frozenset(
    {"parts.surface_defect_detected", "parts.final_quality", "parts.reject_reason"}
)

_TEMP = "sensor_readings.temperature_c"
_LOAD = "sensor_readings.spindle_load_pct"
_DIM = "parts.measured_dimension_mm"


@dataclass(frozen=True)
class FaultSpec:
    target_kind: TargetKind
    magnitude_unit: str
    magnitude_min: float
    magnitude_max: float
    min_abs_magnitude: float
    # Range used when sampling evaluation scenarios (by absolute value; sign random
    # when ``signed``).
    sample_range: tuple[float, float]
    signed: bool
    signals: frozenset[str]
    description: str
    has_time_window: bool = True


FAULT_SPECS: dict[FaultType, FaultSpec] = {
    FaultType.ACCELERATED_TOOL_WEAR: FaultSpec(
        TargetKind.CUTTING_MACHINE,
        "wear-rate multiplier",
        1.0,
        10.0,
        1.2,
        (2.0, 3.5),
        False,
        frozenset({"sensor_readings.tool_wear_mm", "sensor_readings.vibration_mm_s", _LOAD, _TEMP}),
        "Tool wears faster than planned, so it exceeds its wear limit before the "
        "scheduled tool change.",
    ),
    FaultType.SPINDLE_OVERLOAD: FaultSpec(
        TargetKind.CUTTING_MACHINE,
        "spindle-load multiplier",
        1.0,
        3.0,
        1.1,
        (1.3, 1.7),
        False,
        frozenset({_LOAD, _TEMP}),
        "Spindle load too high for the tool's wear state at correct setpoints.",
    ),
    FaultType.COOLANT_FLOW_LOSS: FaultSpec(
        TargetKind.CUTTING_MACHINE,
        "fraction of coolant flow lost",
        0.0,
        0.9,
        0.1,
        (0.25, 0.5),
        False,
        frozenset({_TEMP}),
        "Coolant flow drops, so process heat is removed less effectively.",
    ),
    FaultType.DRIVE_POWER_DEVIATION: FaultSpec(
        TargetKind.CUTTING_MACHINE,
        "relative deviation of actual from commanded power",
        -0.5,
        0.5,
        0.05,
        (0.12, 0.25),
        True,
        frozenset({"sensor_readings.power_actual_kw"}),
        "Drive delivers power that deviates from the commanded power.",
    ),
    FaultType.FIXTURE_OFFSET: FaultSpec(
        TargetKind.CUTTING_MACHINE,
        "positional offset in mm",
        -0.2,
        0.2,
        0.005,
        (0.02, 0.035),
        True,
        frozenset({_DIM}),
        "Fixture shifted: every part machined there gets a persistent dimensional offset.",
    ),
    FaultType.BAD_MATERIAL_BATCH: FaultSpec(
        TargetKind.BATCH,
        "true hardness deviation in HB (certificate unchanged)",
        -100.0,
        100.0,
        20.0,
        (35.0, 60.0),
        True,
        frozenset({_LOAD, _TEMP}),
        "One material batch is outside specification although its certificate is not.",
        has_time_window=False,
    ),
    FaultType.PARAMETER_SETPOINT_CHANGE: FaultSpec(
        TargetKind.CUTTING_MACHINE,
        "relative spindle-speed change of the new parameter set",
        -0.5,
        0.5,
        0.15,
        (0.2, 0.35),
        True,
        frozenset(
            {
                "sensor_readings.parameter_set_id",
                "sensor_readings.setpoint_speed_rpm",
                "sensor_readings.power_commanded_kw",
                "sensor_readings.power_actual_kw",
                _LOAD,
                _TEMP,
            }
        ),
        "A parameter set with a spindle speed outside the validated window is loaded.",
    ),
    FaultType.SENSOR_CALIBRATION_DRIFT: FaultSpec(
        TargetKind.GAUGE,
        "gauge drift in mm per hour",
        -0.02,
        0.02,
        0.0002,
        (0.0006, 0.0012),
        True,
        frozenset({_DIM, "station_events.value"}),
        "The dimension gauge drifts; good parts are rejected (and bad ones may pass) "
        "on the drifted reading. Periodic master-part checks show the drift.",
    ),
    FaultType.AMBIENT_TEMPERATURE_EXCURSION: FaultSpec(
        TargetKind.HALL,
        "added hall temperature in degC",
        -15.0,
        15.0,
        3.0,
        (6.0, 9.0),
        False,  # sampled warm only: a cold hall has no modelled quality effect
        frozenset({"sensor_readings.ambient_temperature_c", _TEMP}),
        "Hall temperature leaves its specified range.",
    ),
    FaultType.HANDLING_ERROR_ON_SHIFT: FaultSpec(
        TargetKind.SHIFT,
        "probability of a handling error per part",
        0.0,
        1.0,
        0.02,
        (0.15, 0.3),
        False,
        frozenset({_DIM}),
        "Intermittent mis-loading at S1 during one shift: random dimensional offset.",
    ),
}


class FaultConfig(BaseModel):
    """One injected fault as written in a scenario YAML file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: FaultType
    target: str = Field(min_length=1)
    magnitude: float
    onset_hours: float = Field(default=0.0, ge=0)
    duration_hours: float | None = Field(default=None, gt=0)

    @property
    def spec(self) -> FaultSpec:
        return FAULT_SPECS[self.type]

    @model_validator(mode="after")
    def _check(self) -> FaultConfig:
        spec = self.spec
        if not spec.magnitude_min <= self.magnitude <= spec.magnitude_max:
            raise ValueError(
                f"{self.type}: magnitude {self.magnitude} outside "
                f"[{spec.magnitude_min}, {spec.magnitude_max}] ({spec.magnitude_unit})"
            )
        if self.type in (FaultType.ACCELERATED_TOOL_WEAR, FaultType.SPINDLE_OVERLOAD):
            if self.magnitude < spec.min_abs_magnitude:
                raise ValueError(f"{self.type}: magnitude must be >= {spec.min_abs_magnitude}")
        elif abs(self.magnitude) < spec.min_abs_magnitude:
            raise ValueError(f"{self.type}: |magnitude| must be >= {spec.min_abs_magnitude}")
        _check_target(spec.target_kind, self.target, self.type)
        if not spec.has_time_window and (self.onset_hours != 0 or self.duration_hours):
            raise ValueError(
                f"{self.type}: onset/duration are not used (the batch defines when it acts)"
            )
        return self

    def active(self, t_min: float) -> bool:
        onset = self.onset_hours * MINUTES_PER_HOUR
        if t_min < onset:
            return False
        if self.duration_hours is None:
            return True
        return t_min < onset + self.duration_hours * MINUTES_PER_HOUR


def cause_code_for(taxonomy: Taxonomy, simulator_fault: str) -> str:
    matches = [c.code for c in taxonomy.causes if c.simulator_fault == simulator_fault]
    if len(matches) != 1:
        raise ValueError(
            f"simulator_fault {simulator_fault!r} maps to {len(matches)} taxonomy causes, "
            "expected exactly 1"
        )
    return matches[0]


def batch_id(number: int) -> str:
    return f"{BATCH_ID_PREFIX}{number:04d}"


def _check_target(kind: TargetKind, target: str, fault: FaultType) -> None:
    if kind is TargetKind.CUTTING_MACHINE:
        ok = target in CUTTING_MACHINES
        expected = f"one of {CUTTING_MACHINES}"
    elif kind is TargetKind.GAUGE:
        ok, expected = target == GAUGE_ID, repr(GAUGE_ID)
    elif kind is TargetKind.HALL:
        ok, expected = target == HALL_ID, repr(HALL_ID)
    elif kind is TargetKind.SHIFT:
        ok, expected = target in SHIFT_CODES, f"one of {SHIFT_CODES}"
    else:
        digits = target[len(BATCH_ID_PREFIX) :]
        ok = (
            target.startswith(BATCH_ID_PREFIX)
            and len(digits) == 4
            and digits.isdigit()
            and int(digits) >= 1
        )
        expected = "a batch id like 'B0003'"
    if not ok:
        raise ValueError(f"{fault}: target {target!r} invalid, expected {expected}")


@dataclass(frozen=True)
class IndexedFault:
    fault_id: str
    config: FaultConfig


class ActiveFaults:
    """Fault lookups used by the line model. Inactive faults return neutral values."""

    def __init__(self, faults: tuple[FaultConfig, ...]) -> None:
        self.faults = tuple(IndexedFault(f"F{i + 1:02d}", f) for i, f in enumerate(faults))

    def _matching(self, fault_type: FaultType, target: str, t_min: float) -> list[IndexedFault]:
        return [
            f
            for f in self.faults
            if f.config.type is fault_type and f.config.target == target and f.config.active(t_min)
        ]

    def _product(self, fault_type: FaultType, target: str, t_min: float, hits: set[str]) -> float:
        value = 1.0
        for f in self._matching(fault_type, target, t_min):
            value *= f.config.magnitude
            hits.add(f.fault_id)
        return value

    def _sum(self, fault_type: FaultType, target: str, t_min: float, hits: set[str]) -> float:
        value = 0.0
        for f in self._matching(fault_type, target, t_min):
            value += f.config.magnitude
            hits.add(f.fault_id)
        return value

    def wear_multiplier(self, machine: str, t: float, hits: set[str]) -> float:
        return self._product(FaultType.ACCELERATED_TOOL_WEAR, machine, t, hits)

    def overload_factor(self, machine: str, t: float, hits: set[str]) -> float:
        return self._product(FaultType.SPINDLE_OVERLOAD, machine, t, hits)

    def coolant_loss(self, machine: str, t: float, hits: set[str]) -> float:
        return min(0.9, self._sum(FaultType.COOLANT_FLOW_LOSS, machine, t, hits))

    def drive_deviation(self, machine: str, t: float, hits: set[str]) -> float:
        return self._sum(FaultType.DRIVE_POWER_DEVIATION, machine, t, hits)

    def fixture_offset_mm(self, machine: str, t: float, hits: set[str]) -> float:
        return self._sum(FaultType.FIXTURE_OFFSET, machine, t, hits)

    def speed_change(self, machine: str, t: float, hits: set[str]) -> float:
        return self._sum(FaultType.PARAMETER_SETPOINT_CHANGE, machine, t, hits)

    def ambient_offset_c(self, t: float, hits: set[str]) -> float:
        return self._sum(FaultType.AMBIENT_TEMPERATURE_EXCURSION, HALL_ID, t, hits)

    def gauge_drift_mm(self, t: float, hits: set[str]) -> float:
        drift = 0.0
        for f in self._matching(FaultType.SENSOR_CALIBRATION_DRIFT, GAUGE_ID, t):
            hours = (t - f.config.onset_hours * MINUTES_PER_HOUR) / MINUTES_PER_HOUR
            drift += f.config.magnitude * hours
            hits.add(f.fault_id)
        return drift

    def batch_hardness_offset(self, batch: str) -> tuple[float, list[str]]:
        offset, ids = 0.0, []
        for f in self.faults:
            if f.config.type is FaultType.BAD_MATERIAL_BATCH and f.config.target == batch:
                offset += f.config.magnitude
                ids.append(f.fault_id)
        return offset, ids

    def handling_error_probability(self, shift_code: str, t: float) -> tuple[float, list[str]]:
        matches = self._matching(FaultType.HANDLING_ERROR_ON_SHIFT, shift_code, t)
        p = 1.0
        for f in matches:
            p *= 1.0 - f.config.magnitude
        return 1.0 - p, [f.fault_id for f in matches]
