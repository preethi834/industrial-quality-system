"""Scenario files: what to simulate, for how long, with which injected faults.

A scenario is a YAML file in ``configs/simulator/scenarios/``::

    name: tool_wear_m1
    seed: 7
    start: 2026-01-05T06:00:00Z
    duration_hours: 72
    faults:
      - type: accelerated_tool_wear
        target: M1
        magnitude: 2.5
        onset_hours: 24

The simulation is a pure function of the scenario: same scenario => same output
files, byte for byte. The scenario itself (name, description, faults) is ground
truth and is never written to the observable output.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from iqs_sim.faults import (
    FAULT_SPECS,
    SHIFT_CODES,
    FaultConfig,
    FaultType,
    TargetKind,
    batch_id,
)
from iqs_sim.machines import CUTTING_MACHINES, GAUGE_ID, HALL_ID
from iqs_sim.rng import stream

SCENARIO_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"


class LineConfig(BaseModel):
    """Tunable line parameters (defaults are the reference line)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    takt_minutes: float = Field(default=1.8, gt=0.5, le=60)
    batch_size: int = Field(default=150, ge=10, le=100_000)
    gauge_check_interval_hours: float = Field(default=2.0, gt=0, le=24)
    base_defect_logit: float = Field(default=-4.0, le=0)


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=SCENARIO_NAME_PATTERN, max_length=80)
    description: str = ""
    seed: int = Field(ge=0, lt=2**63)
    start: datetime
    duration_hours: float = Field(gt=0, le=24 * 31)
    line: LineConfig = LineConfig()
    faults: tuple[FaultConfig, ...] = ()

    @field_validator("start")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("start must include a timezone (use ...Z for UTC)")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _onsets_inside_horizon(self) -> Scenario:
        for f in self.faults:
            if f.onset_hours >= self.duration_hours:
                raise ValueError(
                    f"{f.type}: onset_hours {f.onset_hours} is not before the end of the "
                    f"run ({self.duration_hours} h)"
                )
        return self

    def with_overrides(
        self, *, seed: int | None = None, duration_hours: float | None = None
    ) -> Scenario:
        data = self.model_dump()
        if seed is not None:
            data["seed"] = seed
        if duration_hours is not None:
            data["duration_hours"] = duration_hours
        return Scenario.model_validate(data)

    def without_faults(self) -> Scenario:
        return self.model_copy(update={"faults": ()})


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a scenario YAML file (safe_load only)."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return Scenario.model_validate(data)


def scenario_to_yaml(scenario: Scenario) -> str:
    data = scenario.model_dump(mode="json", exclude_defaults=False)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


# --------------------------------------------------------------------------------------
# Scenario families (input for the step-5b evaluation; the gate needs a manifest hash).
# --------------------------------------------------------------------------------------


def _sample_target(kind: TargetKind, rng: np.random.Generator, n_batches: int) -> str:
    if kind is TargetKind.CUTTING_MACHINE:
        return str(rng.choice(CUTTING_MACHINES))
    if kind is TargetKind.GAUGE:
        return GAUGE_ID
    if kind is TargetKind.HALL:
        return HALL_ID
    if kind is TargetKind.SHIFT:
        return str(rng.choice(SHIFT_CODES))
    # A batch used after the earliest onset and well before the end.
    lo = max(2, n_batches // 3)
    hi = max(lo + 1, (2 * n_batches) // 3)
    return batch_id(int(rng.integers(lo, hi)))


def near_limit_range(fault_type: FaultType) -> tuple[float, float]:
    """Weaker magnitudes around the monitoring detection limit.

    From halfway between the validity floor and the standard range up into the bottom
    quarter of the standard range: ``[floor + (lo - floor)/2, lo + (hi - lo)/4]``. One
    rule for every fault; measured alarm rates in this band differ by fault (step-3 notes).
    """
    spec = FAULT_SPECS[fault_type]
    floor = spec.min_abs_magnitude
    lo, hi = spec.sample_range
    return (floor + (lo - floor) / 2, lo + (hi - lo) / 4)


def sample_fault(
    fault_type: FaultType,
    rng: np.random.Generator,
    *,
    duration_hours: float,
    line: LineConfig,
    near_limit: bool = False,
) -> FaultConfig:
    """Draw one fault of ``fault_type`` with random target, magnitude and onset.

    ``near_limit``: draw the magnitude from ``near_limit_range`` (weaker faults around
    the detection limit, used for the harder diagnosis corpus).
    """
    spec = FAULT_SPECS[fault_type]
    lo, hi = near_limit_range(fault_type) if near_limit else spec.sample_range
    magnitude = float(rng.uniform(lo, hi))
    sign = float(rng.choice([-1.0, 1.0])) if spec.signed else 1.0
    n_batches = int(duration_hours * 60 / line.takt_minutes / line.batch_size)
    target = _sample_target(spec.target_kind, rng, max(n_batches, 3))
    onset = float(rng.uniform(0.3, 0.5) * duration_hours) if spec.has_time_window else 0.0
    return FaultConfig(
        type=fault_type,
        target=target,
        magnitude=round(sign * magnitude, 6),
        onset_hours=round(onset, 3),
    )


def generate_family(
    *,
    family: str,
    seed: int,
    n_per_fault: int,
    start: datetime,
    duration_hours: float,
    fault_types: tuple[FaultType, ...] = tuple(FaultType),
    near_limit: bool = False,
) -> list[Scenario]:
    """A reproducible family of single-fault scenarios (``n_per_fault`` per fault type)."""
    scenarios: list[Scenario] = []
    line = LineConfig()
    for fault_type in fault_types:
        rng = stream(seed, f"family:{family}:{fault_type}")
        for i in range(n_per_fault):
            fault = sample_fault(
                fault_type, rng, duration_hours=duration_hours, line=line, near_limit=near_limit
            )
            scenarios.append(
                Scenario(
                    name=f"{family}_{fault_type}_{i:03d}",
                    seed=int(rng.integers(0, 2**31 - 1)),
                    start=start,
                    duration_hours=duration_hours,
                    line=line,
                    faults=(fault,),
                )
            )
    return scenarios


def generate_pair_family(
    *,
    family: str,
    seed: int,
    n_pairs: int,
    start: datetime,
    duration_hours: float,
    fault_types: tuple[FaultType, ...] = tuple(FaultType),
    near_limit: bool = False,
) -> list[Scenario]:
    """A reproducible family of two-fault scenarios: two distinct fault types per scenario,
    drawn uniformly from ``fault_types``, each with its own target, magnitude and onset.

    Scenario names do not name the faults (``<family>_pair_<i>``).
    """
    if len(fault_types) < 2:
        raise ValueError("need at least two fault types for pairs")
    line = LineConfig()
    rng = stream(seed, f"pairs:{family}")
    scenarios: list[Scenario] = []
    for i in range(n_pairs):
        a, b = rng.choice(len(fault_types), size=2, replace=False)
        faults = tuple(
            sample_fault(
                fault_types[int(k)],
                rng,
                duration_hours=duration_hours,
                line=line,
                near_limit=near_limit,
            )
            for k in (a, b)
        )
        scenarios.append(
            Scenario(
                name=f"{family}_pair_{i:03d}",
                seed=int(rng.integers(0, 2**31 - 1)),
                start=start,
                duration_hours=duration_hours,
                line=line,
                faults=faults,
            )
        )
    return scenarios


def write_family(scenarios: list[Scenario], out_dir: Path) -> dict[str, Any]:
    """Write one YAML per scenario plus ``manifest.json`` (per-file and overall SHA-256)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for sc in scenarios:
        path = out_dir / f"{sc.name}.yaml"
        text = scenario_to_yaml(sc)
        path.write_text(text, encoding="utf-8")
        files[path.name] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    overall = hashlib.sha256(
        "\n".join(f"{k} {v}" for k, v in sorted(files.items())).encode("utf-8")
    ).hexdigest()
    manifest = {"n_scenarios": len(scenarios), "family_sha256": overall, "files": files}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
