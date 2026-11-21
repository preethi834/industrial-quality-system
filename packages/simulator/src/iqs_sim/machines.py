"""Line layout: stations, machines and their nominal (observable) master data.

Line L1 has four stations in series::

    S1 turning (M1) -> S2 milling (M2A | M2B, parallel) -> S3 drilling (M3) -> S4 inspection (Q1)

S2 has two parallel machines so that machine-level faults (for example a fixture
offset on M2B) are distinguishable from station-level or line-level causes through
the part's routing. Q1 is a measuring station (dimension gauge + surface camera).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

LINE_ID = "L1"
STATIONS: tuple[str, ...] = ("S1", "S2", "S3", "S4")
CUTTING_STATIONS: tuple[str, ...] = ("S1", "S2", "S3")
INSPECTION_STATION = "S4"
GAUGE_ID = "Q1"
HALL_ID = "HALL"

# Validated process window: actual setpoints may differ from nominal by at most this
# relative amount. Outside it, METHOD.WRONG_PROCESS_PARAMETERS applies.
PARAMETER_WINDOW_REL = 0.10
STANDARD_PARAMETER_SET = "PS-STD"
CHANGED_PARAMETER_SET = "PS-ALT"

# Product and environment specification (observable plant knowledge).
NOMINAL_DIMENSION_MM = 50.0
DIMENSION_TOLERANCE_MM = 0.03
AMBIENT_SPEC_C: tuple[float, float] = (18.0, 26.0)
NOMINAL_HARDNESS_HB = 200.0
MATERIAL = "42CrMo4"


class MachineKind(StrEnum):
    CUTTING = "cutting"
    GAUGE = "gauge"


@dataclass(frozen=True)
class MachineSpec:
    machine_id: str
    station_id: str
    kind: MachineKind
    process: str
    cycle_minutes: float
    nominal_speed_rpm: float | None = None
    nominal_feed_mm_rev: float | None = None
    nominal_power_kw: float | None = None
    nominal_load_pct: float | None = None
    tool_life_parts: int | None = None
    tool_wear_limit_mm: float | None = None

    @property
    def speed_window(self) -> tuple[float, float] | None:
        if self.nominal_speed_rpm is None:
            return None
        return (
            self.nominal_speed_rpm * (1 - PARAMETER_WINDOW_REL),
            self.nominal_speed_rpm * (1 + PARAMETER_WINDOW_REL),
        )


LAYOUT: tuple[MachineSpec, ...] = (
    MachineSpec("M1", "S1", MachineKind.CUTTING, "turning", 1.6, 1800, 0.25, 6.0, 45, 120, 0.30),
    MachineSpec("M2A", "S2", MachineKind.CUTTING, "milling", 3.0, 1200, 0.15, 5.0, 42, 100, 0.30),
    MachineSpec("M2B", "S2", MachineKind.CUTTING, "milling", 3.0, 1200, 0.15, 5.0, 42, 100, 0.30),
    MachineSpec("M3", "S3", MachineKind.CUTTING, "drilling", 1.5, 2400, 0.10, 4.0, 40, 150, 0.25),
    MachineSpec(GAUGE_ID, "S4", MachineKind.GAUGE, "inspection", 0.5),
)

MACHINES: dict[str, MachineSpec] = {m.machine_id: m for m in LAYOUT}
CUTTING_MACHINES: tuple[str, ...] = tuple(
    m.machine_id for m in LAYOUT if m.kind is MachineKind.CUTTING
)


def machines_at(station_id: str) -> tuple[str, ...]:
    return tuple(m.machine_id for m in LAYOUT if m.station_id == station_id)


@dataclass
class ToolState:
    """Mutable wear state of the tool currently mounted on one cutting machine."""

    tool_number: int = 1
    wear_mm: float = 0.0
    parts_cut: int = 0
