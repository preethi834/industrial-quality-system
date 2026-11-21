"""Near-limit magnitudes and two-fault families (harder diagnosis corpus)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from iqs_sim.faults import FAULT_SPECS, FaultType
from iqs_sim.rng import stream
from iqs_sim.scenarios import (
    LineConfig,
    generate_family,
    generate_pair_family,
    near_limit_range,
    sample_fault,
)

START = datetime(2026, 1, 5, 6, tzinfo=UTC)


@pytest.mark.parametrize("ft", list(FaultType))
def test_near_limit_band_is_weaker_than_standard(ft: FaultType) -> None:
    spec = FAULT_SPECS[ft]
    lo, hi = near_limit_range(ft)
    assert spec.min_abs_magnitude < lo < spec.sample_range[0] < hi < spec.sample_range[1]
    rng = stream(1, f"t:{ft}")
    for _ in range(20):
        f = sample_fault(ft, rng, duration_hours=72, line=LineConfig(), near_limit=True)
        assert lo - 1e-9 <= abs(f.magnitude) <= hi + 1e-9


def test_near_limit_family_is_deterministic() -> None:
    kw = dict(family="nl", seed=3, n_per_fault=2, start=START, duration_hours=72.0)
    a = generate_family(**kw, near_limit=True)  # type: ignore[arg-type]
    b = generate_family(**kw, near_limit=True)  # type: ignore[arg-type]
    std = generate_family(**kw)  # type: ignore[arg-type]
    assert a == b and a != std


def test_pair_family() -> None:
    types = tuple(f for f in FaultType if f is not FaultType.DRIVE_POWER_DEVIATION)
    pairs = generate_pair_family(
        family="pf", seed=9, n_pairs=40, start=START, duration_hours=72.0, fault_types=types
    )
    assert len(pairs) == 40
    assert pairs == generate_pair_family(
        family="pf", seed=9, n_pairs=40, start=START, duration_hours=72.0, fault_types=types
    )
    for sc in pairs:
        assert len(sc.faults) == 2 and sc.faults[0].type != sc.faults[1].type
        assert all(f.type in types for f in sc.faults)
        assert not any(f.value in sc.name for f in FaultType)
    with pytest.raises(ValueError):
        generate_pair_family(
            family="x",
            seed=1,
            n_pairs=1,
            start=START,
            duration_hours=72.0,
            fault_types=(FaultType.FIXTURE_OFFSET,),
        )
