"""Scenario validation and scenario families."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from iqs_sim.faults import FaultConfig, FaultType
from iqs_sim.scenarios import (
    Scenario,
    generate_family,
    load_scenario,
    scenario_to_yaml,
    write_family,
)

START = datetime(2026, 1, 5, 6, tzinfo=UTC)


def _scenario(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "s",
        "seed": 1,
        "start": "2026-01-05T06:00:00Z",
        "duration_hours": 24,
        "faults": [],
    }
    data.update(overrides)
    return data


def _fault(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "type": "spindle_overload",
        "target": "M1",
        "magnitude": 1.5,
        "onset_hours": 2,
    }
    data.update(overrides)
    return data


def test_valid_scenario_round_trips_through_yaml(tmp_path: Path) -> None:
    sc = Scenario.model_validate(_scenario(faults=[_fault()]))
    path = tmp_path / "s.yaml"
    path.write_text(scenario_to_yaml(sc), encoding="utf-8")
    assert load_scenario(path) == sc
    assert sc.start == START


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        (_fault(target="M9"), "target"),
        (_fault(target="Q1"), "target"),
        (_fault(magnitude=5.0), "outside"),
        (_fault(magnitude=1.05), ">= 1.1"),
        (_fault(type="drive_power_deviation", magnitude=0.01), "magnitude"),
        (_fault(type="sensor_calibration_drift", target="M1", magnitude=0.001), "target"),
        (_fault(type="handling_error_on_shift", target="D", magnitude=0.2), "target"),
        (_fault(type="bad_material_batch", target="B0003", magnitude=40), "onset"),
        (_fault(type="bad_material_batch", target="X1", magnitude=40, onset_hours=0), "batch"),
        (_fault(type="no_such_fault"), "type"),
        (_fault(extra=1), "extra"),
        (_fault(onset_hours=30), "onset_hours"),
    ],
)
def test_invalid_faults_are_rejected(fault: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Scenario.model_validate(_scenario(faults=[fault]))


def test_scenario_requires_timezone_and_known_keys() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        Scenario.model_validate(_scenario(start="2026-01-05T06:00:00"))
    with pytest.raises(ValidationError, match="extra"):
        Scenario.model_validate(_scenario(unknown=True))
    with pytest.raises(ValidationError, match="name"):
        Scenario.model_validate(_scenario(name="Bad Name"))


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "x.yaml"
    path.write_text("- 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_scenario(path)


def test_fault_activity_window() -> None:
    f = FaultConfig.model_validate(_fault(onset_hours=1, duration_hours=2))
    assert not f.active(59.9)
    assert f.active(60.0)
    assert f.active(179.9)
    assert not f.active(180.0)


def test_overrides_and_without_faults() -> None:
    sc = Scenario.model_validate(_scenario(faults=[_fault()]))
    o = sc.with_overrides(seed=9, duration_hours=10)
    assert (o.seed, o.duration_hours, o.faults) == (9, 10, sc.faults)
    assert sc.without_faults().faults == ()


def test_family_is_reproducible_valid_and_balanced(tmp_path: Path) -> None:
    kwargs: dict[str, Any] = {
        "family": "fam",
        "seed": 3,
        "n_per_fault": 4,
        "start": START,
        "duration_hours": 72.0,
    }
    a, b = generate_family(**kwargs), generate_family(**kwargs)
    assert a == b
    assert len(a) == 4 * len(FaultType)
    assert len({sc.name for sc in a}) == len(a)
    for ft in FaultType:
        assert sum(sc.faults[0].type is ft for sc in a) == 4
    assert generate_family(**{**kwargs, "seed": 4}) != a

    m1 = write_family(a, tmp_path / "one")
    m2 = write_family(b, tmp_path / "two")
    assert m1 == m2
    manifest = json.loads((tmp_path / "one" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_scenarios"] == len(a)
    reloaded = load_scenario(tmp_path / "one" / f"{a[0].name}.yaml")
    assert reloaded == a[0]


def test_family_can_be_limited_to_some_faults() -> None:
    fam = generate_family(
        family="f",
        seed=1,
        n_per_fault=2,
        start=START,
        duration_hours=72.0,
        fault_types=(FaultType.FIXTURE_OFFSET,),
    )
    assert [sc.faults[0].type for sc in fam] == [FaultType.FIXTURE_OFFSET] * 2
