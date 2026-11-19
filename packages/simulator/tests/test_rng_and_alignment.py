"""Random streams, and the 1:1 link between fault types and taxonomy leaves."""

from __future__ import annotations

import numpy as np
import pytest

from iqs_common.taxonomy import Taxonomy
from iqs_sim.faults import FAULT_SPECS, OUTCOME_COLUMNS, FaultType, cause_code_for
from iqs_sim.rng import stream
from iqs_sim.scenarios import Scenario


def test_streams_are_reproducible_and_independent() -> None:
    a1, a2, b = stream(7, "a"), stream(7, "a"), stream(7, "b")
    x1, x2, y = a1.random(5), a2.random(5), b.random(5)
    assert np.array_equal(x1, x2)
    assert not np.array_equal(x1, y)
    assert not np.array_equal(stream(8, "a").random(5), x1)
    with pytest.raises(ValueError):
        stream(-1, "a")


def test_fault_types_equal_taxonomy_simulator_faults(taxonomy: Taxonomy) -> None:
    leaves = [taxonomy.get(code) for code in taxonomy.leaf_codes]
    names = [c.simulator_fault for c in leaves]
    assert None not in names, "every leaf cause needs a simulator_fault"
    assert len(set(names)) == len(names)
    assert set(names) == {f.value for f in FaultType}


@pytest.mark.parametrize("fault", list(FaultType))
def test_each_fault_maps_to_exactly_one_cause(taxonomy: Taxonomy, fault: FaultType) -> None:
    code = cause_code_for(taxonomy, fault.value)
    assert taxonomy.get(code).simulator_fault == fault.value


def test_unknown_fault_name_is_rejected(taxonomy: Taxonomy) -> None:
    with pytest.raises(ValueError, match="maps to 0"):
        cause_code_for(taxonomy, "no_such_fault")


def test_every_fault_declares_signals_disjoint_from_outcomes() -> None:
    assert set(FAULT_SPECS) == set(FaultType)
    for spec in FAULT_SPECS.values():
        assert spec.signals, "a fault must change at least one signal besides the outcome"
        assert not spec.signals & OUTCOME_COLUMNS


def test_one_reference_scenario_per_fault(reference_scenarios: dict[str, Scenario]) -> None:
    assert "baseline" in reference_scenarios
    assert reference_scenarios["baseline"].faults == ()
    covered = [f.type for sc in reference_scenarios.values() for f in sc.faults]
    assert sorted(covered) == sorted(FaultType)
    for sc in reference_scenarios.values():
        assert len(sc.faults) <= 1, "reference scenarios are single-fault"
