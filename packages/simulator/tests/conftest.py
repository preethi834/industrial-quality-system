"""Shared fixtures for iqs_sim tests.

Simulations are cached per (scenario name, seed, duration) for the whole session:
the model is a pure function of the scenario, so sharing results is safe and keeps
the suite fast. Tests use a 48 h horizon (reference faults start at hour 24).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from iqs_common.taxonomy import Taxonomy, load_taxonomy
from iqs_sim.line import SimulationResult, simulate
from iqs_sim.scenarios import Scenario, load_scenario

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_DIR = REPO_ROOT / "configs" / "simulator" / "scenarios"
TAXONOMY_PATH = REPO_ROOT / "configs" / "taxonomy" / "causes.yaml"
TEST_DURATION_H = 48.0

_cache: dict[tuple[str, int, float, int], SimulationResult] = {}


@pytest.fixture(scope="session")
def taxonomy() -> Taxonomy:
    return load_taxonomy(TAXONOMY_PATH)


@pytest.fixture(scope="session")
def reference_scenarios() -> dict[str, Scenario]:
    """All scenario files, shortened to the test horizon."""
    return {
        p.stem: load_scenario(p).with_overrides(duration_hours=TEST_DURATION_H)
        for p in sorted(SCENARIO_DIR.glob("*.yaml"))
    }


@pytest.fixture(scope="session")
def run_sim() -> Callable[[Scenario], SimulationResult]:
    def _run(scenario: Scenario) -> SimulationResult:
        key = (scenario.name, scenario.seed, scenario.duration_hours, hash(scenario.faults))
        if key not in _cache:
            _cache[key] = simulate(scenario)
        return _cache[key]

    return _run
