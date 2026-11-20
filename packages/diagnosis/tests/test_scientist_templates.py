"""Scientist templates on the reference scenarios (effect) and the fault-free baseline (null).

The 1,000-simulation false-positive check per template (gate 2) is a separate script
(scripts/check_scientist_null.py); here one null run is a sanity check only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.data.process_store import ProcessStore
from iqs_diagnosis.ingestion.simulator import load_case
from iqs_diagnosis.scientist.templates import TEMPLATES, Window, _bonferroni

from .conftest import TAXONOMY_PATH

# reference scenario -> the template of its injected cause
OWN: dict[str, str] = {
    "tool_wear_m1": "defect_rate_by_tool_wear_level",
    "spindle_overload_m2a": "spindle_load_defective_vs_good",
    "coolant_loss_m3": "temperature_delta_defective_vs_good",
    "drive_fault_m1": "power_command_residual_defective_vs_good",
    "fixture_offset_m2b": "station_defect_rate_change_point",
    "bad_batch_b0008": "defect_rate_by_batch",
    "wrong_parameters_m3": "defect_rate_by_parameter_set",
    "gauge_drift_q1": "sensor_reference_residual_trend",
    "ambient_excursion": "ambient_temperature_defective_vs_good",
    "handling_error_shift_b": "defect_rate_by_shift",
}
TARGET = {
    "tool_wear_m1": "M1", "spindle_overload_m2a": "M2A", "coolant_loss_m3": "M3",
    "drive_fault_m1": "M1", "fixture_offset_m2b": "M2B", "bad_batch_b0008": "B0008",
    "wrong_parameters_m3": "M3", "handling_error_shift_b": "B",
}  # fmt: skip


def _window(store: ProcessStore, baseline_hours: float, start: pd.Timestamp) -> Window:
    t0 = store.query(
        "SELECT min(completed_ts) AS t FROM read_parquet(?)",
        [str(store.observed_dir / "parts.parquet")],
    )["t"][0]
    return Window(pd.Timestamp(t0).floor("h") + pd.Timedelta(hours=baseline_hours), start)


def test_every_taxonomy_template_exists() -> None:
    names = {c.test_template for c in load_taxonomy(TAXONOMY_PATH).causes}
    assert names == set(TEMPLATES) and set(OWN.values()) == set(TEMPLATES)


@pytest.mark.parametrize("scenario", sorted(OWN))
def test_own_template_supports_injected_cause(
    scenario: str, reference_runs: dict[str, Path]
) -> None:
    case = load_case(reference_runs[scenario])
    assert case is not None and case.window_start is not None
    with ProcessStore(case.observed_dir) as store:  # type: ignore[arg-type]
        w = _window(store, case.baseline_hours or 12, pd.Timestamp(case.window_start))
        r = TEMPLATES[OWN[scenario]](store, w)
    assert r.p_value is not None and r.p_value < 0.01, (scenario, r.p_value, r.reason)
    if scenario in TARGET:
        assert r.target == TARGET[scenario]
    assert r.sql.startswith("SELECT") and r.params and r.plot is not None
    json.dumps(r.model_dump(mode="json"))  # evidence is JSON-serialisable


def test_fault_free_run_supports_nothing_strongly(reference_runs: dict[str, Path]) -> None:
    obs = reference_runs["baseline"] / "observed"
    with ProcessStore(obs) as store:
        t0 = store.query(
            "SELECT min(completed_ts) AS t FROM read_parquet(?)", [str(obs / "parts.parquet")]
        )["t"][0]
        w = _window(store, 12, pd.Timestamp(t0).floor("h") + pd.Timedelta(hours=36))
        results = {n: f(store, w) for n, f in TEMPLATES.items()}
    for name, r in results.items():
        assert r.p_value is None or r.p_value > 0.001, (name, r.p_value)
        if r.p_value is None:
            assert r.reason


def test_bonferroni_helper() -> None:
    assert _bonferroni({"a": 0.01, "b": 0.5}) == (0.02, "a", 0.01)
    assert _bonferroni({"a": 0.6, "b": 0.7})[0] == 1.0
