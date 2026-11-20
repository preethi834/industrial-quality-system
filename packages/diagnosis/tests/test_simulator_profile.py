"""Symptom profiles of the 11 reference scenarios, and the no-leak rules around them.

The vocabulary thresholds were set by looking at these same reference runs, so the
signature test is a regression check (each fault keeps its readable signature), not
an accuracy claim. Accuracy is measured on separate scenario families (step 3d).
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import duckdb
import pytest

from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.data.process_store import ProcessStore
from iqs_diagnosis.domain.case import SymptomProfile
from iqs_diagnosis.ingestion.simulator import load_case
from iqs_diagnosis.symptoms import extract_profile
from iqs_diagnosis.symptoms.simulator_profile import FEATURE_NAMES
from iqs_diagnosis.symptoms.vocabulary import Vocabulary, load_vocabulary

from .conftest import TAXONOMY_PATH, VOCABULARY_PATH

# scenario -> (symptoms that must appear, with location if given; symptoms that must not)
SIGNATURES: dict[str, tuple[dict[str, str | None], set[str]]] = {
    "ambient_excursion": (
        {"temperature_rise_all_machines": None, "ambient_temperature_high": None},
        {"machine_temperature_rise"},
    ),
    "bad_batch_b0008": (
        {"rejects_concentrate_batch": "B0008", "batch_load_rise": "B0008"},
        {"setpoint_change", "power_deviation"},
    ),
    "coolant_loss_m3": (
        {"machine_temperature_rise": "M3"},
        {"temperature_rise_all_machines", "ambient_temperature_high", "spindle_load_rise"},
    ),
    "drive_fault_m1": ({"power_deviation": "M1"}, {"setpoint_change"}),
    "fixture_offset_m2b": (
        {"dimension_shift_one_machine": "M2B", "reject_reason_dimension": None},
        {"gauge_check_drift", "rejects_concentrate_shift"},
    ),
    "gauge_drift_q1": (
        {"gauge_check_drift": None, "dimension_mean_shift": None},
        {"dimension_shift_one_machine"},
    ),
    "handling_error_shift_b": (
        {"rejects_concentrate_shift": "B", "dimension_spread_increase": None},
        {"gauge_check_drift", "dimension_shift_one_machine"},
    ),
    "spindle_overload_m2a": ({"spindle_load_rise": "M2A"}, {"tool_wear_rate_rise"}),
    "tool_wear_m1": (
        {"tool_wear_rate_rise": "M1", "vibration_rise": "M1"},
        {"power_deviation", "setpoint_change"},
    ),
    "wrong_parameters_m3": ({"setpoint_change": "M3"}, {"power_deviation"}),
}


@pytest.fixture(scope="module")
def vocab() -> Vocabulary:
    return load_vocabulary(VOCABULARY_PATH)


@pytest.fixture(scope="module")
def profiles(reference_runs: dict[str, Path], vocab: Vocabulary) -> dict[str, SymptomProfile]:
    out: dict[str, SymptomProfile] = {}
    for name, run in reference_runs.items():
        case = load_case(run)
        if case is not None:
            out[name] = extract_profile(case, vocab)
    return out


def test_baseline_raises_no_case(reference_runs: dict[str, Path]) -> None:
    assert load_case(reference_runs["baseline"]) is None


def test_every_fault_scenario_has_a_case(profiles: dict[str, SymptomProfile]) -> None:
    assert set(profiles) == set(SIGNATURES)


@pytest.mark.parametrize("scenario", sorted(SIGNATURES))
def test_reference_signature(scenario: str, profiles: dict[str, SymptomProfile]) -> None:
    must, must_not = SIGNATURES[scenario]
    p = profiles[scenario]
    by_code = {s.code: s for s in p.symptoms}
    assert p.codes[0] == "reject_rate_increase"
    for code, where in must.items():
        assert code in by_code, f"{scenario}: missing {code}; got {p.codes}"
        if where is not None:
            assert by_code[code].evidence.get("where") == where
    assert not (must_not & set(by_code)), f"{scenario}: unexpected {must_not & set(by_code)}"


def test_features_complete_and_finite(profiles: dict[str, SymptomProfile]) -> None:
    for p in profiles.values():
        assert list(p.features) == list(FEATURE_NAMES)
        assert all(math.isfinite(x) for x in p.features.values())


def test_profile_is_deterministic(reference_runs: dict[str, Path], vocab: Vocabulary) -> None:
    case = load_case(reference_runs["tool_wear_m1"])
    assert case is not None
    assert extract_profile(case, vocab) == extract_profile(case, vocab)


def test_profile_needs_no_ground_truth(
    reference_runs: dict[str, Path], vocab: Vocabulary, tmp_path: Path
) -> None:
    src = reference_runs["fixture_offset_m2b"]
    run = tmp_path / "run"
    shutil.copytree(src / "observed", run / "observed")
    shutil.copytree(src / "cases", run / "cases")
    case = load_case(run)
    original = load_case(src)
    assert case is not None and original is not None
    assert extract_profile(case, vocab) == extract_profile(original, vocab)


def test_profile_contains_no_ground_truth_tokens(profiles: dict[str, SymptomProfile]) -> None:
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    tokens = {c.code.lower() for c in taxonomy.causes} | {
        str(c.simulator_fault) for c in taxonomy.causes if c.simulator_fault
    }
    tokens |= {"onset", "magnitude", "scenario", "seed"}
    for p in profiles.values():
        text = json.dumps(p.model_dump(mode="json")).lower()
        assert not [t for t in tokens if t in text]


def test_process_store_cannot_read_ground_truth(reference_runs: dict[str, Path]) -> None:
    run = reference_runs["tool_wear_m1"].resolve()
    with ProcessStore(run / "observed") as store:
        assert len(store.table("parts")) > 0
        with pytest.raises(duckdb.Error):
            store.query("SELECT * FROM read_parquet(?)", [str(run / "ground_truth/faults.parquet")])
        with pytest.raises(duckdb.Error):
            store.query("SET enable_external_access = true")
        with pytest.raises(ValueError):
            store.table("faults")


def test_process_store_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ProcessStore(tmp_path / "nope")
