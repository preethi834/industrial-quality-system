"""Symptom vocabulary: config validity and rule evaluation (no simulator needed)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from iqs_diagnosis.symptoms.simulator_profile import FEATURE_NAMES
from iqs_diagnosis.symptoms.vocabulary import Vocabulary, load_vocabulary

from .conftest import VOCABULARY_PATH


def _zero() -> dict[str, float]:
    return dict.fromkeys(FEATURE_NAMES, 0.0)


def test_config_loads_with_unique_codes() -> None:
    vocab = load_vocabulary(VOCABULARY_PATH)
    assert len(vocab.codes) == len(set(vocab.codes)) >= 15
    assert "reject_rate_increase" in vocab.codes


def test_every_rule_feature_exists() -> None:
    vocab = load_vocabulary(VOCABULARY_PATH)
    used = {f for s in vocab.symptoms for f, _, _ in s.all}
    assert used <= set(FEATURE_NAMES)


def test_no_symptom_code_names_a_cause() -> None:
    """Symptoms are observations; cause codes (UPPER.DOTTED) must not appear as symptoms."""
    vocab = load_vocabulary(VOCABULARY_PATH)
    assert all(c == c.lower() and "." not in c for c in vocab.codes)


@pytest.mark.parametrize(
    "bad",
    [
        {"code": "x", "name_en": "X", "description": "d", "all": [["no_such", ">=", 1]]},
        {
            "code": "x",
            "name_en": "X",
            "description": "d",
            "source": "case",
            "all": [["temp_z_max", ">=", 1]],
        },
        {"code": "x", "name_en": "X", "description": "d"},
        {"code": "Bad.Code", "name_en": "X", "description": "d", "all": [["temp_z_max", ">=", 1]]},
    ],
)
def test_invalid_symptoms_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Vocabulary.model_validate({"version": "t", "symptoms": [bad]})


def test_duplicate_codes_rejected() -> None:
    s = {"code": "a", "name_en": "A", "description": "d", "all": [["temp_z_max", ">=", 1]]}
    with pytest.raises(ValidationError, match="duplicate"):
        Vocabulary.model_validate({"version": "t", "symptoms": [s, s]})


def test_observe_applies_all_conditions_and_where() -> None:
    vocab = load_vocabulary(VOCABULARY_PATH)
    f = _zero()
    f["temp_z_max"], f["temp_z_min"] = 5.0, 0.0
    obs = vocab.observe(f, {"temp_z_max": "M3"}, {"reject_rate_increase"})
    codes = [o.code for o in obs]
    assert codes == ["reject_rate_increase", "machine_temperature_rise"]
    assert obs[1].evidence["where"] == "M3"
    assert obs[1].evidence["temp_z_max"] == 5.0
    # All machines warm: the single-machine symptom no longer holds.
    f["temp_z_min"] = 3.0
    codes = [o.code for o in vocab.observe(f, {}, set())]
    assert codes == ["temperature_rise_all_machines"]


def test_case_symptom_only_from_case(tmp_path: Path) -> None:
    vocab = load_vocabulary(VOCABULARY_PATH)
    assert vocab.observe(_zero(), {}, set()) == ()
