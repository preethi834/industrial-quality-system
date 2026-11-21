"""Symptom extraction: observable data -> features -> symptom codes."""

from __future__ import annotations

from iqs_diagnosis.data.process_store import ProcessStore
from iqs_diagnosis.domain.case import DiagnosisCase, SymptomProfile
from iqs_diagnosis.symptoms.simulator_profile import compute_features
from iqs_diagnosis.symptoms.vocabulary import Vocabulary


def extract_profile(case: DiagnosisCase, vocabulary: Vocabulary) -> SymptomProfile:
    """Symptom profile of a case with observable process data (simulator cases)."""
    if case.observed_dir is None:
        raise ValueError(f"{case.case_id}: no observable process data")
    with ProcessStore(case.observed_dir) as store:
        f = compute_features(case, store)
    case_codes = {str(s.get("code")) for s in case.reported_symptoms}
    return SymptomProfile(
        case_id=case.case_id,
        symptoms=vocabulary.observe(f.values, f.where, case_codes),
        features=f.values,
    )
