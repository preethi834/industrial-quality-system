"""Synthetic symptom profiles for fast retrieval / scoring tests (no simulator)."""

from __future__ import annotations

import numpy as np

from iqs_diagnosis.domain.case import SymptomObservation, SymptomProfile
from iqs_diagnosis.retrieval.base import LibraryCase
from iqs_diagnosis.symptoms.simulator_profile import FEATURE_NAMES

# cause -> (feature that is high, symptom code shown)
CLUSTERS: dict[str, tuple[str, str]] = {
    "MACHINE.TOOL_WEAR": ("wear_rate_log2_ratio_max", "tool_wear_rate_rise"),
    "MACHINE.DRIVE_FAULT": ("power_residual_z_max", "power_deviation"),
    "MATERIAL.BAD_BATCH": ("load_batch_z_max", "batch_load_rise"),
    "MEASUREMENT.SENSOR_DRIFT": ("gauge_check_shift_um", "gauge_check_drift"),
}


def profile(
    case_id: str, cause: str, rng: np.random.Generator, level: float = 5.0
) -> SymptomProfile:
    feature, code = CLUSTERS[cause]
    f = {n: float(rng.normal(0.0, 0.3)) for n in FEATURE_NAMES}
    f[feature] = level + float(rng.normal(0.0, 0.5))
    return SymptomProfile(
        case_id=case_id,
        symptoms=(SymptomObservation(code="reject_rate_increase"), SymptomObservation(code=code)),
        features=f,
    )


def library(causes: list[str], n_per: int, seed: int = 0) -> list[LibraryCase]:
    rng = np.random.default_rng(seed)
    return [LibraryCase(profile(f"L-{c}-{i}", c, rng), c) for c in causes for i in range(n_per)]
