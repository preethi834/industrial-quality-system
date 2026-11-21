"""Vision-originated cases (step 7): a ``new_defect_cluster`` event -> ``DiagnosisCase``.

The case ID is a UUIDv5 of the cluster ID, so one cluster can only ever open one case,
whatever happens to the event (ADR-015). The process data are the observed tables of the
simulator run the images were linked to (**synthetic linkage**); the case never reads the
linker manifest or ``ground_truth/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from iqs_common.events import NewDefectClusterPayload
from iqs_common.ids import deterministic_id
from iqs_diagnosis.domain.case import DiagnosisCase
from iqs_diagnosis.ingestion.simulator import OBSERVED_DIR

VISION_SOURCE: Final = "vision"
# Reported with the case; not in the symptom vocabulary, so retrieval and scoring ignore it
# (the vocabulary version recorded by the frozen calibration run stays unchanged).
VISUAL_CLUSTER_SYMPTOM: Final = "new_visual_defect_cluster"


class VisionCaseError(ValueError):
    """The event cannot become a diagnosable case (retried, then parked as failed)."""


def vision_case_id(cluster_id: object) -> str:
    return str(deterministic_id("vision-cluster", str(cluster_id)))


def case_from_cluster_event(
    payload: NewDefectClusterPayload, runs_root: Path | None, baseline_hours: float
) -> DiagnosisCase:
    if payload.simulator_run_id is None or payload.linkage != "synthetic_simulator":
        raise VisionCaseError(
            "cluster has no linked process data (no simulator_run_id); "
            "text-only vision cases need step 3e"
        )
    if runs_root is None:
        raise VisionCaseError("runs_root is not configured")
    observed = (runs_root / payload.simulator_run_id / OBSERVED_DIR).resolve()
    if runs_root.resolve() not in observed.parents:
        raise VisionCaseError("run directory escapes runs_root")
    if not observed.is_dir():
        raise VisionCaseError(f"no observed data for {payload.simulator_run_id}")
    symptom: dict[str, Any] = {
        "code": VISUAL_CLUSTER_SYMPTOM,
        "category": payload.category,
        "cluster_id": str(payload.cluster_id),
        "size": payload.size,
        "first_seen": payload.first_seen.isoformat(),
        "last_seen": payload.last_seen.isoformat(),
        "score_max": payload.score_max,
        "score_median": payload.score_median,
        "model_version": payload.model_version,
        "linkage": payload.linkage,
    }
    first_seen = payload.first_seen.strftime("%Y-%m-%d %H:%M")
    return DiagnosisCase(
        case_id=vision_case_id(payload.cluster_id),
        source=VISION_SOURCE,
        title=f"New visual defect cluster: {payload.category} ({payload.size} images)",
        description=(
            f"The vision discovery trigger found {payload.size} anomalous {payload.category} "
            f"images that match no known defect class, first seen {first_seen} UTC. "
            f"Process data: simulator run {payload.simulator_run_id} "
            "(SYNTHETIC image-to-part linkage, simulated process data)."
        ),
        line_id=payload.line_id,
        alarm_ts=payload.last_seen,
        window_start=payload.first_seen,
        baseline_hours=baseline_hours,
        observed_dir=observed,
        reported_symptoms=(symptom,),
        source_cluster_id=str(payload.cluster_id),
        linkage=payload.linkage,
    )
