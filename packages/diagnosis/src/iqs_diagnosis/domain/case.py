"""Diagnosis case and symptom models (pure data, no I/O).

A ``DiagnosisCase`` is what the agent is given: the case text plus a pointer to the
observable process data. It never carries a cause label; labels live in
``iqs_diagnosis.evaluation.labels`` and are joined only by the evaluation harness.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DiagnosisCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    source: str = Field(min_length=1)  # simulator / nhtsa / complaint / vision / manual
    title: str
    description: str
    line_id: str | None = None
    alarm_ts: datetime | None = None
    window_start: datetime | None = None
    baseline_hours: float | None = None
    # Observable process data for this case (read-only); None for text-only cases.
    observed_dir: Path | None = None
    reported_symptoms: tuple[dict[str, Any], ...] = ()
    # Step 7: vision-originated cases (source="vision") point back to their cluster.
    # linkage="synthetic_simulator": the parts behind the images are simulated parts linked
    # to images by iqs_sim.linker, not plant data.
    source_cluster_id: str | None = None
    linkage: str | None = None


class SymptomObservation(BaseModel):
    """One symptom code from ``configs/diagnosis/symptoms.yaml`` with its evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    # Where it was seen (machine, batch, shift) and the numbers behind it.
    evidence: dict[str, Any] = Field(default_factory=dict)


class SymptomProfile(BaseModel):
    """Descriptive summary of a case: discrete symptoms plus the numeric feature vector.

    The features are descriptive statistics (reference period vs affected period). They
    are not hypothesis tests; statistical testing is the scientist agent's job (step 5b).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    symptoms: tuple[SymptomObservation, ...]
    features: dict[str, float]

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(s.code for s in self.symptoms)

    def text(self) -> str:
        """Plain-text rendering used by keyword retrieval (codes + where they occur)."""
        parts: list[str] = []
        for s in self.symptoms:
            parts.append(s.code)
            where = s.evidence.get("where")
            if isinstance(where, str):
                parts.append(f"{s.code}@{where}")
        return " ".join(parts)
