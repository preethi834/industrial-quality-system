"""Calibration runs: frozen, versioned thresholds for conformal cause sets.

A calibration run stores everything needed to reproduce and audit its thresholds: the
scorer parameters (fitted on a disjoint *fit* set), the calibration case IDs, their
scores and the resulting thresholds per method. It is written once and never updated
(§2.8 point 5: confirmed cases are not appended automatically; a new run gets a new ID).

Disjointness (§2.8 point 1) is checked when a run is built: calibration cases must not
be library cases (retrieval could find themselves) or fit cases (vote smoothing and
novelty threshold were chosen on those).
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from iqs_diagnosis.conformal.scores import Method, prediction_set, score
from iqs_diagnosis.conformal.split import conformal_threshold, conformal_threshold_pac
from iqs_diagnosis.scoring.ranker import RankerParams


class LeakageError(ValueError):
    pass


def hash_half(case_id: str, salt: str) -> str:
    """Deterministic 50/50 split of a case: "fit" or "cal" (by SHA-256 of salt + id)."""
    h = hashlib.sha256(f"{salt}\x1f{case_id}".encode()).digest()
    return "fit" if h[0] % 2 == 0 else "cal"


def check_disjoint(
    calibration_ids: Collection[str],
    *,
    library_ids: Collection[str],
    fit_ids: Collection[str],
) -> None:
    cal = set(calibration_ids)
    if len(cal) != len(calibration_ids):
        raise LeakageError("duplicate calibration case ids")
    for name, other in (("library", library_ids), ("fit", fit_ids)):
        overlap = cal & set(other)
        if overlap:
            raise LeakageError(f"{len(overlap)} calibration cases are also {name} cases")


class MethodThreshold(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Method
    qhat: float
    k: int
    delta: float | None = None


class CalibrationRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(pattern=r"^[a-z0-9_]+$")
    corpus: str
    source: str = "simulator"  # data source (Mondrian group); real cases get their own run
    alpha: float
    delta: float
    n_calibration: int
    thresholds: dict[str, MethodThreshold]  # "lac", "aps", "lac_pac", "aps_pac"
    scorer: RankerParams
    vocabulary_version: str
    taxonomy_version: str
    calibration_case_ids: list[str]
    fit_case_ids: list[str]
    calibration_scores: dict[str, list[float]]  # method -> sorted scores
    note: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CalibrationRun:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def calibrate(
    *,
    run_id: str,
    corpus: str,
    probabilities: Sequence[Mapping[str, float]],
    truths: Sequence[str],
    calibration_ids: Sequence[str],
    fit_ids: Sequence[str],
    library_ids: Collection[str],
    scorer: RankerParams,
    vocabulary_version: str,
    taxonomy_version: str,
    alpha: float = 0.1,
    delta: float = 0.1,
    note: str = "",
) -> CalibrationRun:
    if not len(probabilities) == len(truths) == len(calibration_ids):
        raise ValueError("one probability vector and one truth per calibration case")
    check_disjoint(calibration_ids, library_ids=library_ids, fit_ids=fit_ids)
    thresholds: dict[str, MethodThreshold] = {}
    scores_out: dict[str, list[float]] = {}
    for m in Method:
        s = [score(m, p, y) for p, y in zip(probabilities, truths, strict=True)]
        scores_out[m.value] = sorted(s)
        t = conformal_threshold(s, alpha)
        thresholds[m.value] = MethodThreshold(method=m, qhat=t.qhat, k=t.k)
        tp = conformal_threshold_pac(s, alpha, delta)
        thresholds[f"{m.value}_pac"] = MethodThreshold(method=m, qhat=tp.qhat, k=tp.k, delta=delta)
    return CalibrationRun(
        run_id=run_id,
        corpus=corpus,
        alpha=alpha,
        delta=delta,
        n_calibration=len(truths),
        thresholds=thresholds,
        scorer=scorer,
        vocabulary_version=vocabulary_version,
        taxonomy_version=taxonomy_version,
        calibration_case_ids=sorted(calibration_ids),
        fit_case_ids=sorted(fit_ids),
        calibration_scores=scores_out,
        note=note,
    )


class ConformalPredictor:
    """Cause set for one case from a frozen calibration run and a chosen threshold."""

    def __init__(self, run: CalibrationRun, variant: str = "lac") -> None:
        if variant not in run.thresholds:
            raise ValueError(f"unknown threshold variant {variant!r}")
        self.run = run
        self.variant = variant
        self.threshold = run.thresholds[variant]

    def cause_set(self, probabilities: Mapping[str, float]) -> list[str]:
        return prediction_set(self.threshold.method, probabilities, self.threshold.qhat)
