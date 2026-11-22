"""GET /v1/calibrations[/{category}] and POST /v1/calibrations (step 2e).

Thresholds are recomputed from the per-image scores saved by scripts/calibrate.py
(scores.json), so the API never needs the calibration images. A POST activates the new
threshold in memory for subsequent inspections; it is not written to disk (models are
mounted read-only in Docker). The persistent calibration is the calibration.json written by
scripts/calibrate.py.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from iqs_vision.calibration.calibrate import DEFAULT_DELTA, build_calibration, load_scores
from iqs_vision.calibration.conformal import InsufficientCalibrationError
from iqs_vision.inference.registry import ModelInfo, ModelRegistry, UnknownModelError


class CalibrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1, max_length=64)
    alpha: float | None = Field(
        default=None, gt=0, lt=1, description="target false-alarm rate; default: policy D1"
    )
    delta: float = Field(default=DEFAULT_DELTA, gt=0, lt=1)


class CalibrationSummary(BaseModel):
    category: str
    threshold: float | None
    threshold_method: str | None
    alpha: float | None
    n_calib: int | None
    k: int | None
    source: str | None = Field(description='"file", "api" (in memory) or null (provisional)')
    persisted: bool


class CalibrationDetail(CalibrationSummary):
    calibration: dict[str, Any] | None


def _summary(info: ModelInfo) -> dict[str, Any]:
    cal = info.calibration
    return {
        "category": info.category,
        "threshold": info.threshold,
        "threshold_method": info.threshold_method,
        "alpha": info.threshold_alpha,
        "n_calib": cal["n_calib"] if cal else None,
        "k": cal["active"]["k"] if cal else None,
        "source": info.calibration_source,
        "persisted": info.calibration_source == "file",
    }


def build_router(registry: ModelRegistry) -> APIRouter:
    router = APIRouter(tags=["calibrations"])

    def _info(category: str) -> ModelInfo:
        try:
            return registry.info(category)
        except UnknownModelError:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"no model for category {category!r}; available: {registry.categories()}",
            ) from None

    @router.get("/calibrations", response_model=list[CalibrationSummary])
    def list_calibrations() -> list[dict[str, Any]]:
        return [_summary(registry.info(c)) for c in registry.categories()]

    @router.get("/calibrations/{category}", response_model=CalibrationDetail)
    def get_calibration(category: str) -> dict[str, Any]:
        info = _info(category)
        return _summary(info) | {"calibration": info.calibration}

    @router.post(
        "/calibrations",
        response_model=CalibrationDetail,
        responses={
            404: {"description": "No model, or no saved scores for this category"},
            422: {"description": "Too few calibration images for this alpha"},
        },
    )
    def calibrate(req: CalibrationRequest) -> dict[str, Any]:
        info = _info(req.category)
        scores = registry.scores_file(req.category)
        if not scores.is_file():
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"no saved scores for {req.category!r}; run scripts/calibrate.py first",
            )
        meta, records = load_scores(scores)
        try:
            cal = build_calibration(records, meta, alpha=req.alpha, delta=req.delta)
        except InsufficientCalibrationError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        info = registry.set_calibration(info.category, cal)
        return _summary(info) | {"calibration": cal}

    return router
