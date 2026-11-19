"""POST /v1/inspections (image -> score, heatmap, decision) and GET /v1/models."""

from __future__ import annotations

import time
from typing import Annotated, Literal

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from iqs_common.ids import uuid7
from iqs_common.log import get_logger
from iqs_vision.data.preprocess import InvalidImageError, decode_image, image_to_tensor
from iqs_vision.explain.anomaly_map import normalize_map, overlay, png_base64
from iqs_vision.explain.vlm import VLMDescriber
from iqs_vision.inference.registry import ModelRegistry, UnknownModelError

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/bmp"}
THRESHOLD_NOTE = (
    "provisional: largest anomaly score among held-out good calibration images; "
    "not a conformal guarantee (step 2)"
)

_log = get_logger("iqs_vision.inspections")


class ModelSummary(BaseModel):
    category: str
    detector: str
    version: str
    protocol: str
    threshold: float | None
    threshold_method: str | None
    mlflow_run_id: str | None
    image_auroc: float | None = None
    pixel_auroc: float | None = None


class InspectionResult(BaseModel):
    inspection_id: str
    category: str
    model: ModelSummary
    score: float = Field(description="Image anomaly score; higher = more anomalous")
    threshold: float | None
    threshold_note: str
    decision: Literal["anomalous", "normal", "undecided"]
    heatmap_png_base64: str = Field(description="Anomaly map overlaid on the image (PNG)")
    heatmap_scale: str
    map_max: float
    image_width: int
    image_height: int
    latency_ms: float
    description: str | None = Field(
        default=None, description="VLM text; absent when no VLM is configured"
    )


def _summary(info: object) -> ModelSummary:
    from iqs_vision.inference.registry import ModelInfo

    assert isinstance(info, ModelInfo)
    return ModelSummary(
        category=info.category,
        detector=info.detector,
        version=info.version,
        protocol=info.protocol,
        threshold=info.threshold,
        threshold_method=info.threshold_method,
        mlflow_run_id=info.mlflow_run_id,
        image_auroc=info.metrics.get("image_auroc"),
        pixel_auroc=info.metrics.get("pixel_auroc"),
    )


def build_router(
    registry: ModelRegistry, max_upload_bytes: int, describer: VLMDescriber | None
) -> APIRouter:
    router = APIRouter(tags=["inspections"])

    @router.get("/models", response_model=list[ModelSummary])
    def list_models() -> list[ModelSummary]:
        return [_summary(registry.info(c)) for c in registry.categories()]

    @router.post(
        "/inspections",
        response_model=InspectionResult,
        responses={
            404: {"description": "No model for this category"},
            413: {"description": "Upload too large"},
            415: {"description": "Unsupported file type"},
            422: {"description": "Not a decodable image"},
            501: {"description": "No detector models installed"},
        },
    )
    def inspect(
        file: Annotated[UploadFile, File(description="PNG, JPEG or BMP image")],
        category: Annotated[str, Form(min_length=1, max_length=64)],
    ) -> JSONResponse:
        started = time.perf_counter()
        if not registry.categories():
            raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "no detector models installed")
        if (file.content_type or "").lower() not in ALLOWED_TYPES:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f"unsupported type {file.content_type!r}; use one of {sorted(ALLOWED_TYPES)}",
            )
        data = file.file.read(max_upload_bytes + 1)
        if len(data) > max_upload_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"file larger than {max_upload_bytes} bytes",
            )
        try:
            image = decode_image(data)
        except InvalidImageError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        try:
            detector, info = registry.get(category)
        except UnknownModelError:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"no model for category {category!r}; available: {registry.categories()}",
            ) from None
        x = image_to_tensor(image, detector.image_size).unsqueeze(0)
        with torch.inference_mode():
            out = detector.predict(x)
        score, amap = float(out.scores[0]), out.maps[0]
        threshold = info.threshold
        decision: Literal["anomalous", "normal", "undecided"] = (
            "undecided" if threshold is None else "anomalous" if score > threshold else "normal"
        )
        heat = png_base64(overlay(image, normalize_map(amap, threshold)))
        result = InspectionResult(
            inspection_id=str(uuid7()),
            category=category,
            model=_summary(info),
            score=score,
            threshold=threshold,
            threshold_note=THRESHOLD_NOTE if threshold is not None else "no threshold",
            decision=decision,
            heatmap_png_base64=heat,
            heatmap_scale=(
                "absolute: 0.5 = image threshold, 1.0 = 2x threshold"
                if threshold
                else "relative: min-max of this image"
            ),
            map_max=float(amap.max()),
            image_width=image.width,
            image_height=image.height,
            latency_ms=0.0,
            description=(describer.describe(image, amap, category) if describer else None),
        )
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        _log.info(
            "inspection",
            category=category,
            score=round(score, 4),
            decision=decision,
            latency_ms=result.latency_ms,
        )
        body = result.model_dump(mode="json")
        if body["description"] is None:
            del body["description"]  # absent, not null: nothing is faked
        # JSONResponse so the absent field is not re-added as null by response_model.
        return JSONResponse(body)

    return router
