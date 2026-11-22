"""Discovery routes (step 7): ``POST /v1/discovery/inspect``, ``GET /v1/clusters[/{id}]``.

A separate inspect route keeps ``POST /v1/inspections`` untouched while step 2 changes it on
another branch; both use the same registry, decoder and detector.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from iqs_vision.data.preprocess import InvalidImageError, decode_image
from iqs_vision.discovery.service import DiscoveryService, PartContext
from iqs_vision.discovery.store import ClusterRecord, ClusterStatus, InMemoryDiscoveryStore
from iqs_vision.discovery.trigger import TriggerParams
from iqs_vision.inference.registry import ModelRegistry, UnknownModelError
from iqs_vision.settings import VisionSettings

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/bmp"}
RUN_ID = re.compile(r"^run-[0-9a-f]{12}$")
TRIGGER_NOTE = (
    "minimal step-7 trigger: HDBSCAN over unknown anomalies with untuned starting "
    "parameters; the anomaly threshold is the registry's (no false-alarm guarantee here)"
)


class InspectResponse(BaseModel):
    inspection_id: str
    category: str
    score: float
    threshold: float | None
    decision: str
    routing: str  # normal | undecided | known:<name> | unknown_buffered
    cluster_id: str | None
    alert_event_id: str | None
    model_version: str
    note: str = TRIGGER_NOTE


def cluster_json(c: ClusterRecord) -> dict[str, Any]:
    return {
        "cluster_id": str(c.id),
        "category": c.category,
        "status": c.status.value,
        "size": c.member_count,
        "first_seen": c.first_seen.isoformat(),
        "last_seen": c.last_seen.isoformat(),
        "radius": c.radius,
        "representative_inspection_id": str(c.representative_id),
        "alert_event_id": str(c.alerted_event_id),
        "defect_class_name": c.defect_class_name,
        "cause_code": c.cause_code,
        "case_id": None if c.case_id is None else str(c.case_id),
    }


def _aware(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "inspected_at: not ISO 8601"
        ) from None
    if ts.tzinfo is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "inspected_at needs a zone")
    return ts


def build_discovery_router(service: DiscoveryService, max_upload_bytes: int) -> APIRouter:
    router = APIRouter(tags=["discovery"])

    @router.post("/discovery/inspect", response_model=InspectResponse)
    def inspect(
        file: Annotated[UploadFile, File(description="PNG, JPEG or BMP image")],
        category: Annotated[str, Form(min_length=1, max_length=64)],
        part_id: Annotated[str | None, Form(max_length=64)] = None,
        run_id: Annotated[str | None, Form(max_length=32)] = None,
        line_id: Annotated[str | None, Form(max_length=64)] = None,
        inspected_at: Annotated[str | None, Form(description="ISO 8601 with zone")] = None,
    ) -> InspectResponse:
        if (file.content_type or "").lower() not in ALLOWED_TYPES:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "unsupported type")
        if run_id is not None and not RUN_ID.match(run_id):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid run_id")
        data = file.file.read(max_upload_bytes + 1)
        if len(data) > max_upload_bytes:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "file too large")
        try:
            image = decode_image(data)
        except InvalidImageError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        ctx = PartContext(part_id, run_id, line_id, _aware(inspected_at))
        try:
            r = service.route_observation(image, category, ctx)
        except UnknownModelError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no model for {category!r}") from None
        o = r.observation
        routing = f"known:{r.known_name}" if r.known_name else o.routing.value
        return InspectResponse(
            inspection_id=str(o.id),
            category=category,
            score=o.score,
            threshold=o.threshold,
            decision=o.decision,
            routing=routing,
            cluster_id=None if r.cluster_id is None else str(r.cluster_id),
            alert_event_id=None if r.alert_event_id is None else str(r.alert_event_id),
            model_version=o.model_version,
        )

    @router.get("/clusters")
    def list_clusters(
        category: str | None = None,
        cluster_status: Annotated[ClusterStatus | None, Query(alias="status")] = None,
    ) -> list[dict[str, Any]]:
        out = service.store.clusters(category)
        if cluster_status is not None:
            out = [c for c in out if c.status == cluster_status]
        return [cluster_json(c) for c in out]

    @router.get("/clusters/{cluster_id}")
    def get_cluster(cluster_id: uuid.UUID) -> dict[str, Any]:
        c = service.store.get_cluster(cluster_id)
        if c is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "cluster not found")
        return {**cluster_json(c), "members": [str(m) for m in service.store.members(c.id)]}

    return router


def build_discovery(settings: VisionSettings, registry: ModelRegistry) -> DiscoveryService | None:
    """``None`` when the trigger is disabled (capability ``discovery_trigger`` = false)."""
    from iqs_common.outbox_worker import InMemoryOutbox
    from iqs_vision.discovery.service import PatchCoreScorer

    cfg = settings.vision.discovery
    if not cfg.enabled:
        return None
    params = TriggerParams.load(cfg.config_path)
    store: Any
    if cfg.store == "memory":
        store = InMemoryDiscoveryStore(InMemoryOutbox())
    else:
        from iqs_common.db import create_db_engine, create_session_factory
        from iqs_vision.discovery.store import SqlDiscoveryStore

        store = SqlDiscoveryStore(create_session_factory(create_db_engine(settings.database)))
    return DiscoveryService(PatchCoreScorer(registry, params.top_k_patches), store, params)
