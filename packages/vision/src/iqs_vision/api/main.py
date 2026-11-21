"""Vision service HTTP entry point.

Run: uvicorn --factory iqs_vision.api.main:create_app --port 8001
"""

from __future__ import annotations

from importlib.metadata import version

from fastapi import FastAPI

from iqs_common.api import create_service_app
from iqs_common.db import check_database, create_db_engine
from iqs_common.log import configure_logging
from iqs_vision.api.routes.inspections import build_router
from iqs_vision.explain.vlm import get_describer
from iqs_vision.inference.registry import ModelRegistry
from iqs_vision.settings import VisionSettings, get_vision_settings

SERVICE = "vision"

# Feature flags reported by /v1/capabilities; each step switches its own on.
# "inspections" is computed at startup: true when at least one trained model is found.
FEATURES: dict[str, bool] = {
    "inspections": False,  # step 1: detector + POST /v1/inspections
    "vlm_description": False,  # step 1 interface; disabled until a VLM is approved
    "conformal_threshold": False,  # step 2
    "cost_decisions": False,  # step 2
    "discovery": False,  # step 8c: unknown-defect clustering
    "fewshot": False,  # step 8d
    "edge": False,  # step 8e
    "counterfactual": False,  # step 8g
}


def create_app(
    settings: VisionSettings | None = None, registry: ModelRegistry | None = None
) -> FastAPI:
    settings = settings or get_vision_settings()
    configure_logging(settings.logging.level, json_output=settings.logging.json_output)
    timeout = max(1, int(settings.api.ready_timeout_s))
    engine = create_db_engine(settings.database, connect_timeout_s=timeout)
    v = settings.vision
    registry = registry or ModelRegistry(
        v.models_dir,
        v.protocol,
        assets_file=v.assets_file,
        assets_root=v.assets_root,
        cache_size=v.model_cache_size,
    )
    describer = get_describer(v.vlm_provider)
    features = {
        **FEATURES,
        "inspections": bool(registry.categories()),
        "vlm_description": describer is not None,
    }
    return create_service_app(
        service=SERVICE,
        version=version("iqs-vision"),
        settings=settings.api,
        readiness_checks={"postgres": lambda: check_database(engine)},
        features=features,
        routers=[build_router(registry, v.max_upload_bytes, describer)],
        on_shutdown=[engine.dispose],
    )
