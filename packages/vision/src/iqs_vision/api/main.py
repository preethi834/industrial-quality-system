"""Vision service HTTP entry point.

Run: uvicorn --factory iqs_vision.api.main:create_app --port 8001
"""

from __future__ import annotations

from importlib.metadata import version

from fastapi import FastAPI

from iqs_common.api import create_service_app
from iqs_common.db import check_database, create_db_engine
from iqs_common.log import configure_logging
from iqs_common.settings import CommonSettings, get_settings

SERVICE = "vision"

# Feature flags reported by /v1/capabilities; each step switches its own on.
FEATURES: dict[str, bool] = {
    "inspections": False,  # step 1: detector + POST /v1/inspections
    "conformal_threshold": False,  # step 2
    "cost_decisions": False,  # step 2
    "discovery": False,  # step 8c: unknown-defect clustering
    "fewshot": False,  # step 8d
    "edge": False,  # step 8e
    "counterfactual": False,  # step 8g
}


def create_app(settings: CommonSettings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.logging.level, json_output=settings.logging.json_output)
    timeout = max(1, int(settings.api.ready_timeout_s))
    engine = create_db_engine(settings.database, connect_timeout_s=timeout)
    return create_service_app(
        service=SERVICE,
        version=version("iqs-vision"),
        settings=settings.api,
        readiness_checks={"postgres": lambda: check_database(engine)},
        features=FEATURES,
        on_shutdown=[engine.dispose],
    )
