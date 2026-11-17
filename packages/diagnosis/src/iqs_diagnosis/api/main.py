"""Diagnosis service HTTP entry point.

Run: uvicorn --factory iqs_diagnosis.api.main:create_app --port 8002
"""

from __future__ import annotations

from importlib.metadata import version

from fastapi import FastAPI
from neo4j import Driver, GraphDatabase

from iqs_common.api import create_service_app
from iqs_common.db import check_database, create_db_engine
from iqs_common.log import configure_logging
from iqs_common.settings import CommonSettings, Neo4jSettings, get_settings

SERVICE = "diagnosis"

# Feature flags reported by /v1/capabilities; each step switches its own on.
FEATURES: dict[str, bool] = {
    "cases": False,  # step 3: POST /v1/cases
    "retrieval": False,  # step 3: hybrid RAG
    "graph": False,  # step 3: Neo4j knowledge graph
    "report_8d": False,  # step 3
    "question_loop": False,  # step 4
    "conformal_cause_set": False,  # step 4
    "hypothesis_tests": False,  # step 5
    "fmea_proposals": False,  # step 6
}


def create_neo4j_driver(settings: Neo4jSettings, timeout_s: float) -> Driver:
    """Create (but do not connect) a driver; the first query opens a connection."""
    return GraphDatabase.driver(
        settings.uri,
        auth=(settings.user, settings.password.get_secret_value()),
        connection_timeout=timeout_s,
        connection_acquisition_timeout=timeout_s,
    )


def create_app(settings: CommonSettings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.logging.level, json_output=settings.logging.json_output)
    timeout = settings.api.ready_timeout_s
    engine = create_db_engine(settings.database, connect_timeout_s=max(1, int(timeout)))
    driver = create_neo4j_driver(settings.neo4j, timeout)
    return create_service_app(
        service=SERVICE,
        version=version("iqs-diagnosis"),
        settings=settings.api,
        readiness_checks={
            "postgres": lambda: check_database(engine),
            "neo4j": driver.verify_connectivity,
        },
        features=FEATURES,
        on_shutdown=[engine.dispose, driver.close],
    )
