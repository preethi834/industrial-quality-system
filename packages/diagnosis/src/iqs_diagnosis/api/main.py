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
from iqs_common.settings import CommonSettings, Neo4jSettings
from iqs_diagnosis.api.cases import CaseService, build_cases_router
from iqs_diagnosis.settings import DiagnosisSettings, get_diagnosis_settings

SERVICE = "diagnosis"

# Feature flags reported by /v1/capabilities; each step switches its own on.
# "cases", "retrieval" and "report_8d" turn on when a case library is configured.
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


def create_app(
    settings: CommonSettings | None = None, case_service: CaseService | None = None
) -> FastAPI:
    """``case_service`` is injected by tests; otherwise it is built from settings."""
    if settings is None:
        settings = get_diagnosis_settings()
    elif not isinstance(settings, DiagnosisSettings):
        settings = DiagnosisSettings.model_validate(settings.model_dump())
    configure_logging(settings.logging.level, json_output=settings.logging.json_output)
    timeout = settings.api.ready_timeout_s
    engine = create_db_engine(settings.database, connect_timeout_s=max(1, int(timeout)))
    driver = create_neo4j_driver(settings.neo4j, timeout)
    if case_service is None:
        from iqs_diagnosis.api.wiring import build_case_service

        case_service = build_case_service(settings)
    features = dict(FEATURES)
    routers = []
    if case_service is not None:
        features.update(cases=True, retrieval=True, report_8d=True)
        features["conformal_cause_set"] = case_service.agent.has_conformal
        routers.append(build_cases_router(case_service))
    return create_service_app(
        service=SERVICE,
        version=version("iqs-diagnosis"),
        settings=settings.api,
        readiness_checks={
            "postgres": lambda: check_database(engine),
            "neo4j": driver.verify_connectivity,
        },
        features=features,
        routers=routers,
        on_shutdown=[engine.dispose, driver.close],
    )
