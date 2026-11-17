"""Both service APIs report ready against real Postgres (and Neo4j for diagnosis)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from iqs_common.settings import ApiSettings, CommonSettings, DatabaseSettings, Neo4jSettings
from iqs_diagnosis.api.main import create_app as create_diagnosis_app
from iqs_vision.api.main import create_app as create_vision_app

pytestmark = pytest.mark.integration


def _settings(db: DatabaseSettings, graph: Neo4jSettings | None = None) -> CommonSettings:
    return CommonSettings(
        _env_file=None,
        database=db,
        neo4j=graph or Neo4jSettings(),
        api=ApiSettings(ready_timeout_s=5),
    )


def test_vision_ready_with_postgres(pg_settings: DatabaseSettings) -> None:
    with TestClient(create_vision_app(_settings(pg_settings))) as c:
        r = c.get("/ready")
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ready", "checks": {"postgres": "ok"}}


def test_diagnosis_ready_with_postgres_and_neo4j(
    pg_settings: DatabaseSettings, neo4j_settings: Neo4jSettings
) -> None:
    with TestClient(create_diagnosis_app(_settings(pg_settings, neo4j_settings))) as c:
        r = c.get("/ready")
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ready", "checks": {"postgres": "ok", "neo4j": "ok"}}


def test_diagnosis_not_ready_when_neo4j_password_is_wrong(
    pg_settings: DatabaseSettings, neo4j_settings: Neo4jSettings
) -> None:
    wrong = neo4j_settings.model_copy(update={"password": SecretStr("definitely-wrong")})
    with TestClient(create_diagnosis_app(_settings(pg_settings, wrong))) as c:
        r = c.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"] == {"postgres": "ok", "neo4j": "error: AuthError"}
