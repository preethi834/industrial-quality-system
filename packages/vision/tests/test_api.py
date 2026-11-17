"""Vision API skeleton: /health, /ready (Postgres), /v1/capabilities."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from iqs_common.settings import CommonSettings
from iqs_vision.api.main import FEATURES, create_app

KEY = "vision-test-key"


def _settings() -> CommonSettings:
    # Port 1 on localhost: nothing listens, the connection is refused at once.
    return CommonSettings(
        api={"keys": KEY, "ready_timeout_s": 1},
        database={"host": "127.0.0.1", "port": 1, "password": "x"},
    )


def test_health() -> None:
    with TestClient(create_app(_settings())) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "vision", "version": "0.1.0"}


def test_ready_checks_postgres_only_and_fails_without_it() -> None:
    with TestClient(create_app(_settings())) as c:
        r = c.get("/ready")
    assert r.status_code == 503
    assert r.json() == {"status": "not_ready", "checks": {"postgres": "error: OperationalError"}}


def test_capabilities_needs_key_and_lists_no_features_yet() -> None:
    with TestClient(create_app(_settings())) as c:
        assert c.get("/v1/capabilities").status_code == 401
        r = c.get("/v1/capabilities", headers={"X-API-Key": KEY})
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "vision"
    assert body["api_version"] == "v1"
    assert body["features"] == FEATURES
    assert not any(body["features"].values())


def test_factory_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IQS_API__KEYS", "from-env")
    monkeypatch.setenv("IQS_DATABASE__PORT", "1")
    with TestClient(create_app()) as c:
        assert c.get("/v1/capabilities", headers={"X-API-Key": "from-env"}).status_code == 200
