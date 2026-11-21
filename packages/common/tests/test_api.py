"""Unit tests for iqs_common.api (no real dependencies: readiness checks are fakes)."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator, Mapping

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from pydantic import SecretStr

from iqs_common.api import (
    API_KEY_HEADER,
    REQUEST_ID_HEADER,
    ReadinessCheck,
    create_service_app,
)
from iqs_common.ids import uuid7_timestamp_ms
from iqs_common.settings import ApiSettings, CommonSettings

KEY = "test-key-1"
OTHER_KEY = "test-key-2"


def _ok() -> None:
    return None


def _boom() -> None:
    raise ConnectionError("db at 10.0.0.5:5432 refused")


def _slow() -> None:
    time.sleep(1.0)


def _client(
    checks: Mapping[str, ReadinessCheck] | None = None,
    *,
    keys: str = f"{KEY},{OTHER_KEY}",
    timeout_s: float = 0.2,
    routers: tuple[APIRouter, ...] = (),
) -> TestClient:
    settings = ApiSettings(keys=SecretStr(keys), ready_timeout_s=timeout_s)
    app = create_service_app(
        service="demo",
        version="1.2.3",
        settings=settings,
        readiness_checks=checks,
        features={"thing": True, "other": False},
        routers=routers,
    )
    return TestClient(app)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with _client({"db": _ok}) as c:
        yield c


# --- settings -------------------------------------------------------------------


def test_api_settings_defaults() -> None:
    s = CommonSettings().api
    assert s.key_list() == []
    assert s.cors_origins == ["http://localhost:8501"]
    assert s.ready_timeout_s == 2.0


def test_api_keys_from_comma_separated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IQS_API__KEYS", " new-key , old-key ,, ")
    monkeypatch.setenv("IQS_API__CORS_ORIGINS", '["http://localhost:8501","http://dash:8501"]')
    s = CommonSettings().api
    assert s.key_list() == ["new-key", "old-key"]
    assert s.cors_origins == ["http://localhost:8501", "http://dash:8501"]
    assert "new-key" not in repr(s)  # SecretStr hides keys


# --- health / ready -------------------------------------------------------------


def test_health_needs_no_key_and_no_dependencies() -> None:
    with _client({"db": _boom}) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "demo", "version": "1.2.3"}


def test_ready_when_all_checks_pass(client: TestClient) -> None:
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "checks": {"db": "ok"}}


def test_ready_without_checks_is_ready() -> None:
    with _client(None) as c:
        assert c.get("/ready").json() == {"status": "ready", "checks": {}}


def test_not_ready_reports_each_check_without_leaking_details() -> None:
    with _client({"db": _ok, "graph": _boom}) as c:
        r = c.get("/ready")
    assert r.status_code == 503
    assert r.json() == {
        "status": "not_ready",
        "checks": {"db": "ok", "graph": "error: ConnectionError"},
    }
    assert "10.0.0.5" not in r.text


def test_slow_check_times_out() -> None:
    with _client({"slow": _slow, "db": _ok}, timeout_s=0.1) as c:
        started = time.perf_counter()
        r = c.get("/ready")
        elapsed = time.perf_counter() - started
    assert r.status_code == 503
    assert r.json()["checks"] == {"slow": "timeout", "db": "ok"}
    assert elapsed < 0.9  # did not wait for the slow check to finish


# --- API key ----------------------------------------------------------------------


def test_capabilities_requires_api_key(client: TestClient) -> None:
    r = client.get("/v1/capabilities")
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"] == API_KEY_HEADER


@pytest.mark.parametrize("bad", ["", "wrong", KEY + "x", KEY.upper()])
def test_wrong_api_key_is_rejected(client: TestClient, bad: str) -> None:
    assert client.get("/v1/capabilities", headers={API_KEY_HEADER: bad}).status_code == 401


@pytest.mark.parametrize("key", [KEY, OTHER_KEY])
def test_any_configured_key_is_accepted(client: TestClient, key: str) -> None:
    r = client.get("/v1/capabilities", headers={API_KEY_HEADER: key})
    assert r.status_code == 200
    assert r.json() == {
        "service": "demo",
        "version": "1.2.3",
        "api_version": "v1",
        "features": {"thing": True, "other": False},
        "status": {"thing": "enabled", "other": "disabled"},
        "reason": {"other": "disabled in this configuration"},
    }


def test_no_configured_keys_fails_closed() -> None:
    with _client({"db": _ok}, keys="") as c:
        assert c.get("/v1/capabilities", headers={API_KEY_HEADER: ""}).status_code == 401
        assert c.get("/v1/capabilities", headers={API_KEY_HEADER: "x"}).status_code == 401
        assert c.get("/health").status_code == 200


def test_extra_routers_are_mounted_under_v1_behind_the_key() -> None:
    router = APIRouter()

    @router.get("/things")
    def things() -> list[int]:
        return [1, 2]

    with _client({"db": _ok}, routers=(router,)) as c:
        assert c.get("/v1/things").status_code == 401
        assert c.get("/v1/things", headers={API_KEY_HEADER: KEY}).json() == [1, 2]
        assert c.get("/things").status_code == 404


# --- request ID, CORS, OpenAPI, shutdown ------------------------------------------------


def test_request_id_is_echoed(client: TestClient) -> None:
    r = client.get("/health", headers={REQUEST_ID_HEADER: "abc-123"})
    assert r.headers[REQUEST_ID_HEADER] == "abc-123"


@pytest.mark.parametrize("incoming", [None, "", "x" * 129])
def test_request_id_is_generated_when_missing_or_too_long(
    client: TestClient, incoming: str | None
) -> None:
    headers = {} if incoming is None else {REQUEST_ID_HEADER: incoming}
    r = client.get("/health", headers=headers)
    generated = uuid.UUID(r.headers[REQUEST_ID_HEADER])
    assert uuid7_timestamp_ms(generated) > 0
    # also present on error responses
    assert client.get("/v1/capabilities").headers[REQUEST_ID_HEADER]


def test_cors_allows_only_the_dashboard_origin(client: TestClient) -> None:
    preflight = {
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": API_KEY_HEADER,
    }
    ok = client.options(
        "/v1/capabilities", headers={"Origin": "http://localhost:8501", **preflight}
    )
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "http://localhost:8501"
    bad = client.options("/v1/capabilities", headers={"Origin": "http://evil.example", **preflight})
    assert "access-control-allow-origin" not in bad.headers


def test_openapi_lists_ops_and_v1_routes(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    assert spec["info"] == {"title": "iqs-demo", "version": "1.2.3"}
    assert {"/health", "/ready", "/v1/capabilities"} <= set(spec["paths"])
    assert spec["components"]["securitySchemes"]["APIKeyHeader"]["name"] == API_KEY_HEADER


def test_shutdown_hooks_run_when_the_app_stops() -> None:
    closed: list[str] = []
    app = create_service_app(
        service="demo",
        version="0",
        settings=ApiSettings(),
        on_shutdown=[lambda: closed.append("engine"), lambda: closed.append("driver")],
    )
    with TestClient(app):
        assert closed == []
    assert closed == ["engine", "driver"]
