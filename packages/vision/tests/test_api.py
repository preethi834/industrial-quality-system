"""Vision API: /health, /ready, /v1/capabilities, /v1/models, POST /v1/inspections."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from iqs_vision.api.main import FEATURES, create_app
from iqs_vision.inference.registry import ModelRegistry
from iqs_vision.settings import VisionSettings
from iqs_vision.testing import make_tiny_model, png_bytes

KEY = "vision-test-key"
H = {"X-API-Key": KEY}


def _settings(models_dir: Path | None = None, **vision: object) -> VisionSettings:
    # Port 1 on localhost: nothing listens, the connection is refused at once.
    return VisionSettings(
        api={"keys": KEY, "ready_timeout_s": 1},
        database={"host": "127.0.0.1", "port": 1, "password": "x"},
        vision={"models_dir": models_dir or Path("no-models"), **vision},
    )


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("vision")
    make_tiny_model(base / "models", base / "data", "toy")
    return base / "models"


@pytest.fixture
def client(models_dir: Path) -> TestClient:
    return TestClient(create_app(_settings(models_dir)))


def _post(c: TestClient, data: bytes, ctype: str = "image/png", category: str = "toy", **kw):  # type: ignore[no-untyped-def]
    return c.post(
        "/v1/inspections",
        files={"file": ("part.png", data, ctype)},
        data={"category": category},
        **kw,
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


def test_capabilities_without_models() -> None:
    with TestClient(create_app(_settings())) as c:
        assert c.get("/v1/capabilities").status_code == 401
        r = c.get("/v1/capabilities", headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "vision" and body["api_version"] == "v1"
    assert set(body["features"]) == set(FEATURES)
    assert not any(body["features"].values())


def test_capabilities_with_models(client: TestClient) -> None:
    feats = client.get("/v1/capabilities", headers=H).json()["features"]
    assert feats["inspections"] is True
    assert feats["vlm_description"] is False


def test_factory_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IQS_API__KEYS", "from-env")
    monkeypatch.setenv("IQS_DATABASE__PORT", "1")
    monkeypatch.setenv("IQS_VISION__MODELS_DIR", "/nonexistent")
    with TestClient(create_app()) as c:
        assert c.get("/v1/capabilities", headers={"X-API-Key": "from-env"}).status_code == 200


def test_models_list(client: TestClient) -> None:
    r = client.get("/v1/models", headers=H)
    assert r.status_code == 200
    (m,) = r.json()
    assert m["category"] == "toy" and m["protocol"] == "deploy"
    assert m["version"] == "patchcore-toy-deploy-test0000"
    assert m["threshold"] is not None and m["threshold_method"] == "max_calibration_score"


def test_inspection_happy_path(client: TestClient) -> None:
    r = _post(client, png_bytes(64, defect="spot"), headers=H)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["category"] == "toy"
    assert body["decision"] in {"anomalous", "normal"}
    assert body["score"] > 0 and body["threshold"] > 0
    assert body["image_width"] == 64 and body["image_height"] == 64
    assert "provisional" in body["threshold_note"]
    png = base64.b64decode(body["heatmap_png_base64"])
    assert png.startswith(b"\x89PNG")
    # VLM disabled: the description is absent, not null and not faked
    assert "description" not in body


def test_defect_scores_higher_than_good(client: TestClient) -> None:
    good = _post(client, png_bytes(64, seed=11), headers=H).json()["score"]
    bad = _post(client, png_bytes(64, defect="spot", seed=11), headers=H).json()["score"]
    assert bad > good


def test_inspection_needs_api_key(client: TestClient) -> None:
    assert _post(client, png_bytes()).status_code == 401
    assert _post(client, png_bytes(), headers={"X-API-Key": "wrong"}).status_code == 401


def test_unsupported_type_is_415(client: TestClient) -> None:
    r = _post(client, b"%PDF-1.7", ctype="application/pdf", headers=H)
    assert r.status_code == 415


def test_corrupt_image_is_422(client: TestClient) -> None:
    assert _post(client, b"not really a png", headers=H).status_code == 422


def test_missing_fields_are_422(client: TestClient) -> None:
    r = client.post("/v1/inspections", data={"category": "toy"}, headers=H)
    assert r.status_code == 422


def test_unknown_category_is_404(client: TestClient) -> None:
    r = _post(client, png_bytes(), category="bottle", headers=H)
    assert r.status_code == 404
    assert "toy" in r.json()["detail"]


def test_too_large_is_413(models_dir: Path) -> None:
    with TestClient(create_app(_settings(models_dir, max_upload_bytes=1024))) as c:
        r = _post(c, png_bytes(64) + b"\0" * 2048, headers=H)
    assert r.status_code == 413


def test_no_models_is_501() -> None:
    with TestClient(create_app(_settings())) as c:
        assert _post(c, png_bytes(), headers=H).status_code == 501


def test_registry_cache_evicts(models_dir: Path, tmp_path: Path) -> None:
    reg = ModelRegistry(models_dir, "deploy", cache_size=1)
    a, _ = reg.get("toy")
    b, _ = reg.get("toy")
    assert a is b
    with pytest.raises(KeyError):
        reg.get("missing")
    assert ModelRegistry(tmp_path / "none", "deploy").categories() == []
