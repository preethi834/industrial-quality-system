"""Discovery routes with the tiny real PatchCore (random-init ResNet-18, synthetic images).

This exercises the real detector, patch features and embedding end to end. It is a
plumbing test: random-init features on stripe images say nothing about MVTec quality.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from iqs_common.events import EventType
from iqs_vision.api.main import create_app
from iqs_vision.discovery.service import DiscoveryService
from iqs_vision.discovery.store import InMemoryDiscoveryStore
from iqs_vision.settings import VisionSettings
from iqs_vision.testing import make_tiny_model

KEY = "vision-test-key"
H = {"X-API-Key": KEY}
REPO = Path(__file__).resolve().parents[3]
RUN = "run-0123456789ab"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    base = tmp_path_factory.mktemp("disc")
    make_tiny_model(base / "models", base / "data", "toy")
    return base / "models", base / "data" / "toy"


def _settings(models: Path, enabled: bool = True) -> VisionSettings:
    return VisionSettings(
        api={"keys": KEY, "ready_timeout_s": 1},
        database={"host": "127.0.0.1", "port": 1, "password": "x"},
        vision={
            "models_dir": models,
            "discovery": {
                "enabled": enabled,
                "store": "memory",
                "config_path": REPO / "configs" / "vision" / "discovery.yaml",
            },
        },
    )


def _post(c: TestClient, path: Path, i: int, **extra: str):  # type: ignore[no-untyped-def]
    return c.post(
        "/v1/discovery/inspect",
        files={"file": (path.name, path.read_bytes(), "image/png")},
        data={
            "category": "toy",
            "part_id": f"P{i:06d}",
            "run_id": RUN,
            "line_id": "L1",
            "inspected_at": f"2026-01-05T{6 + i // 60:02d}:{i % 60:02d}:00Z",
            **extra,
        },
        headers=H,
    )


def test_disabled_by_default(tiny: tuple[Path, Path]) -> None:
    c = TestClient(create_app(_settings(tiny[0], enabled=False)))
    caps = c.get("/v1/capabilities", headers=H).json()
    assert caps["features"]["discovery_trigger"] is False
    assert c.get("/v1/clusters", headers=H).status_code == 404


def test_goods_then_new_defect_alerts_once(tiny: tuple[Path, Path]) -> None:
    models, data = tiny
    app = create_app(_settings(models))
    c = TestClient(app)
    assert c.get("/v1/capabilities", headers=H).json()["features"]["discovery_trigger"]
    service = app.state.discovery
    assert isinstance(service, DiscoveryService)
    store = service.store
    assert isinstance(store, InMemoryDiscoveryStore)

    goods = sorted((data / "train" / "good").glob("*.png"))
    scratches = sorted((data / "test" / "scratch").glob("*.png"))
    i = 0
    for g in goods:
        r = _post(c, g, i)
        assert r.status_code == 200, r.text
        assert r.json()["routing"] == "normal"  # training goods (see image_linkage.yaml)
        i += 1
    routings = []
    for s in scratches:
        body = _post(c, s, i).json()
        routings.append(body)
        i += 1
    assert all(b["decision"] == "anomalous" for b in routings)
    alerts = [b for b in routings if b["alert_event_id"]]
    assert len(alerts) == 1, routings
    events = store.outbox.rows(EventType.NEW_DEFECT_CLUSTER)
    assert len(events) == 1
    payload: dict[str, object] = events[0].payload["payload"]  # type: ignore[assignment]
    assert payload["linkage"] == "synthetic_simulator" and payload["simulator_run_id"] == RUN

    clusters = c.get("/v1/clusters", headers=H, params={"status": "alerted"}).json()
    assert len(clusters) == 1 and clusters[0]["size"] >= 5
    one = c.get(f"/v1/clusters/{clusters[0]['cluster_id']}", headers=H).json()
    assert len(one["members"]) == clusters[0]["size"]


def test_validation(tiny: tuple[Path, Path]) -> None:
    models, data = tiny
    c = TestClient(create_app(_settings(models)))
    img = sorted((data / "train" / "good").glob("*.png"))[0]
    assert _post(c, img, 0, run_id="bad").status_code == 422
    assert _post(c, img, 0, inspected_at="2026-01-05T06:00:00").status_code == 422
    assert _post(c, img, 0, category="nope").status_code == 404
    r = c.post(
        "/v1/discovery/inspect",
        files={"file": ("x.gif", b"GIF89a", "image/gif")},
        data={"category": "toy"},
        headers=H,
    )
    assert r.status_code == 415
    assert c.post("/v1/discovery/inspect", data={"category": "toy"}).status_code == 401
    assert c.get("/v1/clusters/00000000-0000-0000-0000-000000000000", headers=H).status_code == 404
