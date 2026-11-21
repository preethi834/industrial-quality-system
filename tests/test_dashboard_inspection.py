"""Dashboard pages run against the real vision API app (in-process), with a tiny
synthetic PatchCore model: upload -> score, decision, heatmap. Cross-package, so it
lives in the root tests/."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from iqs_dashboard import client as dash_client
from iqs_vision.api.main import create_app
from iqs_vision.settings import VisionSettings
from iqs_vision.testing import make_tiny_model, png_bytes

PAGES = Path(__file__).resolve().parents[1] / "packages/dashboard/src/iqs_dashboard"
KEY = "dash-key"


@pytest.fixture(scope="module")
def vision_http(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    base = tmp_path_factory.mktemp("dash")
    make_tiny_model(base / "models", base / "data", "toy")
    settings = VisionSettings(
        _env_file=None,
        api={"keys": KEY},
        database={"host": "127.0.0.1", "port": 1},
        vision={"models_dir": base / "models"},
    )
    return TestClient(create_app(settings))


@pytest.fixture
def wired(vision_http: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # TestClient is an httpx.Client that calls the ASGI app in-process.
    c = dash_client.VisionClient(cast(httpx.Client, vision_http), KEY)
    monkeypatch.setattr(dash_client, "get_client", lambda: c)


def test_home_shows_capabilities(wired: None) -> None:
    at = AppTest.from_file(str(PAGES / "app.py")).run(timeout=60)
    assert not at.exception
    assert any("Vision API" in s.value for s in at.success)


def test_inspection_page_lists_models(wired: None) -> None:
    at = AppTest.from_file(str(PAGES / "pages/2_Defect_Inspection.py")).run(timeout=60)
    assert not at.exception
    assert at.selectbox(key="category").options == ["toy"]


def test_inspection_roundtrip_through_client(wired: None) -> None:
    """What the page does on upload (AppTest cannot drive st.file_uploader)."""
    result = dash_client.get_client().inspect(png_bytes(64, defect="spot"), "toy")
    assert result["decision"] in {"anomalous", "normal"}
    assert result["heatmap_png_base64"]
    assert "description" not in result


def test_inspection_page_handles_unreachable_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> dash_client.VisionClient:
        raise ConnectionError("down")

    monkeypatch.setattr(dash_client, "get_client", boom)
    at = AppTest.from_file(str(PAGES / "pages/2_Defect_Inspection.py")).run(timeout=60)
    assert any("Cannot load models" in e.value for e in at.error)
