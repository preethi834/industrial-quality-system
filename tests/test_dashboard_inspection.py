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


def test_model_evaluation_page_without_calibration(wired: None) -> None:
    at = AppTest.from_file(str(PAGES / "pages/3_Model_Evaluation.py")).run(timeout=60)
    assert not at.exception
    assert at.selectbox(key="category").options == ["toy"]
    assert any("No conformal calibration" in i.value for i in at.info)


def test_model_evaluation_page_shows_levels(
    vision_http: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iqs_vision.calibration.calibrate import ScoreRecord, build_calibration

    recs = [ScoreRecord(f"c{i}", "0", "calib", 0, "good", float(i)) for i in range(60)]
    recs += [
        ScoreRecord("g", "0", "test", 0, "good", 1.0),
        ScoreRecord("b", "0", "test", 1, "x", 99.0),
    ]
    meta = {"dataset": "s", "category": "toy", "protocol": "deploy", "manifest_fingerprint": "f"}
    detail = {"category": "toy", "calibration": build_calibration(recs, meta)}
    c = dash_client.VisionClient(cast(httpx.Client, vision_http), KEY)
    monkeypatch.setattr(c, "calibration", lambda _cat: detail)
    monkeypatch.setattr(dash_client, "get_client", lambda: c)
    at = AppTest.from_file(str(PAGES / "pages/3_Model_Evaluation.py")).run(timeout=60)
    assert not at.exception
    assert any("n = 60 held-out good images" in s.value for s in at.subheader)
    assert len(at.dataframe) == 2


def test_model_evaluation_page_handles_unreachable_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> dash_client.VisionClient:
        raise ConnectionError("down")

    monkeypatch.setattr(dash_client, "get_client", boom)
    at = AppTest.from_file(str(PAGES / "pages/3_Model_Evaluation.py")).run(timeout=60)
    assert any("Cannot load models" in e.value for e in at.error)


def test_cost_decision_view() -> None:
    from iqs_dashboard.costs import cost_decision_view

    cd = {
        "policy": "guaranteed",
        "action": "review",
        "p_defect": 0.1,
        "expected_cost": {"accept": 20.0, "reject": 4.5, "review": 1.59},
        "currency": "EUR",
        "cost_config_id": "example_v1",
        "cost_config_sha256": "ab" * 32,
        "defect_rate_assumed": 0.01,
        "note": "costs are assumptions",
    }
    headline, rows, caption = cost_decision_view(cd)
    assert "SEND TO HUMAN REVIEW" in headline and "1.00%" in headline
    assert [r["chosen"] for r in rows] == ["", "", "✓"]
    assert rows[2]["expected cost (EUR)"] == 1.59
    assert "sha256:abababababab" in caption and "assumptions" in caption
