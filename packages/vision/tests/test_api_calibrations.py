"""GET/POST /v1/calibrations and the conformal threshold in POST /v1/inspections."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from iqs_vision.api.main import create_app
from iqs_vision.calibration.calibrate import ScoreRecord, build_calibration, save_scores
from iqs_vision.settings import VisionSettings
from iqs_vision.testing import make_tiny_model, png_bytes

KEY = "cal-key"
H = {"X-API-Key": KEY}
META = {
    "dataset": "synthetic",
    "category": "toy",
    "protocol": "deploy",
    "manifest_fingerprint": "f",
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("cal")
    make_tiny_model(base / "models", base / "data", "toy")
    return base / "models"


def _records(n_calib: int) -> list[ScoreRecord]:
    rng = np.random.default_rng(0)
    recs = [
        ScoreRecord(f"c{i}", "0", "calib", 0, "good", float(rng.normal())) for i in range(n_calib)
    ]
    recs += [ScoreRecord(f"g{i}", "0", "test", 0, "good", float(rng.normal())) for i in range(20)]
    recs += [ScoreRecord(f"b{i}", "0", "test", 1, "bad", float(rng.normal(3))) for i in range(20)]
    return recs


@pytest.fixture
def models(trained: Path, tmp_path: Path) -> Path:
    out = tmp_path / "models"
    shutil.copytree(trained, out)
    return out


def _client(models_dir: Path) -> TestClient:
    s = VisionSettings(
        _env_file=None,
        api={"keys": KEY},
        database={"host": "127.0.0.1", "port": 1},
        vision={"models_dir": models_dir},
    )
    return TestClient(create_app(s))


def _inspect(c: TestClient) -> dict[str, Any]:
    r = c.post(
        "/v1/inspections",
        files={"file": ("p.png", png_bytes(defect="spot"), "image/png")},
        data={"category": "toy"},
        headers=H,
    )
    assert r.status_code == 200
    return dict(r.json())


def test_without_calibration_file_the_threshold_is_provisional(models: Path) -> None:
    c = _client(models)
    row = c.get("/v1/calibrations", headers=H).json()[0]
    assert row["threshold_method"] == "max_calibration_score" and row["alpha"] is None
    assert row["source"] is None and row["persisted"] is False
    assert c.get("/v1/capabilities", headers=H).json()["features"]["conformal_threshold"] is False
    assert _inspect(c)["threshold_note"].startswith("provisional")


def test_calibration_file_is_used_by_inspections(models: Path) -> None:
    d = models / "toy/deploy"
    cal = build_calibration(_records(60), META)
    cal["active"]["threshold"] = cal["threshold"] = 1e9  # nothing can exceed it
    (d / "calibration.json").write_text(json.dumps(cal))
    c = _client(models)
    assert c.get("/v1/capabilities", headers=H).json()["features"]["conformal_threshold"] is True
    res = _inspect(c)
    assert res["decision"] == "normal" and res["threshold"] == 1e9
    assert res["model"]["threshold_method"] == "split_conformal_marginal"
    assert res["model"]["threshold_alpha"] == 0.02
    assert "<= 0.02" in res["threshold_note"] and "n = 60" in res["threshold_note"]
    detail = c.get("/v1/calibrations/toy", headers=H).json()
    assert detail["source"] == "file" and detail["persisted"] and detail["n_calib"] == 60


def test_post_recalibrates_in_memory_from_saved_scores(models: Path) -> None:
    d = models / "toy/deploy"
    save_scores(d / "scores.json", META, _records(60))
    c = _client(models)
    r = c.post("/v1/calibrations", json={"category": "toy", "alpha": 0.1}, headers=H)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["alpha"] == 0.1 and body["source"] == "api" and body["persisted"] is False
    assert body["k"] == 55 and body["calibration"]["n_calib"] == 60  # ceil(61 * 0.9) = 55
    assert c.get("/v1/calibrations/toy", headers=H).json()["alpha"] == 0.1
    assert _inspect(c)["threshold"] == pytest.approx(body["threshold"])
    assert not (d / "calibration.json").exists()  # nothing written to disk
    default = c.post("/v1/calibrations", json={"category": "toy"}, headers=H).json()
    assert default["alpha"] == 0.02  # policy D1 for n = 60


def test_post_refuses_too_small_calibration_set(models: Path) -> None:
    save_scores(models / "toy/deploy/scores.json", META, _records(30))
    c = _client(models)
    r = c.post("/v1/calibrations", json={"category": "toy", "alpha": 0.02}, headers=H)
    assert r.status_code == 422 and "need n >= 49" in r.json()["detail"]


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"category": "nope"}, 404),
        ({"category": "toy"}, 404),  # no scores.json yet
        ({"category": "toy", "alpha": 0}, 422),
        ({"category": "toy", "alpha": 1.5}, 422),
        ({"category": "toy", "extra": 1}, 422),
    ],
)
def test_post_errors(models: Path, body: dict[str, Any], code: int) -> None:
    r = _client(models).post("/v1/calibrations", json=body, headers=H)
    assert r.status_code == code, r.text


def test_calibrations_need_api_key(models: Path) -> None:
    c = _client(models)
    assert c.get("/v1/calibrations").status_code == 401
    assert c.post("/v1/calibrations", json={"category": "toy"}).status_code == 401
    assert c.get("/v1/calibrations/nope", headers=H).status_code == 404
