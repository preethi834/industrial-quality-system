"""Step 6c: cost_decision in POST /v1/inspections, present only when every artefact exists."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from iqs_vision.api.main import create_app
from iqs_vision.calibration.calibrate import ScoreRecord, build_calibration
from iqs_vision.decision.calibrate_proba import fit_platt
from iqs_vision.decision.costs import load_cost_config
from iqs_vision.decision.serving import DECISION_FILE, decision_artifact
from iqs_vision.settings import VisionSettings
from iqs_vision.testing import make_tiny_model, png_bytes

REPO = Path(__file__).resolve().parents[3]
COSTS = REPO / "configs/costs/example_costs.yaml"
KEY = "cost-key"
H = {"X-API-Key": KEY}
META = {
    "dataset": "synthetic",
    "category": "toy",
    "protocol": "deploy",
    "manifest_fingerprint": "f",
}
RUN = "test0000run"  # mlflow_run_id written by make_tiny_model


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("costs")
    make_tiny_model(base / "models", base / "data", "toy")
    return base / "models"


@pytest.fixture
def models(trained: Path, tmp_path: Path) -> Path:
    out = tmp_path / "models"
    shutil.copytree(trained, out)
    return out


def _write_calibration(models: Path, threshold: float) -> None:
    rng = np.random.default_rng(0)
    recs = [ScoreRecord(f"c{i}", "0", "calib", 0, "good", float(rng.normal())) for i in range(60)]
    cal = build_calibration(recs, META)
    cal["active"]["threshold"] = cal["threshold"] = threshold
    (models / "toy/deploy/calibration.json").write_text(json.dumps(cal))


def _write_decision(models: Path, run: str = RUN) -> None:
    rng = np.random.default_rng(1)
    s = np.r_[rng.normal(0, 1, 30), rng.normal(3, 1, 60)]
    y = np.r_[np.zeros(30, int), np.ones(60, int)]
    art = decision_artifact(
        fit_platt(s, y),
        dataset="synthetic",
        category="toy",
        model_run_id=run,
        manifest_fingerprint="f",
    )
    (models / "toy/deploy" / DECISION_FILE).write_text(json.dumps(art))


def _client(models: Path, cost_config: Path | None = COSTS) -> TestClient:
    s = VisionSettings(
        _env_file=None,
        api={"keys": KEY},
        database={"host": "127.0.0.1", "port": 1},
        vision={"models_dir": models, "cost_config": cost_config},
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


def _feature(c: TestClient) -> bool:
    return bool(c.get("/v1/capabilities", headers=H).json()["features"]["cost_decisions"])


def test_below_threshold_is_accepted_with_costs_and_hash(models: Path) -> None:
    _write_calibration(models, 1e9)
    _write_decision(models)
    c = _client(models)
    assert _feature(c)
    res = _inspect(c)
    cd = res["cost_decision"]
    assert res["decision"] == "normal" and cd["action"] == "accept"
    assert cd["policy"] == "guaranteed" and 0 <= cd["p_defect"] <= 1
    assert set(cd["expected_cost"]) == {"accept", "reject", "review"}
    assert cd["cost_config_sha256"] == load_cost_config(COSTS)[1]
    assert cd["currency"] == "EUR" and "assumption" in cd["note"]


def test_above_threshold_is_never_accepted(models: Path) -> None:
    _write_calibration(models, -1e9)
    _write_decision(models)
    res = _inspect(_client(models))
    assert res["decision"] == "anomalous"
    cd = res["cost_decision"]
    assert cd["action"] in ("reject", "review")
    costs = cd["expected_cost"]
    # the cheaper of reject / review is chosen (ties go to reject)
    assert cd["action"] == ("reject" if costs["reject"] <= costs["review"] else "review")


@pytest.mark.parametrize("missing", ["decision", "calibration", "cost_config", "other_run"])
def test_cost_fields_absent_when_an_artefact_is_missing(models: Path, missing: str) -> None:
    if missing != "calibration":
        _write_calibration(models, 1e9)
    if missing != "decision":
        _write_decision(models, run="another-run" if missing == "other_run" else RUN)
    c = _client(models, None if missing == "cost_config" else COSTS)
    assert not _feature(c)
    assert "cost_decision" not in _inspect(c)


def test_missing_cost_file_disables_the_feature(models: Path, tmp_path: Path) -> None:
    _write_calibration(models, 1e9)
    _write_decision(models)
    c = _client(models, tmp_path / "nope.yaml")
    assert not _feature(c) and "cost_decision" not in _inspect(c)
