"""Calibration records, D1 alpha policy, leakage refusal, calibration.json content."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from iqs_vision.calibration.calibrate import (
    ScoreRecord,
    build_calibration,
    check_calibration_disjoint,
    load_scores,
    policy_alpha,
    save_scores,
)
from iqs_vision.calibration.conformal import InsufficientCalibrationError
from iqs_vision.data.mvtec import scan_category
from iqs_vision.data.splits import SplitLeakageError, SplitManifest, build_manifest
from iqs_vision.testing import make_mvtec_like

REPO = Path(__file__).resolve().parents[3]
META = {"dataset": "toy", "category": "c", "protocol": "deploy", "manifest_fingerprint": "f" * 64}


def _records(
    n_calib: int, n_val: int = 0, n_good: int = 30, n_bad: int = 20, seed: int = 0
) -> list[ScoreRecord]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_calib):
        out.append(ScoreRecord(f"t{i}", "0", "calib", 0, "good", float(rng.normal())))
    for i in range(n_val):
        out.append(
            ScoreRecord(f"v{i}", "0", "calib", 0, "good", float(rng.normal()), source="validation")
        )
    for i in range(n_good):
        out.append(ScoreRecord(f"g{i}", "0", "test", 0, "good", float(rng.normal())))
    for i in range(n_bad):
        out.append(ScoreRecord(f"b{i}", "0", "test", 1, "bad", float(rng.normal(4))))
    return out


@pytest.mark.parametrize(
    ("n", "alpha"),
    [(49, 0.02), (99, 0.02), (48, 0.05), (19, 0.05), (18, 0.10), (12, 0.10), (9, 0.10)],
)
def test_policy_alpha_follows_d1(n: int, alpha: float) -> None:
    assert policy_alpha(n) == alpha


def test_policy_alpha_refuses_tiny_sets() -> None:
    with pytest.raises(InsufficientCalibrationError):
        policy_alpha(8)


def test_build_calibration_active_threshold_and_levels() -> None:
    res = build_calibration(_records(53), META)
    assert res["n_calib"] == 53 and res["active"]["alpha"] == 0.02
    assert res["active"]["k"] == 53 and res["threshold"] == res["active"]["threshold"]
    levels = {lv["alpha"]: lv for lv in res["levels"]}
    assert levels[0.02]["alpha_delta"]["feasible"] is False  # n < 114
    assert levels[0.10]["alpha_delta"]["k"] >= levels[0.10]["marginal"]["k"]
    test = levels[0.02]["marginal"]["test"]
    assert test["n_good"] == 30 and test["n_defect"] == 20
    lo, hi = test["false_alarm_ci95"]
    assert lo <= test["false_alarm_rate"] <= hi
    assert levels[0.02]["marginal"]["predicted_false_alarm"]["m"] == 30
    assert "validation_only" not in res
    json.dumps(res)  # serialisable


def test_small_set_uses_larger_alpha_and_marks_infeasible_levels() -> None:
    res = build_calibration(_records(42), META)
    assert res["active"]["alpha"] == 0.05
    assert res["levels"][0]["feasible"] is False and "need n >= 49" in res["levels"][0]["reason"]


def test_given_alpha_too_small_for_n_is_refused() -> None:
    with pytest.raises(InsufficientCalibrationError):
        build_calibration(_records(42), META, alpha=0.02)


def test_validation_only_check_is_reported() -> None:
    res = build_calibration(_records(58, n_val=41), META)
    assert res["n_calib"] == 99 and res["active"]["alpha"] == 0.02
    v = res["validation_only"]
    assert v["n_calib"] == 41 and v["levels"][0]["feasible"] is False
    assert v["levels"][1]["feasible"] is True  # alpha = 0.05 needs 19


def test_scores_roundtrip(tmp_path: Path) -> None:
    recs = _records(10, n_val=2)
    save_scores(tmp_path / "s.json", META, recs)
    meta, back = load_scores(tmp_path / "s.json")
    assert back == recs and meta["category"] == "c"


def test_official_protocol_cannot_be_calibrated(tmp_path: Path) -> None:
    root = make_mvtec_like(tmp_path / "d", n_train=20)
    m = build_manifest(scan_category(root, "toy"), root=root, dataset="t", category="toy")
    check_calibration_disjoint(m, "deploy")
    with pytest.raises(SplitLeakageError, match="cannot be calibrated"):
        check_calibration_disjoint(m, "official")


def test_leakage_by_content_is_refused(tmp_path: Path) -> None:
    root = make_mvtec_like(tmp_path / "d", n_train=20)
    m = build_manifest(scan_category(root, "toy"), root=root, dataset="t", category="toy")
    fit0 = m.select("fit")[0]
    entries = [
        e.model_copy(update={"sha256": fit0.sha256}) if e.split == "calib" and i == 0 else e
        for i, e in enumerate(sorted(m.entries, key=lambda e: e.split != "calib"))
    ]
    # bypass the manifest validator (it would catch this too) to test the calibration guard
    leaky = SplitManifest.model_construct(**{**m.__dict__, "entries": tuple(entries)})
    with pytest.raises(SplitLeakageError):
        check_calibration_disjoint(leaky, "deploy")


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("calibrate", REPO / "scripts/calibrate.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_from_scores_writes_calibration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_dir = tmp_path / "models" / "toy" / "deploy"
    model_dir.mkdir(parents=True)
    (model_dir / "model.pt").write_bytes(b"")
    save_scores(model_dir / "scores.json", META | {"category": "toy"}, _records(60))
    rc = _script().main(
        [
            "--categories",
            "toy",
            "--models-dir",
            str(tmp_path / "models"),
            "--from-scores",
            "--no-mlflow",
        ]
    )
    assert rc == 0
    cal = json.loads((model_dir / "calibration.json").read_text())
    assert cal["active"]["alpha"] == 0.02 and cal["n_calib"] == 60
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["category"] == "toy" and out["k"] == cal["active"]["k"]
