"""Repeated-split validity check (step 2 gate 3) and the evaluate_conformal report."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from iqs_vision.calibration.calibrate import ScoreRecord, build_calibration, save_scores
from iqs_vision.calibration.conformal import InsufficientCalibrationError
from iqs_vision.calibration.validity import resplit_check

REPO = Path(__file__).resolve().parents[3]


def test_exchangeable_scores_pass_and_match_prediction() -> None:
    rng = np.random.default_rng(0)
    calib, good, bad = rng.normal(size=49), rng.normal(size=30), rng.normal(3, size=40)
    res = resplit_check(calib, good, bad, 0.02, resplits=400, seed=1)
    assert res.passed and res.k == 49 and res.m == 30 and len(res.false_alarm_rates) == 400
    assert res.mean_false_alarm == pytest.approx(res.predicted_mean, abs=4 * res.se_false_alarm)
    assert res.mean_miss is not None and 0 <= res.mean_miss <= 1


def test_gate_passes_across_many_independent_datasets() -> None:
    """The gate is a 3-SE test, so a correct implementation fails it only rarely."""
    fails = 0
    for s in range(40):
        rng = np.random.default_rng(100 + s)
        res = resplit_check(rng.normal(size=53), rng.normal(size=21), np.array([]), 0.02, seed=s)
        fails += not res.passed
    assert fails <= 4


def test_wrong_index_rule_fails_the_gate() -> None:
    rng = np.random.default_rng(5)
    calib, good = rng.normal(size=19), rng.normal(size=300)

    def floor_rule(n: int, alpha: float) -> int:  # a typical off-by-one bug
        return math.floor(n * (1 - alpha))

    assert resplit_check(calib, good, np.array([]), 0.10, seed=0).passed
    bad = resplit_check(calib, good, np.array([]), 0.10, seed=0, index_rule=floor_rule)
    assert not bad.passed and bad.mean_false_alarm > 0.1


def test_too_small_calibration_set_is_refused() -> None:
    with pytest.raises(InsufficientCalibrationError):
        resplit_check(np.zeros(40), np.zeros(10), np.zeros(3), 0.02)


def test_evaluate_conformal_writes_report_and_plot(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    recs = [ScoreRecord(f"c{i}", "0", "calib", 0, "good", float(rng.normal())) for i in range(53)]
    recs += [ScoreRecord(f"g{i}", "0", "test", 0, "good", float(rng.normal())) for i in range(21)]
    recs += [ScoreRecord(f"b{i}", "0", "test", 1, "bad", float(rng.normal(3))) for i in range(30)]
    meta = {
        "dataset": "mvtec_ad",
        "category": "grid",
        "protocol": "deploy",
        "manifest_fingerprint": "f",
    }
    d = tmp_path / "models/grid/deploy"
    d.mkdir(parents=True)
    save_scores(d / "scores.json", meta, recs)
    (d / "calibration.json").write_text(json.dumps(build_calibration(recs, meta)))
    spec = importlib.util.spec_from_file_location("ec", REPO / "scripts/evaluate_conformal.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main(["--models-dir", str(tmp_path / "models"), "--out-dir", str(tmp_path / "r")])
    assert rc == 0
    md = (tmp_path / "r/conformal_mvtec_ad.md").read_text()
    assert "| grid | 53 | 0.02 | 53 |" in md and "pass" in md
    assert (tmp_path / "r/figures/conformal_mvtec_ad.svg").read_text().startswith("<svg")
    rows = json.loads((tmp_path / "r/conformal_mvtec_ad.json").read_text())
    assert rows[0]["resplit"]["n"] == 53
