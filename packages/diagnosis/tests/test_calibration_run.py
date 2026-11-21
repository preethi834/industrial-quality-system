"""Calibration runs: hash split, disjointness, thresholds, the committed frozen run."""

from __future__ import annotations

from collections import Counter

import pytest

from iqs_diagnosis.conformal.calibration import (
    CalibrationRun,
    ConformalPredictor,
    LeakageError,
    calibrate,
    check_disjoint,
    hash_half,
)
from iqs_diagnosis.conformal.split import InsufficientCalibrationError, marginal_k
from iqs_diagnosis.scoring.combine import EXTENDED_ALPHAS, on_grid_edge
from iqs_diagnosis.scoring.ranker import RankerParams

from .conftest import REPO_ROOT

RUN_PATH = REPO_ROOT / "configs/diagnosis/calibration/cal_dx_corpus_v2_dev_a010.json"
PARAMS = RankerParams(
    n_neighbours=5,
    alpha=0.01,
    novelty_threshold=None,
    novelty_alpha=0.05,
    n_novelty_calibration=0,
)


def test_hash_half_is_deterministic_and_balanced() -> None:
    ids = [f"SIM-{i:012x}" for i in range(2000)]
    halves = [hash_half(i, "s") for i in ids]
    assert halves == [hash_half(i, "s") for i in ids]
    c = Counter(halves)
    assert set(c) == {"fit", "cal"} and abs(c["fit"] - 1000) < 100
    assert [hash_half(i, "other") for i in ids] != halves  # salt matters


def test_check_disjoint() -> None:
    check_disjoint(["a", "b"], library_ids={"c"}, fit_ids={"d"})
    with pytest.raises(LeakageError, match="library"):
        check_disjoint(["a"], library_ids={"a"}, fit_ids=set())
    with pytest.raises(LeakageError, match="fit"):
        check_disjoint(["a"], library_ids=set(), fit_ids={"a"})
    with pytest.raises(LeakageError, match="duplicate"):
        check_disjoint(["a", "a"], library_ids=set(), fit_ids=set())


def _run(n: int = 40) -> CalibrationRun:
    # Truth "A" with p(A) = i/(n+1): LAC scores 1 - i/(n+1), APS scores vary too.
    probs = [{"A": (i + 1) / (n + 1), "B": 1 - (i + 1) / (n + 1)} for i in range(n)]
    return calibrate(
        run_id="t_run",
        corpus="c",
        probabilities=probs,
        truths=["A"] * n,
        calibration_ids=[f"c{i}" for i in range(n)],
        fit_ids=["f0"],
        library_ids={"l0"},
        scorer=PARAMS,
        vocabulary_version="v",
        taxonomy_version="t",
    )


def test_calibrate_thresholds_match_order_statistics() -> None:
    # (alpha, delta) = (0.1, 0.1) needs 1 - 0.9**n >= 0.9, i.e. n >= 22.
    with pytest.raises(InsufficientCalibrationError):
        _run(19)
    run = _run()
    assert run.n_calibration == 40
    k = marginal_k(40, 0.1)
    for m in ("lac", "aps"):
        assert run.thresholds[m].k == k
        assert run.thresholds[m].qhat == run.calibration_scores[m][k - 1]
        assert run.thresholds[f"{m}_pac"].k >= k
    with pytest.raises(LeakageError):
        calibrate(
            run_id="x",
            corpus="c",
            probabilities=[{"A": 1.0}],
            truths=["A"],
            calibration_ids=["l0"],
            fit_ids=[],
            library_ids={"l0"},
            scorer=PARAMS,
            vocabulary_version="v",
            taxonomy_version="t",
        )
    pred = ConformalPredictor(run, "lac")
    assert pred.cause_set({"A": 0.97, "B": 0.03}) == ["A"]
    with pytest.raises(ValueError):
        ConformalPredictor(run, "nope")


def test_committed_run_is_consistent_and_leak_free() -> None:
    run = CalibrationRun.load(RUN_PATH)
    assert not set(run.calibration_case_ids) & set(run.fit_case_ids)
    assert len(set(run.calibration_case_ids)) == run.n_calibration >= 9
    for name, t in run.thresholds.items():
        scores = run.calibration_scores[t.method.value]
        assert scores == sorted(scores)
        assert t.qhat == scores[t.k - 1], name
    assert run.thresholds["lac"].k == marginal_k(run.n_calibration, run.alpha)
    # The refit smoothing is not on the edge of the extended grid (user requirement).
    assert not on_grid_edge(run.scorer.alpha, EXTENDED_ALPHAS)
    assert "not valid for real cases" in run.note
