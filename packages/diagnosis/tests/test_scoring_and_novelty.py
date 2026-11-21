"""Vote probabilities, parameter fit, novelty threshold and the cause ranker."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from iqs_diagnosis.novelty.threshold import fit_novelty_threshold, novelty_rates
from iqs_diagnosis.retrieval.base import Hit
from iqs_diagnosis.scoring.ranker import CauseRanker, RankerParams, fit_ranker
from iqs_diagnosis.scoring.votes import (
    VoteParams,
    fit_vote_params,
    mean_nll,
    top_k,
    vote_probabilities,
)

from .synthetic import library, profile

CAUSES = ["A", "B", "C", "D"]
LABEL = {"x": "A", "y": "A", "z": "B"}


def test_vote_probabilities_hand_computed() -> None:
    hits = [Hit("x", 2.0, 1), Hit("y", 1.0, 2), Hit("z", 1.0, 3)]
    p = vote_probabilities(hits, LABEL.__getitem__, CAUSES, VoteParams(3, 0.1))
    denom = 1 + 0.1 * 4
    assert p["A"] == pytest.approx((0.75 + 0.1) / denom)
    assert p["B"] == pytest.approx((0.25 + 0.1) / denom)
    assert p["C"] == p["D"] == pytest.approx(0.1 / denom)
    assert sum(p.values()) == pytest.approx(1.0)
    assert list(p) == CAUSES


def test_only_n_neighbours_vote() -> None:
    hits = [Hit("z", 1.0, 1), Hit("x", 1.0, 2), Hit("y", 1.0, 3)]
    p = vote_probabilities(hits, LABEL.__getitem__, CAUSES, VoteParams(1, 0.001))
    assert top_k(p, 1)[0][0] == "B"


def test_vote_rejects_unknown_cause_and_bad_params() -> None:
    with pytest.raises(ValueError):
        vote_probabilities([Hit("x", 1, 1)], lambda _: "Q", CAUSES, VoteParams())
    with pytest.raises(ValueError):
        VoteParams(0, 0.1)
    with pytest.raises(ValueError):
        VoteParams(1, 0.0)


def test_top_k_ties_by_code() -> None:
    assert top_k({"B": 0.4, "A": 0.4, "C": 0.2}, 2) == [("A", 0.4), ("B", 0.4)]


def test_fit_prefers_params_with_lower_nll() -> None:
    # The truth is always the 1st neighbour, never the majority of 5: 1 neighbour wins.
    hit_lists = [[Hit("z", 1.0, 1), Hit("x", 1.0, 2), Hit("y", 1.0, 3)]] * 4
    truths = ["B"] * 4
    params, nll = fit_vote_params(
        hit_lists, truths, LABEL.__getitem__, CAUSES, neighbours=(1, 3), alphas=(0.01, 0.1)
    )
    assert params == VoteParams(1, 0.01)
    assert nll == pytest.approx(mean_nll(hit_lists, truths, LABEL.__getitem__, CAUSES, params))


def test_novelty_threshold_order_statistic() -> None:
    scores = [float(i) for i in range(1, 20)]  # n = 19
    d = fit_novelty_threshold(scores, 0.10)  # k = floor(0.1 * 20) = 2
    assert d.threshold == 2.0 and d.n_calibration == 19
    assert d.is_novel(1.99) and not d.is_novel(2.0)
    never = fit_novelty_threshold(scores, 0.04)  # k = 0
    assert never.threshold == -math.inf and not never.is_novel(-1e9)
    with pytest.raises(ValueError):
        fit_novelty_threshold([], 0.1)
    with pytest.raises(ValueError):
        fit_novelty_threshold([1.0], 1.5)


def test_novelty_false_rate_is_bounded_by_alpha() -> None:
    """Exchangeable known cases: realised false-novelty rate <= alpha (within noise)."""
    rng = np.random.default_rng(0)
    alpha, n, trials = 0.1, 49, 4000
    flags = 0
    for _ in range(trials):
        x = rng.normal(size=n + 1)
        d = fit_novelty_threshold(list(x[:n]), alpha)
        flags += d.is_novel(float(x[n]))
    rate = flags / trials
    se = math.sqrt(alpha * (1 - alpha) / trials)
    assert rate <= alpha + 3 * se
    assert rate == pytest.approx(5 / 50, abs=4 * se)  # k/(n+1) exactly in expectation


def test_novelty_rates() -> None:
    d = fit_novelty_threshold([0.5, 0.6, 0.7, 0.8], 0.2)  # k = 1 -> t = 0.5
    r = novelty_rates(d, known=[0.4, 0.9], novel=[0.1, 0.2, 0.6])
    assert r.false_novelty_rate == 0.5 and r.detection_rate == pytest.approx(2 / 3)
    assert math.isnan(novelty_rates(d, [0.9], []).detection_rate)


TAXONOMY = [
    "MACHINE.TOOL_WEAR",
    "MACHINE.DRIVE_FAULT",
    "MATERIAL.BAD_BATCH",
    "MEASUREMENT.SENSOR_DRIFT",
]
KNOWN = TAXONOMY[:3]


def test_fit_and_rank_end_to_end(tmp_path: Path) -> None:
    lib = library(KNOWN, 15, seed=1)
    rng = np.random.default_rng(2)
    dev = [profile(f"D-{c}-{i}", c, rng) for c in TAXONOMY for i in range(10)]
    dev_causes = [c for c in TAXONOMY for _ in range(10)]
    report = fit_ranker(lib, TAXONOMY, dev, dev_causes, novelty_alpha=0.1)
    assert report.novelty.n_known == 30 and report.novelty.n_novel == 10
    assert report.novelty.detection_rate == 1.0  # the unseen cluster is far away
    assert report.novelty.false_novelty_rate <= 0.1

    report.params.save(tmp_path / "p.json")
    params = RankerParams.load(tmp_path / "p.json")
    assert params == report.params

    ranker = CauseRanker(lib, TAXONOMY, params)
    rng_q = np.random.default_rng(3)
    known_q = [ranker.rank(profile(f"Q{i}", "MATERIAL.BAD_BATCH", rng_q)) for i in range(20)]
    assert all(r.top3[0][0] == "MATERIAL.BAD_BATCH" for r in known_q)
    # False novelty is allowed at rate alpha = 0.1; 20 draws stay well below 6 flags.
    assert sum(r.novel for r in known_q) <= 5
    q = known_q[0]
    assert len(q.top3) == 3 and sum(q.probabilities.values()) == pytest.approx(1.0)
    assert all(n.cause_code == "MATERIAL.BAD_BATCH" for n in q.neighbours[:3])
    new = ranker.rank(profile("Q2", "MEASUREMENT.SENSOR_DRIFT", np.random.default_rng(4)))
    assert new.novel
    assert new.probabilities["MEASUREMENT.SENSOR_DRIFT"] > 0


def test_ranker_rejects_library_cause_outside_taxonomy() -> None:
    lib = library(["MACHINE.TOOL_WEAR"], 2)
    params = RankerParams(
        n_neighbours=1,
        alpha=0.1,
        novelty_threshold=None,
        novelty_alpha=0.1,
        n_novelty_calibration=0,
    )
    with pytest.raises(ValueError, match="not in the taxonomy"):
        CauseRanker(lib, ["MATERIAL.BAD_BATCH"], params)
