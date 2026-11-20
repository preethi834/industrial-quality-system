"""Conformal core (blueprint §2.8): index rule, coverage law, (alpha, delta), LAC / APS.

MAPIE is not used (user decision); the set tests compare against reference sets worked
out by hand in the comments.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import beta, kstest

from iqs_diagnosis.conformal.scores import Method, aps_score, lac_score, prediction_set
from iqs_diagnosis.conformal.split import (
    InsufficientCalibrationError,
    beta_k,
    conformal_threshold,
    conformal_threshold_pac,
    marginal_k,
)
from iqs_diagnosis.evaluation.metrics import clopper_pearson

# ------------------------------------------------------------------ thresholds


def test_index_rule() -> None:
    assert marginal_k(9, 0.1) == 9  # ceil(10 * 0.9)
    assert marginal_k(49, 0.02) == 49  # blueprint example
    assert marginal_k(99, 0.1) == 90
    t = conformal_threshold([float(i) for i in range(1, 10)], 0.1)  # n = 9
    assert (t.k, t.qhat) == (9, 9.0)
    with pytest.raises(InsufficientCalibrationError, match="need n >= 9"):
        conformal_threshold([float(i) for i in range(8)], 0.1)  # n = 8 -> k = 9 > 8
    with pytest.raises(ValueError):
        conformal_threshold([1.0, math.nan], 0.1)
    with pytest.raises(ValueError):
        conformal_threshold([1.0], 1.0)


def test_marginal_coverage_and_beta_law() -> None:
    """10,000 trials of exchangeable continuous scores: mean coverage lies in
    [1 - alpha, 1 - alpha + 1/(n + 1)] (Monte-Carlo error allowed), and the realised
    coverage of each fixed threshold follows Beta(k, n + 1 - k) (KS test)."""
    rng = np.random.default_rng(0)
    alpha, n, trials = 0.1, 49, 10_000
    k = marginal_k(n, alpha)
    realised = np.empty(trials)
    hits = 0
    for t in range(trials):
        x = rng.uniform(size=n + 1)
        q = conformal_threshold(list(x[:n]), alpha).qhat
        realised[t] = q  # U(0,1) scores: coverage of threshold q is exactly q
        hits += x[n] <= q
    cov = hits / trials
    se = math.sqrt(cov * (1 - cov) / trials)
    assert 1 - alpha - 3 * se <= cov <= 1 - alpha + 1 / (n + 1) + 3 * se
    assert kstest(realised, beta(k, n + 1 - k).cdf).pvalue > 0.01


def test_pac_threshold() -> None:
    n, alpha, delta = 200, 0.1, 0.1
    k = beta_k(n, alpha, delta)
    assert k > marginal_k(n, alpha)  # more conservative than the marginal threshold
    assert beta.cdf(1 - alpha, k, n + 1 - k) <= delta < beta.cdf(1 - alpha, k - 1, n + 2 - k)
    t = conformal_threshold_pac([float(i) for i in range(n)], alpha, delta)
    assert t.k == k and t.delta == delta
    with pytest.raises(InsufficientCalibrationError):
        conformal_threshold_pac([0.1] * 10, 0.1, 0.05)
    # Realised coverage >= 1 - alpha in at least 1 - delta of calibration draws.
    rng = np.random.default_rng(1)
    ok = sum(
        conformal_threshold_pac(list(rng.uniform(size=n)), alpha, delta).qhat >= 1 - alpha
        for _ in range(2000)
    )
    assert ok / 2000 >= 1 - delta - 3 * math.sqrt(delta * (1 - delta) / 2000)


def test_ties_make_coverage_conservative() -> None:
    """Discrete scores (3 values): coverage stays >= 1 - alpha but exceeds the upper bound."""
    rng = np.random.default_rng(2)
    alpha, n, trials = 0.1, 30, 20_000
    hits = 0
    for _ in range(trials):
        x = rng.choice([0.0, 0.5, 1.0], p=[0.6, 0.35, 0.05], size=n + 1)
        hits += x[n] <= conformal_threshold(list(x[:n]), alpha).qhat
    cov = hits / trials
    assert cov >= 1 - alpha
    assert cov > 1 - alpha + 1 / (n + 1)  # the continuous upper bound does not hold


# ------------------------------------------------------------------ sets (hand-computed)

P = {"A": 0.5, "B": 0.3, "C": 0.15, "D": 0.05}


def test_scores_hand_computed() -> None:
    assert lac_score(P, "B") == pytest.approx(0.7)
    # APS order A, B, C, D; cumulative 0.5, 0.8, 0.95, 1.0
    assert aps_score(P, "A") == pytest.approx(0.5)
    assert aps_score(P, "C") == pytest.approx(0.95)
    assert aps_score(P, "D") == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("method", "qhat", "expected"),
    [
        # LAC: include y with 1 - p(y) <= q. Scores A .5, B .7, C .85, D .95.
        (Method.LAC, 0.70, ["A", "B"]),
        (Method.LAC, 0.49, []),  # empty set: escalate
        (Method.LAC, 0.95, ["A", "B", "C", "D"]),
        # APS: add in order until cumulative >= q (the crossing cause is included).
        (Method.APS, 0.40, ["A"]),  # 0.5 >= 0.4 at once: never empty
        (Method.APS, 0.80, ["A", "B"]),  # reaches exactly 0.8
        (Method.APS, 0.81, ["A", "B", "C"]),
    ],
)
def test_sets_hand_computed(method: Method, qhat: float, expected: list[str]) -> None:
    assert prediction_set(method, P, qhat) == expected


def test_aps_tie_order_is_fixed_by_code() -> None:
    probs = {"B": 0.25, "A": 0.25, "C": 0.5}
    # Order C (0.5), then A before B (tie -> code): cumulative 0.5, 0.75, 1.0.
    assert aps_score(probs, "A") == pytest.approx(0.75)
    assert aps_score(probs, "B") == pytest.approx(1.0)
    assert prediction_set(Method.APS, probs, 0.6) == ["C", "A"]


def test_true_label_covered_iff_score_below_q() -> None:
    """Consistency: score(y) <= q implies y is in the set, for both methods."""
    rng = np.random.default_rng(3)
    for _ in range(500):
        w = rng.dirichlet(np.ones(6))
        probs = {f"C{i}": float(v) for i, v in enumerate(w)}
        q = float(rng.uniform())
        for m in Method:
            s = prediction_set(m, probs, q)
            for y in probs:
                sc = lac_score(probs, y) if m is Method.LAC else aps_score(probs, y)
                if sc <= q:
                    assert y in s


def test_clopper_pearson() -> None:
    p = clopper_pearson(90, 100)
    assert p.low == pytest.approx(0.8238, abs=1e-4) and p.high == pytest.approx(0.9510, abs=1e-4)
    assert clopper_pearson(0, 10).low == 0.0 and clopper_pearson(10, 10).high == 1.0
