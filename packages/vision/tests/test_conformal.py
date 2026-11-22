"""Conformal threshold: index rule, refusal, coverage properties (blueprint Rev 2 §2.8 tests)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from iqs_vision.calibration.beta_bound import (
    InfeasibleAlphaDeltaError,
    alpha_delta_index,
    alpha_delta_threshold,
    clopper_pearson,
    coverage_distribution,
    predicted_false_alarm,
    prob_coverage_at_least,
)
from iqs_vision.calibration.conformal import (
    InsufficientCalibrationError,
    conformal_index,
    false_alarm_rate,
    min_calibration_size,
    miss_rate,
    split_conformal_threshold,
    threshold_at_rank,
)

TRIALS = 10_000

# --- index rule ---------------------------------------------------------------------------


@pytest.mark.parametrize(("alpha", "n_min"), [(0.02, 49), (0.05, 19), (0.10, 9), (0.3, 3)])
def test_min_calibration_size(alpha: float, n_min: int) -> None:
    assert min_calibration_size(alpha) == n_min
    conformal_index(n_min, alpha)  # finite
    with pytest.raises(InsufficientCalibrationError, match=f"need n >= {n_min}"):
        conformal_index(n_min - 1, alpha)


def test_index_rule_blueprint_examples() -> None:
    assert conformal_index(49, 0.02) == 49  # (50)(0.98) = 49 exactly, despite float error
    assert conformal_index(99, 0.02) == 98
    assert conformal_index(90, 0.02) == 90  # ceil(91 * 0.98) = ceil(89.18)
    assert conformal_index(53, 0.02) == 53
    assert conformal_index(12, 0.10) == 12


@given(n=st.integers(1, 2000), alpha=st.floats(0.001, 0.5))
def test_index_matches_exact_rational_formula(n: int, alpha: float) -> None:
    exact = math.ceil(round((n + 1) * (1 - alpha), 9))
    if exact > n:
        with pytest.raises(InsufficientCalibrationError):
            conformal_index(n, alpha)
    else:
        assert conformal_index(n, alpha) == max(1, exact)


def test_refuses_too_small_calibration_set() -> None:
    with pytest.raises(InsufficientCalibrationError, match="k = 49 > n"):
        split_conformal_threshold(np.arange(48.0), 0.02)
    with pytest.raises(InsufficientCalibrationError):
        split_conformal_threshold([], 0.1)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_alpha_must_be_in_unit_interval(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        conformal_index(100, alpha)


def test_threshold_is_kth_smallest_score_and_ignores_order() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=49)
    t = split_conformal_threshold(rng.permutation(scores), 0.02)
    assert (t.n, t.k, t.method) == (49, 49, "marginal")
    assert t.threshold == scores.max()
    assert threshold_at_rank(scores, 1) == scores.min()
    assert t.to_dict()["threshold"] == t.threshold


def test_non_finite_scores_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        split_conformal_threshold([*range(60), float("nan")], 0.05)


def test_false_alarm_and_miss_rates_use_strict_flagging() -> None:
    assert false_alarm_rate([0.1, 0.5, 0.9], 0.5) == pytest.approx(1 / 3)  # 0.5 is accepted
    assert miss_rate([0.5, 0.6], 0.5) == 0.5


# --- coverage properties (simulated exchangeable scores) -----------------------------------


@pytest.mark.parametrize(("n", "alpha"), [(49, 0.02), (99, 0.02), (19, 0.05), (12, 0.10)])
def test_mean_coverage_within_conformal_bounds(n: int, alpha: float) -> None:
    """Over TRIALS calibration sets + one new score: 1-a <= P(S_new <= q) <= 1-a+1/(n+1)."""
    rng = np.random.default_rng(n)
    cal = rng.exponential(size=(TRIALS, n))  # any continuous distribution: distribution-free
    new = rng.exponential(size=TRIALS)
    k = conformal_index(n, alpha)
    q = np.sort(cal, axis=1)[:, k - 1]
    covered = np.mean(new <= q)
    mc = 4 * math.sqrt(0.25 / TRIALS)  # 4 Monte-Carlo standard errors (worst case p = 0.5)
    assert 1 - alpha - mc <= covered <= 1 - alpha + 1 / (n + 1) + mc


@pytest.mark.parametrize(("n", "alpha"), [(49, 0.02), (90, 0.02), (19, 0.05)])
def test_realised_coverage_follows_beta(n: int, alpha: float) -> None:
    """Given one calibration set, coverage = F(q_hat) ~ Beta(k, n + 1 - k) (KS test)."""
    rng = np.random.default_rng(1000 + n)
    cal = rng.uniform(size=(TRIALS, n))  # F = identity, so coverage = q_hat itself
    k = conformal_index(n, alpha)
    realised = np.sort(cal, axis=1)[:, k - 1]
    assert stats.kstest(realised, coverage_distribution(n, k).cdf).pvalue > 0.001


def test_leaked_calibration_breaks_the_guarantee() -> None:
    """Calibrating on images the detector was fitted on (scores ~ 0) inflates false alarms."""
    rng = np.random.default_rng(7)
    fitted = rng.exponential(0.05, size=60)  # in-bank images: near-zero scores
    new_good = rng.exponential(1.0, size=5000)
    q = split_conformal_threshold(fitted, 0.02).threshold
    assert false_alarm_rate(new_good, q) > 0.5


def test_ties_keep_coverage_conservative() -> None:
    rng = np.random.default_rng(3)
    cal = rng.integers(0, 4, size=(TRIALS, 49)).astype(float)
    new = rng.integers(0, 4, size=TRIALS)
    q = np.sort(cal, axis=1)[:, conformal_index(49, 0.02) - 1]
    assert np.mean(new <= q) >= 0.98


# --- (alpha, delta) and intervals ----------------------------------------------------------


def test_alpha_delta_is_infeasible_for_mvtec_sizes_at_two_percent() -> None:
    for n in (49, 53, 78, 99):
        with pytest.raises(InfeasibleAlphaDeltaError, match="even k = n"):
            alpha_delta_index(n, 0.02, 0.1)
    assert alpha_delta_index(114, 0.02, 0.1) == 114  # 1 - 0.98^114 = 0.900...


def test_alpha_delta_index_is_minimal_and_more_conservative() -> None:
    n, alpha, delta = 99, 0.05, 0.1
    k = alpha_delta_index(n, alpha, delta)
    assert prob_coverage_at_least(n, k, 1 - alpha) >= 1 - delta
    assert prob_coverage_at_least(n, k - 1, 1 - alpha) < 1 - delta
    assert k >= conformal_index(n, alpha)
    scores = np.arange(n, dtype=float)
    t = alpha_delta_threshold(scores, alpha, delta)
    assert (t.method, t.k, t.delta, t.threshold) == ("alpha_delta", k, delta, float(k - 1))


@settings(max_examples=50)
@given(n=st.integers(20, 300), alpha=st.sampled_from([0.05, 0.1, 0.2]))
def test_alpha_delta_holds_in_simulation(n: int, alpha: float) -> None:
    try:
        k = alpha_delta_index(n, alpha, 0.1)
    except InfeasibleAlphaDeltaError:
        return
    rng = np.random.default_rng(n)
    realised = np.sort(rng.uniform(size=(2000, n)), axis=1)[:, k - 1]
    # P(coverage >= 1 - alpha) >= 0.9, with Monte-Carlo slack
    assert np.mean(realised >= 1 - alpha) >= 0.9 - 4 * math.sqrt(0.09 / 2000)


def test_clopper_pearson_known_values() -> None:
    lo, hi = clopper_pearson(0, 20)
    assert lo == 0 and hi == pytest.approx(0.16843, abs=1e-4)
    lo, hi = clopper_pearson(5, 10)
    assert (lo, hi) == pytest.approx((0.18709, 0.81291), abs=1e-4)
    assert clopper_pearson(10, 10)[1] == 1.0
    with pytest.raises(ValueError):
        clopper_pearson(3, 2)


def test_predicted_false_alarm_mean_and_simulation() -> None:
    n, m = 49, 20
    k = conformal_index(n, 0.02)
    p = predicted_false_alarm(n, k, m)
    assert p.mean == pytest.approx((n + 1 - k) / (n + 1))
    rng = np.random.default_rng(11)
    cal = rng.uniform(size=(TRIALS, n))
    q = np.sort(cal, axis=1)[:, k - 1]
    far = np.mean(rng.uniform(size=(TRIALS, m)) > q[:, None], axis=1)
    assert far.mean() == pytest.approx(p.mean, abs=4 * p.sd / math.sqrt(TRIALS))
    assert far.std() == pytest.approx(p.sd, rel=0.05)
    assert p.q05 <= p.mean <= p.q95
