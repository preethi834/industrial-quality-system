"""Step 6a: cost config, decision policies and probability calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from scipy import stats

from iqs_vision.decision.calibrate_proba import (
    PlattCalibrator,
    brier,
    ece,
    fit_isotonic,
    fit_platt,
    shift_prior,
)
from iqs_vision.decision.costs import CostConfig, load_cost_config
from iqs_vision.decision.policy import (
    ACCEPT,
    REJECT,
    REVIEW,
    cut_offs,
    decide_guaranteed,
    decide_threshold,
    decide_unconstrained,
    expected_costs,
)

REPO = Path(__file__).resolve().parents[3]
EXAMPLE = REPO / "configs/costs/example_costs.yaml"


def _cfg(
    c_fr: float = 5.0,
    c_esc: float = 200.0,
    c_rev: float = 0.5,
    eps_miss: float = 0.05,
    eps_fr: float = 0.02,
    pi: float = 0.01,
) -> CostConfig:
    def a(v: float) -> dict[str, object]:
        return {"value": v, "note": "assumption: test"}

    return CostConfig.model_validate(
        {
            "id": "test",
            "cost_false_reject": a(c_fr),
            "cost_escape": a(c_esc),
            "cost_review": a(c_rev),
            "review_miss_rate": a(eps_miss),
            "review_false_reject_rate": a(eps_fr),
            "defect_rate": a(pi),
        }
    )


# --- cost config ----------------------------------------------------------------------------


def test_example_config_loads_with_hash() -> None:
    cfg, digest = load_cost_config(EXAMPLE)
    assert (cfg.c_fr, cfg.c_esc, cfg.c_rev) == (5.0, 200.0, 0.5)
    assert (cfg.eps_miss, cfg.eps_fr, cfg.pi) == (0.05, 0.02, 0.01)
    assert len(digest) == 64


def test_every_value_must_be_labelled_assumption() -> None:
    data = _cfg().model_dump()
    data["cost_escape"]["note"] = "measured at plant X"
    with pytest.raises(ValidationError, match="assumption"):
        CostConfig.model_validate(data)


@pytest.mark.parametrize(("field", "value"), [("pi", 0.0), ("pi", 1.0), ("eps_miss", 1.0)])
def test_rates_out_of_range_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        _cfg(**{field: value})


def test_with_values_keeps_notes() -> None:
    c = _cfg().with_values(cost_escape=1000.0)
    assert c.c_esc == 1000.0 and c.cost_escape.note.startswith("assumption")


# --- hand-computed cases --------------------------------------------------------------------


def test_cut_offs_hand_computed() -> None:
    k = cut_offs(_cfg())
    assert k.p_review_vs_accept == pytest.approx(0.6 / 190.1)
    assert k.p_review_vs_reject == pytest.approx(4.4 / 14.9)
    assert k.p_reject_vs_accept == pytest.approx(5 / 205)
    assert k.review_possible
    assert (k.lo, k.hi) == (pytest.approx(0.6 / 190.1), pytest.approx(4.4 / 14.9))


def test_expected_costs_hand_computed() -> None:
    a, r, v = expected_costs(0.1, _cfg())
    assert (a, r, v) == (pytest.approx(20.0), pytest.approx(4.5), pytest.approx(1.59))


def test_unconstrained_actions_around_cut_offs() -> None:
    k = cut_offs(_cfg())
    p = np.array([0.0, k.lo - 1e-9, k.lo, k.lo + 1e-9, 0.1, k.hi - 1e-9, k.hi, 1.0])
    got = decide_unconstrained(p, _cfg()).tolist()
    assert got == [ACCEPT, ACCEPT, ACCEPT, REVIEW, REVIEW, REVIEW, REJECT, REJECT]


def test_review_dropped_when_it_never_wins() -> None:
    c = _cfg(c_rev=10.0)  # a look costs more than scrapping the part
    k = cut_offs(c)
    assert not k.review_possible and k.lo == k.hi == pytest.approx(5 / 205)
    p = np.linspace(0, 1, 1001)
    got = decide_unconstrained(p, c)
    assert not (got == REVIEW).any()
    assert ((got == ACCEPT) == (p < 5 / 205)).all()


def test_guaranteed_policy_hand_computed() -> None:
    k = cut_offs(_cfg())
    p = np.array([0.9, 0.9, 0.1, k.p_review_vs_reject, 0.0])
    s = np.array([1.0, 2.0, 2.0, 2.0, 2.0])
    got = decide_guaranteed(p, s, threshold=1.0, c=_cfg()).tolist()
    # score <= q_hat accepted whatever p; above: review below p_rr, reject at/above it.
    # p = 0 above q_hat is reviewed, never accepted: accept is not allowed above q_hat.
    assert got == [ACCEPT, REJECT, REVIEW, REJECT, REVIEW]


def test_threshold_baseline() -> None:
    assert decide_threshold(np.array([0.5, 1.0, 1.5]), 1.0).tolist() == [ACCEPT, ACCEPT, REJECT]


def test_p_outside_unit_interval_rejected() -> None:
    with pytest.raises(ValueError):
        decide_unconstrained(np.array([1.2]), _cfg())


# --- invariants -----------------------------------------------------------------------------

P_GRID = np.concatenate([np.linspace(0, 1, 401), np.geomspace(1e-6, 0.5, 200)])
SCORES = np.random.default_rng(0).normal(size=P_GRID.size)

pos = st.floats(0.01, 1000.0)
rate = st.floats(0.0, 0.5)
factor = st.floats(1.0, 50.0)


def _share(codes: np.ndarray, action: int) -> float:
    return float(np.mean(codes == action))


@settings(max_examples=200, deadline=None)
@given(pos, pos, pos, rate, rate, factor)
def test_raising_review_cost_never_raises_review_share(
    fr: float, esc: float, rev: float, em: float, ef: float, f: float
) -> None:
    lo, hi = _cfg(fr, esc, rev, em, ef), _cfg(fr, esc, rev * f, em, ef)
    assert _share(decide_unconstrained(P_GRID, hi), REVIEW) <= _share(
        decide_unconstrained(P_GRID, lo), REVIEW
    )
    g_lo = decide_guaranteed(P_GRID, SCORES, 0.0, lo)
    g_hi = decide_guaranteed(P_GRID, SCORES, 0.0, hi)
    assert _share(g_hi, REVIEW) <= _share(g_lo, REVIEW)


@settings(max_examples=200, deadline=None)
@given(pos, pos, pos, rate, rate, factor)
def test_raising_escape_cost_never_raises_accept_share(
    fr: float, esc: float, rev: float, em: float, ef: float, f: float
) -> None:
    a = decide_unconstrained(P_GRID, _cfg(fr, esc, rev, em, ef))
    b = decide_unconstrained(P_GRID, _cfg(fr, esc * f, rev, em, ef))
    assert _share(b, ACCEPT) <= _share(a, ACCEPT)


@settings(max_examples=200, deadline=None)
@given(pos, pos, pos, rate, rate, factor)
def test_raising_false_reject_cost_never_raises_reject_share(
    fr: float, esc: float, rev: float, em: float, ef: float, f: float
) -> None:
    a, b = _cfg(fr, esc, rev, em, ef), _cfg(fr * f, esc, rev, em, ef)
    assert _share(decide_unconstrained(P_GRID, b), REJECT) <= _share(
        decide_unconstrained(P_GRID, a), REJECT
    )
    g_a = decide_guaranteed(P_GRID, SCORES, 0.0, a)
    g_b = decide_guaranteed(P_GRID, SCORES, 0.0, b)
    assert _share(g_b, REJECT) <= _share(g_a, REJECT)


@settings(max_examples=100, deadline=None)
@given(pos, pos)
def test_free_perfect_review_takes_everything_uncertain(fr: float, esc: float) -> None:
    got = decide_unconstrained(P_GRID, _cfg(fr, esc, 0.0, 0.0, 0.0))
    inner = (P_GRID > 0) & (P_GRID < 1)
    assert (got[inner] == REVIEW).all()
    assert (got[P_GRID == 0] == ACCEPT).all() and (got[P_GRID == 1] == REJECT).all()


@settings(max_examples=200, deadline=None)
@given(pos, pos, pos, rate, rate, st.floats(-2.0, 2.0))
def test_guaranteed_never_flags_at_or_below_threshold(
    fr: float, esc: float, rev: float, em: float, ef: float, q: float
) -> None:
    got = decide_guaranteed(P_GRID, SCORES, q, _cfg(fr, esc, rev, em, ef))
    assert (got[SCORES <= q] == ACCEPT).all()
    assert (got[SCORES > q] != ACCEPT).all()


@settings(max_examples=200, deadline=None)
@given(pos, pos, pos, rate, rate)
def test_unconstrained_matches_brute_force_argmin(
    fr: float, esc: float, rev: float, em: float, ef: float
) -> None:
    c = _cfg(fr, esc, rev, em, ef)
    costs = expected_costs(P_GRID, c)
    got = decide_unconstrained(P_GRID, c)
    chosen = np.take_along_axis(costs, got.astype(np.int64)[:, None], axis=1)[:, 0]
    assert np.all(chosen <= costs.min(axis=1) + 1e-9 * (1 + costs.max(axis=1)))


# --- calibrators ----------------------------------------------------------------------------


def _two_gaussians(n_good: int, n_def: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = np.concatenate([rng.normal(0, 1, n_good), rng.normal(2, 1, n_def)])
    return s, np.r_[np.zeros(n_good, int), np.ones(n_def, int)]


def test_prior_shift_is_exact_on_known_densities() -> None:
    s = np.linspace(-3, 5, 81)
    f0, f1 = stats.norm.pdf(s, 0, 1), stats.norm.pdf(s, 2, 1)

    def post(pi: float) -> np.ndarray:
        return np.asarray(pi * f1 / (pi * f1 + (1 - pi) * f0))

    for pi_fit, pi in [(0.6, 0.01), (0.3, 0.05), (0.5, 0.5)]:
        np.testing.assert_allclose(shift_prior(post(pi_fit), pi_fit, pi), post(pi), rtol=1e-6)


@settings(max_examples=50, deadline=None)
@given(st.integers(5, 60), st.integers(5, 60), st.integers(0, 10_000), st.booleans())
def test_calibrators_are_monotone_in_score(ng: int, nd: int, seed: int, flip: bool) -> None:
    s, y = _two_gaussians(ng, nd, seed)
    if flip:  # defects score lower: Platt slope is clamped at 0, p stays non-decreasing
        y = 1 - y
    grid = np.linspace(s.min() - 1, s.max() + 1, 300)
    for cal in (fit_platt(s, y), fit_isotonic(s, y)):
        for pi in (0.01, 0.5):
            assert np.all(np.diff(cal.predict(grid, pi)) >= -1e-12)


def test_platt_recovers_logistic_model() -> None:
    # Two unit-variance Gaussians at 0 and 2: true log-odds = 2 s - 2 (at equal priors).
    s, y = _two_gaussians(4000, 4000, 1)
    cal = fit_platt(s, y)
    grid = np.linspace(-1, 3, 9)
    np.testing.assert_allclose(cal.predict(grid, 0.5), stats.logistic.cdf(2 * grid - 2), atol=0.03)


def test_platt_finite_on_perfectly_separated_data() -> None:
    s = np.r_[np.linspace(0, 1, 20), np.linspace(5, 6, 20)]
    y = np.r_[np.zeros(20, int), np.ones(20, int)]
    cal = fit_platt(s, y)
    assert np.isfinite(cal.a) and cal.a > 0
    assert cal.predict_fit_prior(np.array([0.0]))[0] > 0


def test_platt_round_trips_through_dict() -> None:
    s, y = _two_gaussians(30, 30, 2)
    cal = fit_platt(s, y)
    again = PlattCalibrator.from_dict(cal.to_dict())
    np.testing.assert_array_equal(again.predict(s, 0.01), cal.predict(s, 0.01))


def test_fit_needs_both_classes() -> None:
    with pytest.raises(ValueError, match="both"):
        fit_platt(np.array([1.0, 2.0]), np.array([0, 0]))


def test_brier_and_ece_hand_computed() -> None:
    p, y = np.array([0.0, 1.0, 0.5, 0.5]), np.array([0, 1, 0, 1])
    assert brier(p, y) == pytest.approx(0.125)
    assert ece(p, y) == pytest.approx(0.0)
    assert ece(np.array([0.9, 0.9]), np.array([0, 0])) == pytest.approx(0.9)
