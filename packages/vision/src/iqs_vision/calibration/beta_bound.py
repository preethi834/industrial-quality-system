"""Calibration-conditional view of split conformal (blueprint Rev 2 §2.8).

For one fixed calibration set the realised coverage C = P(S_new <= q_hat | calibration) is
random. With continuous scores C ~ Beta(k, n + 1 - k). The (alpha, delta) threshold picks the
smallest k with P(C >= 1 - alpha) >= 1 - delta, so that a deployed threshold keeps the
false-alarm rate <= alpha with probability >= 1 - delta. It needs many more calibration
images than the marginal one: even k = n gives P(C >= 1 - alpha) = 1 - (1 - alpha)^n, so
alpha = 0.02, delta = 0.1 needs n >= 114.

On a finite test set of m good images the realised false-alarm count is X | C ~ Bin(m, 1 - C),
so X ~ BetaBinomial(m, n + 1 - k, k): `predicted_false_alarm` gives its mean and spread, the
reference for the repeated-split analysis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from iqs_vision.calibration.conformal import (
    ConformalThreshold,
    InsufficientCalibrationError,
    _check_alpha,
    _scores,
)


class InfeasibleAlphaDeltaError(InsufficientCalibrationError):
    """No rank k <= n reaches P(coverage >= 1 - alpha) >= 1 - delta."""


def _check_nk(n: int, k: int) -> None:
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k = {k}, n = {n}")


def coverage_distribution(n: int, k: int) -> stats.rv_continuous:
    """Beta(k, n + 1 - k): realised coverage of the k-th order statistic (continuous scores)."""
    _check_nk(n, k)
    return stats.beta(k, n + 1 - k)


def prob_coverage_at_least(n: int, k: int, level: float) -> float:
    """P(C >= level) for C ~ Beta(k, n + 1 - k)."""
    return float(coverage_distribution(n, k).sf(level))


def alpha_delta_index(n: int, alpha: float, delta: float) -> int:
    """Smallest k with P(C >= 1 - alpha) >= 1 - delta; raises if even k = n is not enough."""
    _check_alpha(alpha)
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if n < 1:
        raise InsufficientCalibrationError("no calibration scores")
    ks = np.arange(1, n + 1)
    ok = stats.beta.sf(1 - alpha, ks, n + 1 - ks) >= 1 - delta
    if not ok.any():
        best = prob_coverage_at_least(n, n, 1 - alpha)
        raise InfeasibleAlphaDeltaError(
            f"n = {n} is too few for (alpha = {alpha}, delta = {delta}): even k = n gives "
            f"P(coverage >= {1 - alpha:g}) = {best:.3f} < {1 - delta:g}"
        )
    return int(ks[np.argmax(ok)])


def alpha_delta_threshold(
    scores: Sequence[float] | np.ndarray, alpha: float, delta: float
) -> ConformalThreshold:
    s = _scores(scores)
    k = alpha_delta_index(int(s.size), alpha, delta)
    return ConformalThreshold("alpha_delta", alpha, int(s.size), k, float(s[k - 1]), delta)


def clopper_pearson(x: int, m: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact two-sided interval for a binomial proportion x / m."""
    if m < 1 or not 0 <= x <= m:
        raise ValueError(f"need 0 <= x <= m, m >= 1; got x = {x}, m = {m}")
    a = 1 - confidence
    lo = 0.0 if x == 0 else float(stats.beta.ppf(a / 2, x, m - x + 1))
    hi = 1.0 if x == m else float(stats.beta.ppf(1 - a / 2, x + 1, m - x))
    return lo, hi


@dataclass(frozen=True, slots=True)
class FalseAlarmPrediction:
    """Distribution of the realised false-alarm rate X / m on m new good images."""

    n: int
    k: int
    m: int
    mean: float
    sd: float
    q05: float
    q95: float


def predicted_false_alarm(n: int, k: int, m: int) -> FalseAlarmPrediction:
    """X ~ BetaBinomial(m, n + 1 - k, k); rates are X / m."""
    _check_nk(n, k)
    if m < 1:
        raise ValueError("m must be >= 1")
    d = stats.betabinom(m, n + 1 - k, k)
    return FalseAlarmPrediction(
        n=n,
        k=k,
        m=m,
        mean=float(d.mean()) / m,
        sd=float(d.std()) / m,
        q05=float(d.ppf(0.05)) / m,
        q95=float(d.ppf(0.95)) / m,
    )
