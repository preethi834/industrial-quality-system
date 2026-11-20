"""Split-conformal thresholds (blueprint §2.8).

Given calibration scores S_1..S_n of the *true* labels (lower = more conforming) and a
miscoverage level alpha:

- **marginal threshold:** ``k = ceil((n + 1)(1 - alpha))``, ``q = S_(k)`` (k-th smallest).
  Under exchangeability ``P(S_new <= q) >= 1 - alpha`` over calibration set and new case.
  If ``k > n`` no finite threshold exists and ``InsufficientCalibrationError`` is raised
  (never a silent fallback). For alpha = 0.1 this needs n >= 9.
- **(alpha, delta) threshold:** for one *fixed* calibration set the realised coverage of
  ``S_(k)`` is Beta(k, n + 1 - k) (continuous scores). The smallest k with
  ``P(coverage >= 1 - alpha) >= 1 - delta`` is used, so the deployed threshold covers at
  least 1 - alpha with probability 1 - delta over the calibration draw.

**Ties make coverage conservative.** With discrete scores (our vote probabilities take
few distinct values) several calibration scores can equal S_(k); every label whose score
equals q is included, so ``P(S_new <= q) >= 1 - alpha`` still holds but the upper bound
``1 - alpha + 1/(n + 1)`` and the Beta law do not: coverage can be well above target.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from scipy.stats import beta as beta_dist


class InsufficientCalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class Threshold:
    qhat: float
    k: int
    n: int
    alpha: float
    delta: float | None = None  # set for the (alpha, delta) variant


def _check(scores: Sequence[float], alpha: float) -> list[float]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    xs = sorted(float(s) for s in scores)
    if any(not math.isfinite(x) for x in xs):
        raise ValueError("calibration scores must be finite")
    return xs


def marginal_k(n: int, alpha: float) -> int:
    return math.ceil((n + 1) * (1 - alpha) - 1e-12)


def conformal_threshold(scores: Sequence[float], alpha: float) -> Threshold:
    xs = _check(scores, alpha)
    n = len(xs)
    k = marginal_k(n, alpha)
    if k > n:
        need = math.ceil(1 / alpha) - 1
        raise InsufficientCalibrationError(
            f"n={n} calibration cases are too few for alpha={alpha} (need n >= {need})"
        )
    return Threshold(qhat=xs[k - 1], k=k, n=n, alpha=alpha)


def beta_k(n: int, alpha: float, delta: float) -> int:
    """Smallest k with P(Beta(k, n + 1 - k) >= 1 - alpha) >= 1 - delta; n + 1 if none."""
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    for k in range(1, n + 1):
        if float(beta_dist.cdf(1 - alpha, k, n + 1 - k)) <= delta:
            return k
    return n + 1


def conformal_threshold_pac(scores: Sequence[float], alpha: float, delta: float) -> Threshold:
    xs = _check(scores, alpha)
    n = len(xs)
    k = beta_k(n, alpha, delta)
    if k > n:
        raise InsufficientCalibrationError(
            f"n={n} calibration cases are too few for (alpha={alpha}, delta={delta})"
        )
    return Threshold(qhat=xs[k - 1], k=k, n=n, alpha=alpha, delta=delta)
