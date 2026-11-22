"""Split-conformal thresholds for anomaly scores (blueprint Rev 2 §2.8).

Setup: a fixed score function (the fitted detector) and n calibration scores S_1..S_n of good
images that the detector never saw. For a new good image, under exchangeability (A1),

    P(S_{n+1} <= q_hat) >= 1 - alpha,   q_hat = S_(k),  k = ceil((n + 1)(1 - alpha))

so P(false alarm) = P(S_{n+1} > q_hat) <= alpha. With no ties also <= alpha + 1/(n + 1).
If k > n no finite threshold exists: we refuse (n must be >= ceil(1/alpha) - 1, i.e. 49 for
alpha = 0.02) instead of silently returning +inf or the maximum.

The guarantee is marginal (over calibration set and new image) and per group: calibrate per
category and per data source. It says nothing about the miss rate, which is measured.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

# (n + 1)(1 - alpha) is often an integer in exact arithmetic (n = 49, alpha = 0.02 -> 49) but
# a float a few ulps above it; without this slack ceil() would give k = n + 1.
_EPS = 1e-9

Method = Literal["marginal", "alpha_delta"]


class InsufficientCalibrationError(ValueError):
    """Too few calibration scores for a finite threshold at the requested level."""


def _check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def min_calibration_size(alpha: float) -> int:
    """Smallest n with a finite marginal threshold: ceil(1/alpha) - 1 (49 for alpha = 0.02)."""
    _check_alpha(alpha)
    return math.ceil(1 / alpha - _EPS) - 1


def conformal_index(n: int, alpha: float) -> int:
    """k = ceil((n + 1)(1 - alpha)), 1-based rank of the threshold among sorted scores."""
    _check_alpha(alpha)
    if n < 1:
        raise InsufficientCalibrationError("no calibration scores")
    k = max(1, math.ceil((n + 1) * (1 - alpha) - _EPS))
    if k > n:
        raise InsufficientCalibrationError(
            f"n = {n} calibration scores is too few for alpha = {alpha}: k = {k} > n; "
            f"need n >= {min_calibration_size(alpha)}"
        )
    return k


@dataclass(frozen=True, slots=True)
class ConformalThreshold:
    method: Method
    alpha: float
    n: int
    k: int
    threshold: float
    delta: float | None = None  # only for method = "alpha_delta"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _scores(scores: Sequence[float] | np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.size and not np.all(np.isfinite(s)):
        raise ValueError("calibration scores must be finite")
    return np.sort(s)


def threshold_at_rank(scores: Sequence[float] | np.ndarray, k: int) -> float:
    """The k-th smallest score (1-based)."""
    s = _scores(scores)
    if not 1 <= k <= s.size:
        raise InsufficientCalibrationError(f"rank k = {k} outside 1..{s.size}")
    return float(s[k - 1])


def split_conformal_threshold(
    scores: Sequence[float] | np.ndarray, alpha: float
) -> ConformalThreshold:
    """Marginal split-conformal threshold: flag a part when its score is > threshold."""
    s = _scores(scores)
    k = conformal_index(s.size, alpha)
    return ConformalThreshold("marginal", alpha, int(s.size), k, float(s[k - 1]))


def false_alarm_rate(good_scores: Sequence[float] | np.ndarray, threshold: float) -> float:
    """Fraction of good images flagged (score strictly above the threshold)."""
    g = np.asarray(good_scores, dtype=np.float64)
    if g.size == 0:
        raise ValueError("no good images")
    return float(np.mean(g > threshold))


def miss_rate(defect_scores: Sequence[float] | np.ndarray, threshold: float) -> float:
    """Fraction of defective images accepted (score at or below the threshold)."""
    d = np.asarray(defect_scores, dtype=np.float64)
    if d.size == 0:
        raise ValueError("no defective images")
    return float(np.mean(d <= threshold))
