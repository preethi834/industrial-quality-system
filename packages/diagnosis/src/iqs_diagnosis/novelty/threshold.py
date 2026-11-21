"""Novelty check: "no close past case" when the best library similarity is too low.

The threshold is set on dev cases whose cause *is* in the library (known cases), so
that a known case is wrongly called novel with probability at most ``alpha``:
with ``n`` known-case scores, take ``t`` = the ``k``-th smallest, ``k = floor(alpha (n+1))``,
and flag a case as novel when its score is strictly below ``t``. If a new known
case is exchangeable with the dev known cases (and there are no ties), its rank
among the ``n+1`` scores is uniform, so ``P(score < t) <= k / (n+1) <= alpha``.
If ``k`` is 0 (too few dev cases for that alpha) nothing is ever flagged.

This bounds false novelty only; how many truly novel cases are caught (detection
rate) is an empirical result measured on held-out cause types.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class NoveltyThreshold:
    threshold: float  # -inf means "never flag"
    alpha: float
    n_calibration: int

    def is_novel(self, max_similarity: float) -> bool:
        return max_similarity < self.threshold


def fit_novelty_threshold(known_scores: Sequence[float], alpha: float) -> NoveltyThreshold:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    scores = sorted(float(s) for s in known_scores)
    if not scores or not all(math.isfinite(s) for s in scores):
        raise ValueError("need at least one finite known-case score")
    n = len(scores)
    k = math.floor(alpha * (n + 1))
    t = -math.inf if k == 0 else scores[k - 1]
    return NoveltyThreshold(threshold=t, alpha=alpha, n_calibration=n)


@dataclass(frozen=True)
class NoveltyRates:
    false_novelty_rate: float  # known cases flagged as novel
    detection_rate: float  # novel cases flagged as novel
    n_known: int
    n_novel: int


def novelty_rates(
    detector: NoveltyThreshold, known: Sequence[float], novel: Sequence[float]
) -> NoveltyRates:
    def share(xs: Sequence[float]) -> float:
        return sum(detector.is_novel(x) for x in xs) / len(xs) if xs else math.nan

    return NoveltyRates(
        false_novelty_rate=share(known),
        detection_rate=share(novel),
        n_known=len(known),
        n_novel=len(novel),
    )
