"""Conformal cause-set evaluation (step 4 gates 2 and 3, recorded metrics).

Inputs are per-case probability vectors with the primary cause, secondary causes and
whether the primary cause is in the library (``known``):

- **known** cases are the population the guarantee is about (closed label space);
- **novel** cases (primary cause held out of the library) are a stress test: they are
  not exchangeable with the calibration set by construction, so their coverage is
  reported with the label "stress test, no guarantee" (§2.8 point 3).

Gate 2 (statistical validity, §5 step 4): pool calibration and test known cases,
redraw R random calibration/test splits with the frozen calibration size, recompute the
threshold, and check ``mean coverage >= 1 - alpha - 3 * SE`` with SE = sd / sqrt(R).
The resplits overlap, so SE understates the true uncertainty a little; the rule is the
blueprint's.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict

from iqs_diagnosis.conformal.scores import Method, prediction_set, score
from iqs_diagnosis.conformal.split import conformal_threshold
from iqs_diagnosis.evaluation.metrics import Proportion, clopper_pearson


@dataclass(frozen=True)
class ScoredCase:
    case_id: str
    probabilities: Mapping[str, float]
    truth: str
    secondary: tuple[str, ...]
    known: bool


class SetMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    coverage: Proportion
    mean_size: float
    size_histogram: dict[int, int]
    empty: Proportion
    all_injected_covered: Proportion | None = None  # two-fault cases only


def set_metrics(
    cases: Sequence[ScoredCase], method: Method, qhat: float, *, pairs: bool = False
) -> SetMetrics:
    sets = [prediction_set(method, c.probabilities, qhat) for c in cases]
    n = len(cases)
    hits = sum(c.truth in s for c, s in zip(cases, sets, strict=True))
    sizes = [len(s) for s in sets]
    hist: dict[int, int] = {}
    for k in sizes:
        hist[k] = hist.get(k, 0) + 1
    all_inj = (
        clopper_pearson(
            sum({c.truth, *c.secondary} <= set(s) for c, s in zip(cases, sets, strict=True)), n
        )
        if pairs
        else None
    )
    return SetMetrics(
        n=n,
        coverage=clopper_pearson(hits, n),
        mean_size=(sum(sizes) / n) if n else math.nan,
        size_histogram=dict(sorted(hist.items())),
        empty=clopper_pearson(sum(k == 0 for k in sizes), n),
        all_injected_covered=all_inj,
    )


class ResplitResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Method
    n_pool: int
    n_calibration: int
    repeats: int
    mean_coverage: float
    sd_coverage: float
    se: float
    lower_rule: float  # 1 - alpha - 3 se
    passed: bool
    mean_set_size: float


def resplit_check(
    pool: Sequence[ScoredCase],
    method: Method,
    *,
    n_calibration: int,
    alpha: float,
    repeats: int = 100,
    seed: int = 0,
) -> ResplitResult:
    if not all(c.known for c in pool):
        raise ValueError("the resplit check uses known-cause cases only")
    if not 0 < n_calibration < len(pool):
        raise ValueError("n_calibration must leave test cases")
    scores = np.array([score(method, c.probabilities, c.truth) for c in pool])
    rng = np.random.default_rng(seed)
    covs, sizes = [], []
    for _ in range(repeats):
        perm = rng.permutation(len(pool))
        cal, test = perm[:n_calibration], perm[n_calibration:]
        q = conformal_threshold(list(scores[cal]), alpha).qhat
        covs.append(float(np.mean(scores[test] <= q + 1e-12)))
        sizes.append(
            float(np.mean([len(prediction_set(method, pool[i].probabilities, q)) for i in test]))
        )
    mean, sd = float(np.mean(covs)), float(np.std(covs, ddof=1))
    se = sd / math.sqrt(repeats)
    lower = 1 - alpha - 3 * se
    return ResplitResult(
        method=method,
        n_pool=len(pool),
        n_calibration=n_calibration,
        repeats=repeats,
        mean_coverage=mean,
        sd_coverage=sd,
        se=se,
        lower_rule=lower,
        passed=mean >= lower,
        mean_set_size=float(np.mean(sizes)),
    )


def tie_stats(scores: Sequence[float]) -> dict[str, float]:
    """How discrete the calibration scores are (ties make coverage conservative)."""
    rounded = [round(s, 12) for s in scores]
    distinct = len(set(rounded))
    top = max((rounded.count(v) for v in set(rounded)), default=0)
    return {
        "n": len(scores),
        "distinct": distinct,
        "share_distinct": distinct / len(scores) if scores else math.nan,
        "largest_tie": top,
    }
