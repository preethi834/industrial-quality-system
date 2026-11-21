"""Conformal evaluation helpers on synthetic, exchangeable cases."""

from __future__ import annotations

import numpy as np
import pytest

from iqs_diagnosis.conformal.scores import Method
from iqs_diagnosis.evaluation.conformal_eval import (
    ScoredCase,
    resplit_check,
    set_metrics,
    tie_stats,
)

CAUSES = ["A", "B", "C", "D"]


def _cases(n: int, seed: int) -> list[ScoredCase]:
    """Truth drawn from the predicted distribution itself: a calibrated scorer."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p = rng.dirichlet(np.ones(4) * 0.5)
        truth = CAUSES[int(rng.choice(4, p=p))]
        out.append(
            ScoredCase(
                case_id=f"c{i}",
                probabilities=dict(zip(CAUSES, map(float, p), strict=True)),
                truth=truth,
                secondary=(),
                known=True,
            )
        )
    return out


def test_set_metrics_hand_example() -> None:
    cases = [
        ScoredCase("a", {"A": 0.7, "B": 0.3}, "A", ("B",), True),
        ScoredCase("b", {"A": 0.4, "B": 0.6}, "A", (), True),
    ]
    m = set_metrics(cases, Method.LAC, 0.5, pairs=True)  # include p >= 0.5
    assert m.coverage.k == 1 and m.mean_size == 1.0 and m.empty.k == 0
    assert m.all_injected_covered is not None and m.all_injected_covered.k == 0
    m2 = set_metrics(cases, Method.LAC, 0.2)  # include p >= 0.8 -> both empty
    assert m2.empty.k == 2 and m2.size_histogram == {0: 2}


@pytest.mark.parametrize("method", list(Method))
def test_resplit_check_passes_on_exchangeable_cases(method: Method) -> None:
    r = resplit_check(_cases(600, 1), method, n_calibration=200, alpha=0.1, repeats=100)
    assert r.passed and abs(r.mean_coverage - 0.9) < 0.03
    assert r.n_pool == 600 and r.se > 0


def test_resplit_check_rejects_bad_input() -> None:
    cases = _cases(50, 2)
    with pytest.raises(ValueError):
        resplit_check(cases, Method.LAC, n_calibration=50, alpha=0.1)
    novel = [ScoredCase("x", {"A": 1.0}, "A", (), False)]
    with pytest.raises(ValueError):
        resplit_check(cases + novel, Method.LAC, n_calibration=10, alpha=0.1)


def test_tie_stats() -> None:
    t = tie_stats([0.1, 0.1, 0.1, 0.5])
    assert t["distinct"] == 2 and t["largest_tie"] == 3
