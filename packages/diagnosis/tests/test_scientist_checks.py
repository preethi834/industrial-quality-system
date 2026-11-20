"""Gate-2 summary logic (the 1,000-run check itself is scripts/check_scientist_null.py)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from iqs_diagnosis.scientist.checks import null_summary, template_pvalues
from iqs_diagnosis.scientist.templates import TEMPLATES


def test_null_summary_limits_and_inconclusive() -> None:
    rows = [{"t": 0.01}] * 60 + [{"t": 0.5}] * 900 + [{"t": None}] * 40  # n = 1000
    (s,) = null_summary(rows)
    limit = 0.05 + 3 * math.sqrt(0.05 * 0.95 / 1000)
    assert s.limit == pytest.approx(limit)
    assert (s.positives, s.inconclusive) == (60, 40)
    assert s.false_positive_rate.value == pytest.approx(0.06) and s.passed  # 0.06 <= 0.0707
    assert s.rate_among_conclusive.n == 960
    (bad,) = null_summary([{"t": 0.01}] * 80 + [{"t": 0.9}] * 920)
    assert not bad.passed
    with pytest.raises(ValueError):
        null_summary([])


def test_template_pvalues_on_baseline(reference_runs: dict[str, Path]) -> None:
    ps = template_pvalues(reference_runs["baseline"] / "observed")
    assert set(ps) == set(TEMPLATES)
    assert all(p is None or 0.0 <= p <= 1.0 for p in ps.values())


def test_rate_templates_are_valid_when_groups_exist() -> None:
    """Null check for the Fisher-based templates on synthetic tables.

    In fault-free simulations the tool-wear, parameter-set, ambient and (mostly) shift
    templates are inconclusive, because their "exposed" group does not exist, so the
    1,000-run check is vacuous for them. Here the groups exist and both groups have the
    same reject probability: the Bonferroni-over-targets p must still have FP <= alpha.
    """
    import numpy as np

    from iqs_diagnosis.scientist.templates import _rate_result

    rng = np.random.default_rng(5)
    alpha, trials, hits = 0.05, 2000, 0
    for _ in range(trials):
        tables = {}
        for t in ("M1", "M2A", "M3"):
            n1, n2 = 60, 900
            tables[t] = (int(rng.binomial(n1, 0.05)), n1, int(rng.binomial(n2, 0.05)), n2)
        r = _rate_result("t", "SELECT 1", [], tables, ("a", "b"), "t")
        hits += r.p_value is not None and r.p_value <= alpha
    se = math.sqrt(alpha * (1 - alpha) / trials)
    assert hits / trials <= alpha + 3 * se


def test_batch_intersection_union_is_valid() -> None:
    """IUT p = max(p_prev, p_next): under the null its FP rate stays <= alpha."""
    import numpy as np

    from iqs_diagnosis.scientist.templates import _fisher_greater

    rng = np.random.default_rng(6)
    trials, hits = 3000, 0
    for _ in range(trials):
        k = rng.binomial(150, 0.03, size=3)
        p = max(
            _fisher_greater(int(k[1]), 150, int(k[0]), 150),
            _fisher_greater(int(k[1]), 150, int(k[2]), 150),
        )
        hits += p <= 0.05
    assert hits / trials <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / trials)
