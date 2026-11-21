"""Scorer for conformal cause sets: refit of the vote parameters on a *fit* set.

Step 4 keeps the step 3 scorer (retrieval votes over confirmed past cases) and refits
its two vote parameters and the novelty threshold on the fit half of the v2 dev split.
The smoothing grid extends four decades below the step-3 grid so the optimum is not on
its edge (step 3b found alpha = 0.001, the smallest value tried).
"""

from __future__ import annotations

from collections.abc import Sequence

from iqs_diagnosis.domain.case import SymptomProfile
from iqs_diagnosis.retrieval.base import LibraryCase
from iqs_diagnosis.scoring.ranker import FitReport, fit_ranker

EXTENDED_NEIGHBOURS = (3, 5, 10, 20, 40)
EXTENDED_ALPHAS = (
    1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3,
)  # fmt: skip


def on_grid_edge(value: float, grid: Sequence[float]) -> bool:
    return value in (min(grid), max(grid))


def fit_scorer(
    library: Sequence[LibraryCase],
    causes: Sequence[str],
    fit_profiles: Sequence[SymptomProfile],
    fit_causes: Sequence[str],
    *,
    novelty_alpha: float = 0.05,
    fitted_on: dict[str, str] | None = None,
) -> FitReport:
    return fit_ranker(
        library,
        causes,
        fit_profiles,
        fit_causes,
        novelty_alpha=novelty_alpha,
        fitted_on=fitted_on,
        neighbours=EXTENDED_NEIGHBOURS,
        alphas=EXTENDED_ALPHAS,
    )
