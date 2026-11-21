"""Statistical validity checks for the templates (step 5 gate 2) and power summaries.

Gate 2: on null data (fault-free simulations) each template's false-positive rate over
1,000 simulations must be <= alpha + 3 SE, SE = sqrt(alpha (1 - alpha) / n). A run whose
template is inconclusive (not enough data) is not a positive; the summary also reports
the rate among conclusive runs, so a template cannot pass merely by abstaining.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from iqs_diagnosis.data.process_store import ProcessStore
from iqs_diagnosis.evaluation.metrics import Proportion, clopper_pearson
from iqs_diagnosis.scientist.templates import TEMPLATES, Window


def pseudo_window(store: ProcessStore, reference_hours: float, start_hour: float) -> Window:
    t0 = store.query(
        "SELECT min(completed_ts) AS t FROM read_parquet(?)",
        [str(store.observed_dir / "parts.parquet")],
    )["t"][0]
    base = pd.Timestamp(t0).floor("h")
    return Window(
        reference_end=base + pd.Timedelta(hours=reference_hours),
        affected_start=base + pd.Timedelta(hours=start_hour),
    )


def template_pvalues(observed_dir: Path, window: Window | None = None) -> dict[str, float | None]:
    """Every template's p-value on one run (``window`` default: 12 h reference, start 36 h)."""
    with ProcessStore(observed_dir) as store:
        w = window or pseudo_window(store, 12.0, 36.0)
        return {name: f(store, w).p_value for name, f in TEMPLATES.items()}


class NullSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    template: str
    n: int
    positives: int
    inconclusive: int
    false_positive_rate: Proportion  # over all runs (inconclusive = not positive)
    rate_among_conclusive: Proportion
    limit: float  # alpha + 3 SE
    passed: bool


def null_summary(
    pvalues: Sequence[Mapping[str, float | None]], *, alpha: float = 0.05
) -> list[NullSummary]:
    if not pvalues:
        raise ValueError("no null runs")
    n = len(pvalues)
    limit = alpha + 3 * math.sqrt(alpha * (1 - alpha) / n)
    out = []
    for name in sorted(pvalues[0]):
        ps = [row[name] for row in pvalues]
        pos = sum(p is not None and p <= alpha for p in ps)
        inc = sum(p is None for p in ps)
        rate = clopper_pearson(pos, n)
        out.append(
            NullSummary(
                template=name,
                n=n,
                positives=pos,
                inconclusive=inc,
                false_positive_rate=rate,
                rate_among_conclusive=clopper_pearson(pos, n - inc),
                limit=limit,
                passed=rate.value <= limit,
            )
        )
    return out
