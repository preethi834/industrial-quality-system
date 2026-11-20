"""Scientist runner: test several cause hypotheses for one case and correct them together.

For each hypothesised cause the taxonomy's template is run on the case's observable
data. Conclusive p-values are adjusted with Benjamini-Hochberg (FDR q, default 0.05) over
the hypotheses tested *for this case*; inconclusive templates do not enter the correction.

Verdicts:
- ``supported``: adjusted p <= q;
- ``rejected``: enough data, adjusted p > q ("no evidence in the data", not proof that the
  cause is absent);
- ``inconclusive``: the template's minimum group sizes were not met.

The scientist does not change the cause probabilities or the conformal set (they are
calibrated as they are); it adds evidence and names the highest-ranked supported cause.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict
from statsmodels.stats.multitest import multipletests

from iqs_diagnosis.data.process_store import ProcessStore
from iqs_diagnosis.domain.case import DiagnosisCase
from iqs_diagnosis.scientist.templates import TEMPLATES, TemplateResult, Window

Verdict = Literal["supported", "rejected", "inconclusive"]


class HypothesisResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cause_code: str
    rank: int  # position in the hypothesis list (1 = agent's top cause)
    result: TemplateResult
    p_adjusted: float | None
    verdict: Verdict


class ScientistReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    q: float
    reference_end: str
    affected_start: str
    hypotheses: list[HypothesisResult]
    data_supported_cause: str | None  # highest-ranked supported hypothesis


def window_for_case(case: DiagnosisCase, store: ProcessStore) -> Window:
    if case.window_start is None or case.baseline_hours is None:
        raise ValueError(f"{case.case_id}: needs window_start and baseline_hours")
    t0 = store.query(
        "SELECT min(completed_ts) AS t FROM read_parquet(?)",
        [str(store.observed_dir / "parts.parquet")],
    )["t"][0]
    return Window(
        reference_end=pd.Timestamp(t0).floor("h") + pd.Timedelta(hours=case.baseline_hours),
        affected_start=pd.Timestamp(case.window_start),
    )


def bh_adjust(pvalues: Sequence[float | None], q: float) -> list[float | None]:
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    out: list[float | None] = [None] * len(pvalues)
    if idx:
        adj = multipletests([pvalues[i] for i in idx], alpha=q, method="fdr_bh")[1]
        for i, a in zip(idx, adj, strict=True):
            out[i] = float(a)
    return out


def run_hypotheses(
    store: ProcessStore,
    window: Window,
    hypotheses: Sequence[str],
    templates_by_cause: Mapping[str, str],
    *,
    case_id: str,
    q: float = 0.05,
) -> ScientistReport:
    if len(set(hypotheses)) != len(hypotheses):
        raise ValueError("duplicate hypotheses")
    results = [TEMPLATES[templates_by_cause[c]](store, window) for c in hypotheses]
    adjusted = bh_adjust([r.p_value for r in results], q)
    out: list[HypothesisResult] = []
    for rank, (cause, r, a) in enumerate(zip(hypotheses, results, adjusted, strict=True), 1):
        verdict: Verdict = "inconclusive" if a is None else "supported" if a <= q else "rejected"
        out.append(
            HypothesisResult(cause_code=cause, rank=rank, result=r, p_adjusted=a, verdict=verdict)
        )
    supported = [h for h in out if h.verdict == "supported"]
    return ScientistReport(
        case_id=case_id,
        q=q,
        reference_end=str(window.reference_end),
        affected_start=str(window.affected_start),
        hypotheses=out,
        data_supported_cause=supported[0].cause_code if supported else None,
    )


def test_case(
    case: DiagnosisCase,
    hypotheses: Sequence[str],
    templates_by_cause: Mapping[str, str],
    *,
    q: float = 0.05,
) -> ScientistReport:
    if case.observed_dir is None:
        raise ValueError(f"{case.case_id}: no observable process data")
    with ProcessStore(case.observed_dir) as store:
        return run_hypotheses(
            store,
            window_for_case(case, store),
            hypotheses,
            templates_by_cause,
            case_id=case.case_id,
            q=q,
        )


test_case.__test__ = False  # type: ignore[attr-defined]  # not a pytest test
