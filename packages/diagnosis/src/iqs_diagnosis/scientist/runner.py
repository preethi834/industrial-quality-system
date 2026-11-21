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
calibrated as they are); it adds evidence and names one data-supported cause:

- ``mode="top3"`` (default): the agent's top 3 are tested and the pick is the
  highest-ranked supported hypothesis;
- ``mode="all_causes"`` (used when the case is flagged novel; post-evaluation change, see
  docs/notes/step-5b.md): every taxonomy cause is tested, BH over all of them, and the
  pick is the supported cause with the smallest adjusted p (ties: rank). The ranking is
  not trusted for a new problem, so rank does not decide the pick.
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
Mode = Literal["top3", "all_causes"]


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
    mode: Mode = "top3"
    reference_end: str
    affected_start: str
    hypotheses: list[HypothesisResult]
    supported_causes: list[str] = []  # every supported hypothesis, in rank order
    data_supported_cause: str | None  # the pick (rule depends on ``mode``)


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
    mode: Mode = "top3",
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
    if mode == "all_causes":
        supported_by_p = sorted(supported, key=lambda h: (h.p_adjusted or 1.0, h.rank))
        pick = supported_by_p[0].cause_code if supported_by_p else None
    else:
        pick = supported[0].cause_code if supported else None
    return ScientistReport(
        case_id=case_id,
        q=q,
        mode=mode,
        reference_end=str(window.reference_end),
        affected_start=str(window.affected_start),
        hypotheses=out,
        supported_causes=[h.cause_code for h in supported],
        data_supported_cause=pick,
    )


def hypotheses_for(
    top3: Sequence[str], all_causes: Sequence[str], *, novel: bool
) -> tuple[list[str], Mode]:
    """Top 3 for a known-looking case; for a novel case the top 3 first, then every other
    cause in taxonomy order (all causes tested)."""
    if not novel:
        return list(top3), "top3"
    return [*top3, *(c for c in all_causes if c not in top3)], "all_causes"


def test_case(
    case: DiagnosisCase,
    hypotheses: Sequence[str],
    templates_by_cause: Mapping[str, str],
    *,
    q: float = 0.05,
    mode: Mode = "top3",
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
            mode=mode,
        )


test_case.__test__ = False  # type: ignore[attr-defined]  # not a pytest test
