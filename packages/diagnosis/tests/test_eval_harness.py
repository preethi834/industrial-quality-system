"""Deterministic evaluation subset (runs in CI with every push: no LLM, no Docker).

The library is the reference scenarios minus two held-out faults, and the queries are
all reference cases. Library cases are found by their own profile, so the accuracy here
checks the harness, not the method; the method is measured on the test split (3d).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from iqs_common.taxonomy import load_taxonomy
from iqs_diagnosis.evaluation.labels import load_case_label
from iqs_diagnosis.evaluation.run_eval import CaseResult, evaluate_cases, summarize
from iqs_diagnosis.ingestion.simulator import load_case
from iqs_diagnosis.retrieval.base import LibraryCase
from iqs_diagnosis.symptoms.vocabulary import load_vocabulary

from .agent_fixtures import NOVEL_SCENARIOS, PARAMS
from .conftest import TAXONOMY_PATH, VOCABULARY_PATH


def _results(
    reference_runs: dict[str, Path], reference_profiles: dict[str, Any]
) -> list[CaseResult]:
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    library = [
        LibraryCase(p, lab.cause_code)
        for name, (p, lab) in reference_profiles.items()
        if name not in NOVEL_SCENARIOS
    ]
    cases = []
    for name in sorted(reference_profiles):
        case, label = load_case(reference_runs[name]), load_case_label(reference_runs[name])
        assert case is not None and label is not None
        cases.append((case, label))
    return asyncio.run(
        evaluate_cases(
            cases,
            library=library,
            params=PARAMS,
            vocabulary=load_vocabulary(VOCABULARY_PATH),
            taxonomy=taxonomy,
        )
    )


def test_harness_measures_everything(
    reference_runs: dict[str, Path], reference_profiles: dict[str, Any]
) -> None:
    results = _results(reference_runs, reference_profiles)
    s = summarize(results, 10)
    assert (s.n_cases, s.n_known, s.n_novel) == (10, 8, 2)
    assert s.top1_known.k == 8  # library cases find themselves
    assert s.novelty_detection.k == 2 and s.false_novelty.k == 0
    assert s.tool_call_accuracy.k == s.tool_call_accuracy.n == 20
    assert s.completeness_full.k == 10
    assert s.random_expected == {"top1": 0.1, "top3": 0.3}
    assert set(s.baselines_known) == {"majority", "bm25_only", "vector_only"}
    assert s.llm_cost_usd == 0.0 and s.latency.n == 10
