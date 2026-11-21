"""Evaluation harness: run the full agent on simulator cases and measure it (step 3d).

Each case goes through the whole graph: symptom extraction from ``observed/``, retrieval,
novelty, scoring, 8D draft, ``create_ticket`` over MCP (in-process mock QMS), then a
simulated engineer confirms the *true* cause and the agent calls ``update_ticket``.

Measured: top-1 / top-3 accuracy on cases whose cause is in the library (95 % Wilson
intervals), per fault, novelty (false novelty on known causes, detection on held-out
causes), tool-call accuracy (arguments match the agent's own result and the QMS ticket
afterwards holds them), 8D completeness, latency to the confirmation point, LLM cost
(zero: no LLM in this version).

Baselines on the same cases: random over the 10 taxonomy causes (analytic), majority
(most frequent library causes), BM25-only and vector-only retrieval with the same votes.

Simulator accuracy is not real-world accuracy: faults are injected into a model whose
signals were designed to make them observable.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from iqs_common.taxonomy import Taxonomy
from iqs_diagnosis.agent.checkpoint import memory_checkpointer
from iqs_diagnosis.agent.graph import AgentDeps, DiagnosisAgent, severity_for
from iqs_diagnosis.agent.tickets import McpTicketGateway
from iqs_diagnosis.domain.case import DiagnosisCase, SymptomProfile
from iqs_diagnosis.evaluation.labels import CaseLabel, load_case_label
from iqs_diagnosis.evaluation.metrics import Latency, Proportion, latency, wilson
from iqs_diagnosis.ingestion.simulator import iter_run_dirs, load_case
from iqs_diagnosis.persistence.repository import InMemoryCaseRepository
from iqs_diagnosis.retrieval.base import LibraryCase
from iqs_diagnosis.scoring.ranker import CauseRanker, RankerParams
from iqs_diagnosis.scoring.votes import top_k, vote_probabilities
from iqs_diagnosis.symptoms.vocabulary import Vocabulary

ENGINEER = "eval-simulated-engineer"


class CaseResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    truth: str  # primary cause
    simulator_fault: str
    secondary: list[str] = []  # other injected causes (two-fault cases)
    known: bool  # truth is among the library's causes
    top3: list[str]
    novel: bool
    max_similarity: float
    completeness: float
    tool_calls_correct: int
    tool_calls_total: int
    latency_ms: float
    baselines: dict[str, list[str]]


def collect_cases(runs_dir: Path) -> list[tuple[DiagnosisCase, CaseLabel]]:
    out: list[tuple[DiagnosisCase, CaseLabel]] = []
    for run_dir in iter_run_dirs(runs_dir):
        case = load_case(run_dir)
        if case is None:
            continue
        label = load_case_label(run_dir)
        if label is None or label.case_id != case.case_id:
            raise ValueError(f"{run_dir}: label does not match case")
        out.append((case, label))
    return sorted(out, key=lambda cl: cl[0].case_id)


def _check_create(
    call: dict[str, Any], case: DiagnosisCase, out: dict[str, Any], store: Any
) -> bool:
    a = call["arguments"]
    if not (
        call["tool"] == "create_ticket"
        and call["ok"]
        and a["case_id"] == case.case_id
        and a["candidate_causes"] == [c for c, _ in out["top3"]]
        and a["novel"] == out["novel"]
        and a["severity"] == severity_for(case)
    ):
        return False
    t = store.get(out["ticket"]["ticket_id"])
    return bool(t.case_id == case.case_id and list(t.candidate_causes) == a["candidate_causes"])


def _check_update(call: dict[str, Any], ticket_id: str, truth: str, store: Any) -> bool:
    a = call["arguments"]
    if not (
        call["tool"] == "update_ticket"
        and call["ok"]
        and a["ticket_id"] == ticket_id
        and a.get("confirmed_cause") == truth
    ):
        return False
    t = store.get(ticket_id)
    return bool(t.status == "cause_confirmed" and t.confirmed_cause == truth)


async def evaluate_cases(
    cases: Sequence[tuple[DiagnosisCase, CaseLabel]],
    *,
    library: Sequence[LibraryCase],
    params: RankerParams,
    vocabulary: Vocabulary,
    taxonomy: Taxonomy,
) -> list[CaseResult]:
    from iqs_mcp.server import build_server
    from iqs_mcp.store import InMemoryTicketStore

    causes = [c.code for c in taxonomy.causes]
    ranker = CauseRanker(library, causes, params)
    store = InMemoryTicketStore()
    agent = DiagnosisAgent(
        AgentDeps(
            vocabulary=vocabulary,
            ranker=ranker,
            taxonomy=taxonomy,
            tickets=McpTicketGateway(build_server(store)),
            repository=InMemoryCaseRepository(),
        ),
        memory_checkpointer(),
    )
    library_causes = {c.cause_code for c in library}
    counts = Counter(c.cause_code for c in library)
    majority = [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]

    results: list[CaseResult] = []
    for case, label in cases:
        t0 = time.perf_counter()
        out = await agent.run(case)
        elapsed = 1000 * (time.perf_counter() - t0)
        done = await agent.confirm(case.case_id, label.cause_code, ENGINEER)
        calls = done["tool_calls"]
        correct = 0
        if calls and _check_create(calls[0], case, out, store):
            correct += 1
        if (
            len(calls) > 1
            and "ticket" in out
            and _check_update(calls[1], out["ticket"]["ticket_id"], label.cause_code, store)
        ):
            correct += 1

        profile = SymptomProfile.model_validate(out["profile"])
        depth = ranker.retriever.depth
        bm25 = ranker.retriever.bm25.search(profile.text(), depth)
        vec = ranker.retriever.vector.search(profile.features, depth)
        baselines = {
            "majority": majority,
            "bm25_only": [
                c
                for c, _ in top_k(
                    vote_probabilities(bm25, ranker.retriever.cause_of, causes, ranker.votes), 3
                )
            ],
            "vector_only": [
                c
                for c, _ in top_k(
                    vote_probabilities(vec, ranker.retriever.cause_of, causes, ranker.votes), 3
                )
            ],
        }
        results.append(
            CaseResult(
                case_id=case.case_id,
                truth=label.cause_code,
                simulator_fault=label.simulator_fault,
                secondary=list(label.secondary_causes),
                known=label.cause_code in library_causes,
                top3=[c for c, _ in out["top3"]],
                novel=bool(out["novel"]),
                max_similarity=float(out["max_similarity"]),
                completeness=float(out["completeness"]["score"]),
                tool_calls_correct=correct,
                tool_calls_total=2,
                latency_ms=round(elapsed, 3),
                baselines=baselines,
            )
        )
    return results


class PairSummary(BaseModel):
    """Two-fault cases whose primary cause is in the library."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    top1_primary: Proportion
    top3_primary: Proportion
    top3_all_injected: Proportion  # every injected cause in the top 3
    top1_any_injected: Proportion  # top 1 is the primary or a secondary cause


class Summary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_cases: int
    n_known: int
    n_novel: int
    top1_known: Proportion
    top3_known: Proportion
    top3_all: Proportion
    per_fault_top1: dict[str, Proportion]
    false_novelty: Proportion
    novelty_detection: Proportion
    baselines_known: dict[str, dict[str, Proportion]]
    random_expected: dict[str, float]
    single_top1_known: Proportion
    pairs: PairSummary
    tool_call_accuracy: Proportion
    completeness_mean: float
    completeness_full: Proportion
    latency: Latency
    llm_cost_usd: float


def summarize(results: Sequence[CaseResult], n_taxonomy_causes: int) -> Summary:
    known = [r for r in results if r.known]
    novel = [r for r in results if not r.known]

    def hits(rs: Sequence[CaseResult], k: int, key: str | None = None) -> int:
        return sum(r.truth in (r.baselines[key] if key else r.top3)[:k] for r in rs)

    faults = sorted({r.simulator_fault for r in results})
    per_fault = {
        f: wilson(
            hits([r for r in results if r.simulator_fault == f], 1),
            sum(r.simulator_fault == f for r in results),
        )
        for f in faults
    }
    baselines = {
        b: {
            "top1": wilson(hits(known, 1, b), len(known)),
            "top3": wilson(hits(known, 3, b), len(known)),
        }
        for b in ("majority", "bm25_only", "vector_only")
    }
    singles = [r for r in known if not r.secondary]
    pairs = [r for r in known if r.secondary]
    pair_summary = PairSummary(
        n=len(pairs),
        top1_primary=wilson(hits(pairs, 1), len(pairs)),
        top3_primary=wilson(hits(pairs, 3), len(pairs)),
        top3_all_injected=wilson(
            sum({r.truth, *r.secondary} <= set(r.top3) for r in pairs), len(pairs)
        ),
        top1_any_injected=wilson(
            sum(r.top3[0] in {r.truth, *r.secondary} for r in pairs), len(pairs)
        ),
    )
    return Summary(
        n_cases=len(results),
        n_known=len(known),
        n_novel=len(novel),
        top1_known=wilson(hits(known, 1), len(known)),
        top3_known=wilson(hits(known, 3), len(known)),
        top3_all=wilson(hits(results, 3), len(results)),
        per_fault_top1=per_fault,
        false_novelty=wilson(sum(r.novel for r in known), len(known)),
        novelty_detection=wilson(sum(r.novel for r in novel), len(novel)),
        baselines_known=baselines,
        random_expected={"top1": 1 / n_taxonomy_causes, "top3": 3 / n_taxonomy_causes},
        single_top1_known=wilson(hits(singles, 1), len(singles)),
        pairs=pair_summary,
        tool_call_accuracy=wilson(
            sum(r.tool_calls_correct for r in results), sum(r.tool_calls_total for r in results)
        ),
        completeness_mean=sum(r.completeness for r in results) / len(results),
        completeness_full=wilson(sum(r.completeness == 1.0 for r in results), len(results)),
        latency=latency([r.latency_ms for r in results]),
        llm_cost_usd=0.0,
    )
