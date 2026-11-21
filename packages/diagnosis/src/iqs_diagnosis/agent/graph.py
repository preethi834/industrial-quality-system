"""Diagnosis agent: a LangGraph state machine (blueprint §2.3, base version).

    intake -> extract_symptoms -> retrieve -> novelty_check -> score_causes
           -> draft_8d -> open_ticket (MCP) -> confirm (interrupt: engineer) -> close_ticket (MCP)

Not yet in the graph: ``kg_expand`` (Neo4j, 3f), the question loop (step 8a),
hypothesis tests (5b), conformal sets (4), FMEA comparison (8b), feedback update (8b).

Rules:
- State holds JSON-compatible values only, so any checkpointer (memory, Postgres) can
  store it and a case can pause for hours at ``confirm``.
- Probabilities come from retrieval votes; no LLM is involved in this version.
- A node that calls a tool runs alone, *after* any interrupt: LangGraph re-runs a
  node from its start on resume, so a tool call placed before an interrupt in the same
  node would be sent twice.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from iqs_common.taxonomy import Taxonomy
from iqs_diagnosis.agent.tickets import TicketGateway, TicketToolError
from iqs_diagnosis.conformal.calibration import ConformalPredictor
from iqs_diagnosis.domain.case import DiagnosisCase, SymptomProfile
from iqs_diagnosis.persistence.repository import CaseRepository
from iqs_diagnosis.reporting.eight_d import check_completeness, draft_8d
from iqs_diagnosis.retrieval.base import Hit
from iqs_diagnosis.scientist.runner import hypotheses_for, test_case
from iqs_diagnosis.scoring.ranker import CauseRanker
from iqs_diagnosis.scoring.votes import top_k, vote_probabilities
from iqs_diagnosis.symptoms import extract_profile
from iqs_diagnosis.symptoms.vocabulary import Vocabulary


def _merge(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {**a, **b}


class AgentState(TypedDict, total=False):
    case: dict[str, Any]
    profile: dict[str, Any]
    fused: list[dict[str, Any]]
    max_similarity: float
    novel: bool
    probabilities: dict[str, float]
    top3: list[list[Any]]
    neighbours: list[dict[str, Any]]
    cause_set: dict[str, Any] | None
    scientist: dict[str, Any] | None
    report_8d: dict[str, Any]
    completeness: dict[str, Any]
    ticket: dict[str, Any]
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    confirmation: dict[str, Any]
    timings_ms: Annotated[dict[str, float], _merge]


@dataclass(frozen=True)
class AgentDeps:
    vocabulary: Vocabulary
    ranker: CauseRanker
    taxonomy: Taxonomy
    tickets: TicketGateway
    repository: CaseRepository
    conformal: ConformalPredictor | None = None  # step 4; None = top 3 only
    scientist_q: float | None = 0.05  # step 5b; None = no hypothesis tests


def severity_for(case: DiagnosisCase) -> int:
    """1 = high (reject rate >= 15 %), 2 = medium (>= 7 %), 3 = low. A stated rule."""
    rate = next((s for s in case.reported_symptoms if s.get("code") == "reject_rate_increase"), {})
    after = float(rate.get("rate_after", 0.0))
    return 1 if after >= 0.15 else 2 if after >= 0.07 else 3


def _timed(
    name: str, fn: Callable[[AgentState], dict[str, Any]]
) -> Callable[[AgentState], dict[str, Any]]:
    def node(state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        out = fn(state)
        return {**out, "timings_ms": {name: round(1000 * (time.perf_counter() - t0), 3)}}

    return node


def _atimed(
    name: str, fn: Callable[[AgentState], Awaitable[dict[str, Any]]]
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def node(state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        out = await fn(state)
        return {**out, "timings_ms": {name: round(1000 * (time.perf_counter() - t0), 3)}}

    return node


def build_graph(deps: AgentDeps, checkpointer: BaseCheckpointSaver[Any]) -> Any:
    causes = [c.code for c in deps.taxonomy.causes]
    cause_names = {c.code: c.name_en for c in deps.taxonomy.causes}
    symptom_names = {s.code: s.name_en for s in deps.vocabulary.symptoms}
    ranker = deps.ranker

    def intake(state: AgentState) -> dict[str, Any]:
        case = DiagnosisCase.model_validate(state["case"])
        deps.repository.add(case)
        return {"case": case.model_dump(mode="json")}

    def extract_symptoms(state: AgentState) -> dict[str, Any]:
        case = DiagnosisCase.model_validate(state["case"])
        profile = extract_profile(case, deps.vocabulary)
        return {"profile": profile.model_dump(mode="json")}

    def retrieve(state: AgentState) -> dict[str, Any]:
        profile = SymptomProfile.model_validate(state["profile"])
        r = ranker.retriever.search(profile)
        return {
            "fused": [{"case_id": h.case_id, "score": h.score, "rank": h.rank} for h in r.fused],
            "max_similarity": r.max_similarity,
        }

    def novelty_check(state: AgentState) -> dict[str, Any]:
        return {"novel": ranker.novelty.is_novel(state["max_similarity"])}

    def score_causes(state: AgentState) -> dict[str, Any]:
        hits = [Hit(**h) for h in state["fused"]]
        probs = vote_probabilities(hits, ranker.retriever.cause_of, causes, ranker.votes)
        top3 = top_k(probs, 3)
        neighbours = [
            {
                "case_id": h.case_id,
                "cause_code": ranker.retriever.cause_of(h.case_id),
                "score": h.score,
            }
            for h in hits[: ranker.votes.n_neighbours]
        ]
        deps.repository.save_ranking(state["case"]["case_id"], top3, state["novel"])
        return {
            "probabilities": probs,
            "top3": [[c, p] for c, p in top3],
            "neighbours": neighbours,
        }

    templates = {c.code: c.test_template for c in deps.taxonomy.causes if c.test_template}

    testable = [c.code for c in deps.taxonomy.causes if c.test_template]

    def test_hypotheses(state: AgentState) -> dict[str, Any]:
        """Scientist: test the top-3 causes on the case's data (BH over the three); for a
        case flagged novel, test every cause (BH over all), since the ranking comes from
        past cases that do not resemble this one."""
        case = DiagnosisCase.model_validate(state["case"])
        if deps.scientist_q is None or case.observed_dir is None:
            return {"scientist": None}
        top3 = [c for c, _ in state["top3"] if c in templates]
        hyps, mode = hypotheses_for(top3, testable, novel=state["novel"])
        report = test_case(case, hyps, templates, q=deps.scientist_q, mode=mode)
        deps.repository.save_hypothesis_tests(case.case_id, report)
        return {"scientist": report.model_dump(mode="json")}

    def conformal_set(state: AgentState) -> dict[str, Any]:
        pred = deps.conformal
        if pred is None:
            return {"cause_set": None}
        causes_in = pred.cause_set(state["probabilities"])
        run = pred.run
        run_key = f"{run.run_id}:{pred.variant}"
        deps.repository.save_cause_set(state["case"]["case_id"], causes_in, run_key)
        return {
            "cause_set": {
                "causes": causes_in,
                "empty": not causes_in,
                # Shown as "uncertain / possible new problem" (dashboard decision).
                "lac_empty": pred.lac_empty(state["probabilities"]),
                "method": pred.threshold.method.value,
                "variant": pred.variant,
                "alpha": run.alpha,
                "delta": pred.threshold.delta,
                "qhat": pred.threshold.qhat,
                "calibration_run_id": run_key,
                "source": run.source,
                "guarantee": (
                    f"Contains the true cause with probability >= {1 - run.alpha:.2f}, "
                    f"marginally over {run.source} cases exchangeable with the "
                    f"{run.n_calibration} calibration cases; causes outside the taxonomy "
                    "and new problems are not covered."
                ),
            }
        }

    def draft_report(state: AgentState) -> dict[str, Any]:
        case = DiagnosisCase.model_validate(state["case"])
        report = draft_8d(
            case,
            SymptomProfile.model_validate(state["profile"]),
            top3=[(c, float(p)) for c, p in state["top3"]],
            novel=state["novel"],
            max_similarity=state["max_similarity"],
            neighbours=state["neighbours"],
            cause_names=cause_names,
            symptom_names=symptom_names,
            cause_set=state.get("cause_set"),
            scientist=state.get("scientist"),
        )
        completeness = check_completeness(report, causes)
        deps.repository.save_report(report, completeness.score)
        return {
            "report_8d": report.model_dump(mode="json"),
            "completeness": {**completeness.model_dump(mode="json"), "score": completeness.score},
        }

    async def open_ticket(state: AgentState) -> dict[str, Any]:
        case = DiagnosisCase.model_validate(state["case"])
        args = {
            "case_id": case.case_id,
            "title": case.title,
            "severity": severity_for(case),
            "summary": state["report_8d"]["sections"][4]["content"],
            "candidate_causes": [c for c, _ in state["top3"]],
            "novel": state["novel"],
        }
        return await _call(deps.tickets, "create_ticket", args, result_key="ticket")

    def confirm(state: AgentState) -> dict[str, Any]:
        answer = interrupt(
            {
                "ask": "confirm_root_cause",
                "case_id": state["case"]["case_id"],
                "top3": state["top3"],
                "novel": state["novel"],
                "ticket_id": state.get("ticket", {}).get("ticket_id"),
            }
        )
        cause = answer.get("cause_code") if isinstance(answer, dict) else None
        if cause is not None and cause not in causes:
            raise ValueError(f"confirmed cause {cause!r} is not in the taxonomy")
        engineer = str(answer.get("engineer", "unknown")) if isinstance(answer, dict) else "unknown"
        deps.repository.save_confirmation(state["case"]["case_id"], cause)
        return {"confirmation": {"cause_code": cause, "engineer": engineer}}

    async def close_ticket(state: AgentState) -> dict[str, Any]:
        ticket = state.get("ticket") or {}
        if "ticket_id" not in ticket:
            return {}
        conf = state["confirmation"]
        args: dict[str, Any] = {
            "ticket_id": ticket["ticket_id"],
            "status": "cause_confirmed",
            "assignee": conf["engineer"],
        }
        if conf["cause_code"] is not None:
            args["confirmed_cause"] = conf["cause_code"]
        return await _call(deps.tickets, "update_ticket", args, result_key="ticket")

    # Any: LangGraph's add_node overloads do not accept our wrapped (timed) callables.
    g: Any = StateGraph(AgentState)
    g.add_node("intake", _timed("intake", intake))
    g.add_node("extract_symptoms", _timed("extract_symptoms", extract_symptoms))
    g.add_node("retrieve", _timed("retrieve", retrieve))
    g.add_node("novelty_check", _timed("novelty_check", novelty_check))
    g.add_node("score_causes", _timed("score_causes", score_causes))
    g.add_node("test_hypotheses", _timed("test_hypotheses", test_hypotheses))
    g.add_node("conformal_set", _timed("conformal_set", conformal_set))
    g.add_node("draft_8d", _timed("draft_8d", draft_report))
    g.add_node("open_ticket", _atimed("open_ticket", open_ticket))
    g.add_node("confirm", confirm)
    g.add_node("close_ticket", _atimed("close_ticket", close_ticket))
    order = [
        "intake",
        "extract_symptoms",
        "retrieve",
        "novelty_check",
        "score_causes",
        "test_hypotheses",
        "conformal_set",
        "draft_8d",
        "open_ticket",
        "confirm",
        "close_ticket",
    ]
    g.add_edge(START, order[0])
    for a, b in zip(order, order[1:], strict=False):
        g.add_edge(a, b)
    g.add_edge(order[-1], END)
    return g.compile(checkpointer=checkpointer)


async def _call(
    gateway: TicketGateway, tool: str, args: dict[str, Any], *, result_key: str
) -> dict[str, Any]:
    record: dict[str, Any] = {"tool": tool, "arguments": args}
    try:
        result = await gateway.call(tool, args)
    except TicketToolError as exc:
        return {"tool_calls": [{**record, "ok": False, "error": str(exc)}]}
    return {result_key: result, "tool_calls": [{**record, "ok": True}]}


Phase = Literal["awaiting_confirmation", "done", "not_started"]


class DiagnosisAgent:
    """Thin runner: one LangGraph thread per case (thread_id = case_id)."""

    def __init__(self, deps: AgentDeps, checkpointer: BaseCheckpointSaver[Any]) -> None:
        self.graph = build_graph(deps, checkpointer)
        self.has_conformal = deps.conformal is not None
        self.has_scientist = deps.scientist_q is not None

    @staticmethod
    def _config(case_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": case_id}}

    async def run(self, case: DiagnosisCase) -> dict[str, Any]:
        """Run until the engineer's confirmation is needed (or to the end)."""
        out: dict[str, Any] = await self.graph.ainvoke(
            {"case": case.model_dump(mode="json")}, self._config(case.case_id)
        )
        return out

    async def confirm(self, case_id: str, cause_code: str | None, engineer: str) -> dict[str, Any]:
        if self.phase(case_id) != "awaiting_confirmation":
            raise RuntimeError(f"case {case_id} is not waiting for confirmation")
        out: dict[str, Any] = await self.graph.ainvoke(
            Command(resume={"cause_code": cause_code, "engineer": engineer}),
            self._config(case_id),
        )
        return out

    def state(self, case_id: str) -> dict[str, Any]:
        values: dict[str, Any] = self.graph.get_state(self._config(case_id)).values
        return values

    def phase(self, case_id: str) -> Phase:
        snap = self.graph.get_state(self._config(case_id))
        if not snap.values:
            return "not_started"
        return "awaiting_confirmation" if "confirm" in snap.next else "done"
