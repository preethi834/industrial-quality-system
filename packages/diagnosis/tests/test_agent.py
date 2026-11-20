"""Agent graph end to end over reference scenarios (in-process MCP, memory checkpointer)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from iqs_diagnosis.agent.graph import severity_for
from iqs_diagnosis.agent.tickets import TicketToolError
from iqs_diagnosis.ingestion.simulator import load_case
from iqs_diagnosis.persistence.repository import CaseStatus

from .agent_fixtures import make_agent

Profiles = dict[str, tuple[Any, Any]]


def _case(reference_runs: dict[str, Path], name: str) -> Any:
    case = load_case(reference_runs[name])
    assert case is not None
    return case


def test_run_pauses_for_confirmation_then_closes_ticket(
    reference_runs: dict[str, Path], reference_profiles: Profiles
) -> None:
    agent, repo, store = make_agent(reference_profiles)
    case = _case(reference_runs, "fixture_offset_m2b")
    out = asyncio.run(agent.run(case))

    assert agent.phase(case.case_id) == "awaiting_confirmation"
    assert out["top3"][0][0] == "MACHINE.FIXTURE_MISALIGNMENT" and not out["novel"]
    assert sum(out["probabilities"].values()) == pytest.approx(1.0)
    assert out["completeness"]["score"] == 1.0
    assert repo.status(case.case_id) is CaseStatus.AWAITING_CONFIRMATION
    (call,) = out["tool_calls"]
    assert call["tool"] == "create_ticket" and call["ok"]
    assert call["arguments"]["candidate_causes"] == [c for c, _ in out["top3"]]
    ticket = store.get(out["ticket"]["ticket_id"])
    assert ticket.case_id == case.case_id and ticket.status == "open"
    assert set(out["timings_ms"]) >= {"extract_symptoms", "retrieve", "open_ticket"}

    done = asyncio.run(agent.confirm(case.case_id, "MACHINE.FIXTURE_MISALIGNMENT", "eng-1"))
    assert agent.phase(case.case_id) == "done"
    assert [c["tool"] for c in done["tool_calls"]] == ["create_ticket", "update_ticket"]
    ticket = store.get(out["ticket"]["ticket_id"])
    assert ticket.status == "cause_confirmed"
    assert ticket.confirmed_cause == "MACHINE.FIXTURE_MISALIGNMENT"
    assert ticket.assignee == "eng-1" and len(ticket.history) == 2
    assert repo.confirmations[case.case_id] == "MACHINE.FIXTURE_MISALIGNMENT"
    with pytest.raises(RuntimeError, match="not waiting"):
        asyncio.run(agent.confirm(case.case_id, None, "eng-1"))


def test_unseen_cause_is_flagged_novel(
    reference_runs: dict[str, Path], reference_profiles: Profiles
) -> None:
    agent, _, _ = make_agent(reference_profiles)
    out = asyncio.run(agent.run(_case(reference_runs, "drive_fault_m1")))
    assert out["novel"] is True
    assert "No close past case" in out["report_8d"]["sections"][4]["content"]
    assert out["tool_calls"][0]["arguments"]["novel"] is True


def test_confirm_with_no_listed_cause(
    reference_runs: dict[str, Path], reference_profiles: Profiles
) -> None:
    agent, repo, store = make_agent(reference_profiles)
    case = _case(reference_runs, "drive_fault_m1")
    out = asyncio.run(agent.run(case))
    asyncio.run(agent.confirm(case.case_id, None, "eng-2"))
    t = store.get(out["ticket"]["ticket_id"])
    assert t.status == "cause_confirmed" and t.confirmed_cause is None
    assert repo.confirmations[case.case_id] is None


def test_unknown_confirmed_cause_is_rejected(
    reference_runs: dict[str, Path], reference_profiles: Profiles
) -> None:
    agent, _, _ = make_agent(reference_profiles)
    case = _case(reference_runs, "tool_wear_m1")
    asyncio.run(agent.run(case))
    with pytest.raises(ValueError, match="not in the taxonomy"):
        asyncio.run(agent.confirm(case.case_id, "MACHINE.NOT_A_CAUSE", "eng"))


class _FailingGateway:
    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise TicketToolError("qms down")


def test_tool_failure_is_recorded_not_fatal(
    reference_runs: dict[str, Path], reference_profiles: Profiles
) -> None:
    agent, _, _ = make_agent(reference_profiles, gateway=_FailingGateway())
    case = _case(reference_runs, "coolant_loss_m3")
    out = asyncio.run(agent.run(case))
    assert out["tool_calls"] == [{**out["tool_calls"][0], "ok": False, "error": "qms down"}]
    assert "ticket" not in out
    assert agent.phase(case.case_id) == "awaiting_confirmation"
    done = asyncio.run(agent.confirm(case.case_id, "MACHINE.INSUFFICIENT_COOLING", "eng"))
    assert len(done["tool_calls"]) == 1  # no ticket, so no update call


def test_rerun_does_not_open_a_second_ticket(
    reference_runs: dict[str, Path], reference_profiles: Profiles
) -> None:
    from iqs_diagnosis.agent.tickets import McpTicketGateway
    from iqs_mcp.server import build_server
    from iqs_mcp.store import InMemoryTicketStore

    store = InMemoryTicketStore()
    gateway = McpTicketGateway(build_server(store))
    case = _case(reference_runs, "bad_batch_b0008")
    ids = set()
    for _ in range(2):  # two agents (e.g. after a restart without checkpoints)
        agent, _, _ = make_agent(reference_profiles, gateway=gateway)
        ids.add(asyncio.run(agent.run(case))["ticket"]["ticket_id"])
    assert len(ids) == 1


def test_severity_rule(reference_runs: dict[str, Path]) -> None:
    case = _case(reference_runs, "gauge_drift_q1")
    assert severity_for(case) in (1, 2, 3)
    rates = [
        severity_for(_case(reference_runs, n)) for n in ("ambient_excursion", "drive_fault_m1")
    ]
    assert all(r in (1, 2, 3) for r in rates)
