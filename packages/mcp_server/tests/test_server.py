"""Mock QMS MCP server: tools over an in-process MCP client, in-memory store."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from mcp import Client

from iqs_mcp.models import TicketStatus, TicketUpdate
from iqs_mcp.server import build_server
from iqs_mcp.store import InMemoryTicketStore, TicketNotFoundError

ARGS: dict[str, Any] = {
    "case_id": "SIM-abc",
    "title": "Reject rate increase on line L1",
    "severity": 2,
    "summary": "candidates ...",
    "candidate_causes": ["MACHINE.TOOL_WEAR"],
    "novel": False,
}


async def _session(
    store: InMemoryTicketStore, calls: list[tuple[str, dict[str, Any]]]
) -> list[Any]:
    out = []
    async with Client(build_server(store)) as client:
        for name, args in calls:
            out.append(await client.call_tool(name, args))
    return out


def test_tools_are_listed() -> None:
    async def names() -> set[str]:
        async with Client(build_server(InMemoryTicketStore())) as client:
            return {t.name for t in (await client.list_tools()).tools}

    assert asyncio.run(names()) >= {"create_ticket", "get_ticket", "update_ticket"}


def test_create_get_update_flow() -> None:
    store = InMemoryTicketStore()
    created, again, got, updated = asyncio.run(
        _session(
            store,
            [
                ("create_ticket", ARGS),
                ("create_ticket", ARGS),  # idempotent per open case
                ("get_ticket", {"ticket_id": "QMS-000001"}),
                (
                    "update_ticket",
                    {
                        "ticket_id": "QMS-000001",
                        "status": "cause_confirmed",
                        "confirmed_cause": "MACHINE.TOOL_WEAR",
                    },
                ),
            ],
        )
    )
    assert not created.is_error
    assert created.structured_content["ticket_id"] == "QMS-000001"
    assert again.structured_content["ticket_id"] == "QMS-000001"
    assert got.structured_content["candidate_causes"] == ["MACHINE.TOOL_WEAR"]
    assert updated.structured_content["status"] == "cause_confirmed"
    assert len(updated.structured_content["history"]) == 2


def test_tool_errors_are_reported() -> None:
    bad_sev, missing, upd_missing = asyncio.run(
        _session(
            InMemoryTicketStore(),
            [
                ("create_ticket", {**ARGS, "severity": 7}),
                ("get_ticket", {"ticket_id": "QMS-999999"}),
                ("update_ticket", {"ticket_id": "nope", "status": "closed"}),
            ],
        )
    )
    assert bad_sev.is_error and missing.is_error and upd_missing.is_error
    assert "not found" in missing.content[0].text


def test_in_memory_store_rules() -> None:
    s = InMemoryTicketStore()
    t = s.create(case_id="c", title="t", severity=1, summary="", candidate_causes=(), novel=True)
    s.update(t.ticket_id, TicketUpdate(status=TicketStatus.CLOSED))
    # A closed ticket does not block a new one for the same case.
    t2 = s.create(case_id="c", title="t", severity=1, summary="", candidate_causes=(), novel=True)
    assert t2.ticket_id != t.ticket_id
    with pytest.raises(TicketNotFoundError):
        s.get("QMS-000099")
