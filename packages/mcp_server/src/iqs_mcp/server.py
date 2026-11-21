"""Mock QMS/MES exposed as MCP tools (blueprint §2.6).

Tools in step 3c: ``create_ticket``, ``get_ticket``, ``update_ticket``. The remaining
tools of §2.6 (``get_batch_info``, ``query_process_data``, ``search_fmea``) arrive with
the steps that need them (5b, 6).

Run over stdio (development)::

    uv run python -m iqs_mcp            # Postgres store from IQS_DATABASE__*
    uv run python -m iqs_mcp --memory   # in-memory store (demo only)
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from iqs_mcp.models import Ticket, TicketStatus, TicketUpdate
from iqs_mcp.store import InMemoryTicketStore, TicketNotFoundError, TicketStore

SERVER_NAME = "iqs-qms"


def build_server(store: TicketStore) -> MCPServer:
    server: MCPServer = MCPServer(
        SERVER_NAME,
        instructions="Mock quality-management system: open, read and update quality tickets.",
    )

    @server.tool()
    def create_ticket(
        case_id: str,
        title: str,
        severity: int,
        summary: str = "",
        candidate_causes: list[str] | None = None,
        novel: bool = False,
    ) -> Ticket:
        """Open a quality ticket for a diagnosis case (returns the open ticket if one exists).

        severity: 1 = high, 2 = medium, 3 = low. candidate_causes: taxonomy cause codes.
        """
        try:
            return store.create(
                case_id=case_id,
                title=title,
                severity=severity,
                summary=summary,
                candidate_causes=tuple(candidate_causes or ()),
                novel=novel,
            )
        except ValidationError as exc:
            raise ToolError(f"invalid ticket: {exc.errors()[0]['msg']}") from exc

    @server.tool()
    def get_ticket(ticket_id: str) -> Ticket:
        """Read one ticket with its history."""
        try:
            return store.get(ticket_id)
        except TicketNotFoundError:
            raise ToolError(f"ticket {ticket_id} not found") from None

    @server.tool()
    def update_ticket(
        ticket_id: str,
        status: TicketStatus | None = None,
        confirmed_cause: str | None = None,
        assignee: str | None = None,
        summary: str | None = None,
    ) -> Ticket:
        """Change status, confirmed cause, assignee or summary of a ticket."""
        try:
            return store.update(
                ticket_id,
                TicketUpdate(
                    status=status,
                    confirmed_cause=confirmed_cause,
                    assignee=assignee,
                    summary=summary,
                ),
            )
        except TicketNotFoundError:
            raise ToolError(f"ticket {ticket_id} not found") from None

    return server


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="iqs_mcp", description="Mock QMS MCP server (stdio)")
    ap.add_argument("--memory", action="store_true", help="in-memory store (demo only)")
    args = ap.parse_args(argv)
    if args.memory:
        store: TicketStore = InMemoryTicketStore()
    else:
        from iqs_common.db import create_db_engine, create_session_factory
        from iqs_common.settings import get_settings
        from iqs_mcp.store import SqlTicketStore

        engine = create_db_engine(get_settings().database)
        store = SqlTicketStore(create_session_factory(engine))
    build_server(store).run("stdio")
    return 0
