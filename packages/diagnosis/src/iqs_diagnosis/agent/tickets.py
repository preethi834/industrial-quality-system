"""Port to the quality-management system: the agent opens and updates tickets via MCP.

``TicketGateway`` is the port the agent depends on. ``McpTicketGateway`` calls the
tools of the mock QMS MCP server (``iqs_mcp``): in-process (tests, evaluation), over
stdio (development) or streamable HTTP (Docker).
"""

from __future__ import annotations

from typing import Any, Protocol

from mcp import Client


class TicketToolError(RuntimeError):
    pass


class TicketGateway(Protocol):
    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one QMS tool; returns its structured result or raises TicketToolError."""
        ...


class McpTicketGateway:
    def __init__(self, target: Any) -> None:
        """``target``: an ``MCPServer`` instance, ``StdioServerParameters`` or an HTTP URL."""
        self.target = target

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with Client(self.target) as client:
            result = await client.call_tool(tool, arguments)
        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content)
            raise TicketToolError(f"{tool} failed: {text}")
        structured = result.structured_content
        if not isinstance(structured, dict):
            raise TicketToolError(f"{tool} returned no structured result")
        return structured
