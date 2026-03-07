"""MCP Tool 2 – proxy a tool call and accumulate the result in Redis.

Returns a short ``call_id`` the agent can reference later via
``retrieve_context``, plus an immediate LLM summary and gap analysis.

Routes tool calls through the ``MCPClientFactory`` to the remote MCP server.
"""

from __future__ import annotations

import json
from typing import Any

from fastmcp import Context

from mcp_tool_router.mcp_client.factory import MCPClientError
from mcp_tool_router.models.schemas import ContextEntry


async def execute_tool(
    ctx: Context,
    session_id: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    server_id: str = "",
) -> dict[str, Any]:
    """Execute a tool on a remote MCP server and accumulate the result.

    Args:
        session_id: Caller's session identifier.
        tool_name: Name of the tool to call.
        arguments: Tool arguments (JSON object).
        server_id: Target MCP server identifier.

    Returns:
        ``call_id`` for citation, a quick summary, and any information gaps.
    """
    app = ctx.request_context.lifespan_context  # type: ignore[union-attr]
    args = arguments or {}

    if not server_id:
        return {
            "error": "server_id is required to route the tool call",
            "tool_name": tool_name,
        }

    try:
        call_result = await app["mcp_client_factory"].call_tool(server_id, tool_name, args)
        # Extract text content from CallToolResult
        result: Any = [
            getattr(c, "text", str(c)) for c in call_result.content
        ]
        if len(result) == 1:
            result = result[0]
    except MCPClientError as exc:
        return {
            "error": str(exc),
            "tool_name": tool_name,
            "server_id": server_id,
        }

    # Build a snippet for lightweight metadata queries
    snippet = json.dumps(result, default=str)[:200]

    entry = ContextEntry(
        session_id=session_id,
        tool_name=tool_name,
        arguments=args,
        result=result,
        snippet=snippet,
    )

    # Persist in Redis
    call_id = await app["redis_store"].accumulate(entry)

    # Quick LLM summary + gap analysis
    summary_data = await app["compressor"].quick_summarise(entry)

    return {
        "call_id": call_id,
        "tool_name": tool_name,
        "summary": summary_data.get("summary", ""),
        "gaps": summary_data.get("gaps", []),
        "content_size_bytes": entry.content_size_bytes,
    }
