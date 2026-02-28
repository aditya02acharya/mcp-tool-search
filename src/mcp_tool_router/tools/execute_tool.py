"""MCP Tool 2 – proxy a tool call and accumulate the result in Redis.

Returns a short ``call_id`` the agent can reference later via
``retrieve_context``, plus an immediate LLM summary and gap analysis.
"""

from __future__ import annotations

import json
from typing import Any

from fastmcp import Context

from mcp_tool_router.models.schemas import ContextEntry


async def execute_tool(
    ctx: Context,
    session_id: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a tool on the upstream registry and accumulate the result.

    Args:
        session_id: Caller's session identifier.
        tool_name: Name of the tool to call.
        arguments: Tool arguments (JSON object).

    Returns:
        ``call_id`` for citation, a quick summary, and any information gaps.
    """
    app = ctx.request_context.lifespan_context  # type: ignore[union-attr]
    args = arguments or {}

    # Proxy the call
    result = await app["registry"].call_tool(tool_name, args)

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
