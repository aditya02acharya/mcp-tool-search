"""MCP Tool 3 – retrieve accumulated session context on demand.

Supports two modes:
* **Verbatim** – pass specific ``call_ids`` to get raw results + metadata.
* **Smart select** – pass a *query*; the compressor selects, summarises,
  and returns citations the agent can reference for follow-up.
"""

from __future__ import annotations

from typing import Any

from fastmcp import Context

from mcp_tool_router.models.schemas import ContextEntry


async def retrieve_context(
    ctx: Context,
    session_id: str,
    query: str | None = None,
    call_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Retrieve accumulated tool-call context for this session.

    Args:
        session_id: Caller's session identifier.
        query: Natural-language query for smart selection and summarisation.
        call_ids: Specific call_ids to retrieve verbatim results for.

    Returns:
        When *call_ids* given: list of verbatim entries with metadata.
        When *query* given: LLM summary with citations and gap analysis.
        Otherwise: session metadata overview.
    """
    app = ctx.request_context.lifespan_context  # type: ignore[union-attr]

    # Verbatim retrieval by call_id
    if call_ids:
        entries: list[ContextEntry] = await app["redis_store"].get_entries(
            session_id, call_ids=call_ids
        )
        return {
            "mode": "verbatim",
            "entries": [
                {
                    "call_id": e.call_id,
                    "tool_name": e.tool_name,
                    "arguments": e.arguments,
                    "result": e.result,
                    "timestamp": e.timestamp.isoformat(),
                    "content_size_bytes": e.content_size_bytes,
                }
                for e in entries
            ],
        }

    # Smart selection + summarisation
    if query:
        all_entries = await app["redis_store"].get_entries(session_id)
        summary = await app["compressor"].summarise(all_entries, query)
        return {
            "mode": "summary",
            "summary": summary.summary,
            "gaps": summary.gaps,
            "citations": [c.model_dump() for c in summary.citations],
            "total_entries": summary.total_entries,
            "total_size_bytes": summary.total_size_bytes,
        }

    # Metadata only
    info = await app["redis_store"].get_session_info(session_id)
    metadata = await app["redis_store"].get_entry_metadata(session_id)
    return {
        "mode": "metadata",
        **info,
        "entries": metadata,
    }
