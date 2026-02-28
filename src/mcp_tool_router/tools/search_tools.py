"""MCP Tool 1 – semantic / hybrid search over the tool registry.

Returns a ranked list of tools whose descriptions, parameters, or
synthetic questions best match the agent's natural-language query.
"""

from __future__ import annotations

from typing import Any

from fastmcp import Context

from mcp_tool_router.models.schemas import SearchResult


async def search_tools(
    ctx: Context,
    query: str,
    tags: list[str] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search for tools relevant to your query using semantic and keyword matching.

    Args:
        query: Natural-language description of the capability you need.
        tags: Optional tag filter (tools must have at least one matching tag).
        top_k: Maximum number of results to return.

    Returns:
        Ranked list of matching tools with name, description, inputSchema, and score.
    """
    app = ctx.request_context.lifespan_context  # type: ignore[union-attr]

    query_embedding = await app["embeddings"].embed_query(query)

    results: list[SearchResult] = await app["index"].hybrid_search(
        query_text=query,
        query_embedding=query_embedding,
        top_k=top_k,
        alpha=app["settings"].search.hybrid_alpha,
        tags=tags,
        min_score=app["settings"].search.min_score,
    )

    return [r.model_dump() for r in results]
