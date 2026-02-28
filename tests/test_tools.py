"""Integration tests for the three MCP tools (search, execute, retrieve)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from mcp_tool_router.context.compressor import ContextCompressor
from mcp_tool_router.context.redis_store import RedisContextStore
from mcp_tool_router.embeddings.client import EmbeddingClient
from mcp_tool_router.index.store import ToolIndex
from mcp_tool_router.settings import AppSettings, SearchSettings
from mcp_tool_router.tools.execute_tool import execute_tool
from mcp_tool_router.tools.retrieve_context import retrieve_context
from mcp_tool_router.tools.search_tools import search_tools
from tests.conftest import DIM, make_indexed_tool

# ---------------------------------------------------------------------------
# Fake MCP Context
# ---------------------------------------------------------------------------


class FakeContext:
    """Minimal stand-in for ``fastmcp.Context``."""

    def __init__(self, app: dict[str, Any]) -> None:
        self.request_context = SimpleNamespace(lifespan_context=app)


@pytest.fixture
async def app_ctx(
    tool_index: ToolIndex,
    redis_store: RedisContextStore,
) -> dict[str, Any]:
    embeddings = AsyncMock(spec=EmbeddingClient)
    embeddings.embed_query.return_value = (
        np.random.default_rng(1).standard_normal(DIM).astype(np.float32)
    )
    registry = AsyncMock()
    registry.call_tool.return_value = {"temp": 20, "unit": "C"}
    compressor = AsyncMock(spec=ContextCompressor)
    compressor.quick_summarise.return_value = {"summary": "It's 20C", "gaps": []}
    compressor.summarise.return_value = SimpleNamespace(
        summary="temp is 20C",
        gaps=["no forecast"],
        citations=[],
        total_entries=1,
        total_size_bytes=50,
    )

    settings = AppSettings(
        _env_file=None,  # type: ignore[call-arg]
        search=SearchSettings(top_k=5, min_score=0.0, hybrid_alpha=0.7),
    )

    # Seed index with tools
    for t in [
        make_indexed_tool("get_weather", "Get weather for a location", ["weather"]),
        make_indexed_tool("get_stock", "Get stock price", ["finance"]),
    ]:
        await tool_index.upsert_tool(t)

    return {
        "settings": settings,
        "embeddings": embeddings,
        "index": tool_index,
        "redis_store": redis_store,
        "compressor": compressor,
        "registry": registry,
    }


# ---------------------------------------------------------------------------
# search_tools
# ---------------------------------------------------------------------------


class TestSearchTools:
    async def test_returns_results(self, app_ctx: dict[str, Any]) -> None:
        ctx = FakeContext(app_ctx)
        results = await search_tools(ctx, query="weather forecast", top_k=2)  # type: ignore[arg-type]
        assert isinstance(results, list)
        assert len(results) <= 2
        assert all("name" in r for r in results)

    async def test_tag_filter(self, app_ctx: dict[str, Any]) -> None:
        ctx = FakeContext(app_ctx)
        results = await search_tools(ctx, query="data", tags=["finance"])  # type: ignore[arg-type]
        for r in results:
            assert "finance" in r.get("tags", [])


# ---------------------------------------------------------------------------
# execute_tool
# ---------------------------------------------------------------------------


class TestExecuteTool:
    async def test_returns_call_id_and_summary(self, app_ctx: dict[str, Any]) -> None:
        ctx = FakeContext(app_ctx)
        result = await execute_tool(  # type: ignore[arg-type]
            ctx,
            session_id="sess-1",
            tool_name="get_weather",
            arguments={"loc": "London"},
        )
        assert "call_id" in result
        assert len(result["call_id"]) == 12
        assert result["summary"] == "It's 20C"
        assert result["gaps"] == []

    async def test_accumulates_in_redis(
        self, app_ctx: dict[str, Any], redis_store: RedisContextStore
    ) -> None:
        ctx = FakeContext(app_ctx)
        result = await execute_tool(  # type: ignore[arg-type]
            ctx, session_id="sess-2", tool_name="get_weather"
        )
        entries = await redis_store.get_entries("sess-2")
        assert len(entries) == 1
        assert entries[0].call_id == result["call_id"]


# ---------------------------------------------------------------------------
# retrieve_context
# ---------------------------------------------------------------------------


class TestRetrieveContext:
    async def test_verbatim_mode(self, app_ctx: dict[str, Any]) -> None:
        ctx = FakeContext(app_ctx)
        # First accumulate something
        exec_result = await execute_tool(  # type: ignore[arg-type]
            ctx, session_id="sess-3", tool_name="get_weather"
        )
        cid = exec_result["call_id"]

        result = await retrieve_context(  # type: ignore[arg-type]
            ctx, session_id="sess-3", call_ids=[cid]
        )
        assert result["mode"] == "verbatim"
        assert len(result["entries"]) == 1
        assert result["entries"][0]["call_id"] == cid

    async def test_summary_mode(self, app_ctx: dict[str, Any]) -> None:
        ctx = FakeContext(app_ctx)
        await execute_tool(  # type: ignore[arg-type]
            ctx, session_id="sess-4", tool_name="get_weather"
        )
        result = await retrieve_context(  # type: ignore[arg-type]
            ctx, session_id="sess-4", query="temperature"
        )
        assert result["mode"] == "summary"
        assert "summary" in result

    async def test_metadata_mode(self, app_ctx: dict[str, Any]) -> None:
        ctx = FakeContext(app_ctx)
        await execute_tool(  # type: ignore[arg-type]
            ctx, session_id="sess-5", tool_name="get_weather"
        )
        result = await retrieve_context(ctx, session_id="sess-5")  # type: ignore[arg-type]
        assert result["mode"] == "metadata"
        assert "total_entries" in result
