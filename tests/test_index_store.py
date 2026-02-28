"""Tests for the SQLite vector + FTS index store (Python cosine fallback)."""

from __future__ import annotations

import numpy as np

from mcp_tool_router.index.store import (
    ToolIndex,
    _deserialize_f32,
    _rrf_combine,
    _sanitise_fts,
    _serialize_f32,
)
from mcp_tool_router.models.schemas import IndexedTool
from tests.conftest import DIM, make_indexed_tool

# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_roundtrip(self) -> None:
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert np.allclose(_deserialize_f32(_serialize_f32(vec)), vec)


# ---------------------------------------------------------------------------
# ToolIndex – lifecycle & upsert
# ---------------------------------------------------------------------------


class TestToolIndexLifecycle:
    async def test_initialize_creates_tables(self, tool_index: ToolIndex) -> None:
        count = await tool_index.tool_count()
        assert count == 0

    async def test_upsert_and_count(
        self, tool_index: ToolIndex, sample_indexed_tool: IndexedTool
    ) -> None:
        await tool_index.upsert_tool(sample_indexed_tool)
        assert await tool_index.tool_count() == 1

    async def test_upsert_updates_existing(
        self, tool_index: ToolIndex, sample_indexed_tool: IndexedTool
    ) -> None:
        await tool_index.upsert_tool(sample_indexed_tool)
        updated = sample_indexed_tool.model_copy(update={"description": "new desc"})
        await tool_index.upsert_tool(updated)
        assert await tool_index.tool_count() == 1
        t = await tool_index.get_tool("get_weather")
        assert t is not None
        assert t.description == "new desc"


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    async def test_delete_existing(
        self, tool_index: ToolIndex, sample_indexed_tool: IndexedTool
    ) -> None:
        await tool_index.upsert_tool(sample_indexed_tool)
        await tool_index.delete_tools(["get_weather"])
        assert await tool_index.tool_count() == 0

    async def test_delete_nonexistent(self, tool_index: ToolIndex) -> None:
        await tool_index.delete_tools(["nope"])  # should not raise


# ---------------------------------------------------------------------------
# Content hashes
# ---------------------------------------------------------------------------


class TestContentHashes:
    async def test_returns_hashes(
        self, tool_index: ToolIndex, sample_indexed_tool: IndexedTool
    ) -> None:
        await tool_index.upsert_tool(sample_indexed_tool)
        hashes = await tool_index.get_content_hashes()
        assert "get_weather" in hashes


# ---------------------------------------------------------------------------
# Vector search (Python fallback)
# ---------------------------------------------------------------------------


class TestPythonCosineSearch:
    async def test_returns_ranked(self, tool_index: ToolIndex) -> None:
        for t in [
            make_indexed_tool("weather", "get weather info", ["weather"]),
            make_indexed_tool("stock", "get stock prices", ["finance"]),
            make_indexed_tool("translate", "translate text", ["nlp"]),
        ]:
            await tool_index.upsert_tool(t)

        query = np.random.default_rng(99).standard_normal(DIM).astype(np.float32)
        results = await tool_index.hybrid_search(
            query_text="weather forecast",
            query_embedding=query,
            top_k=2,
            alpha=1.0,  # vector only
        )
        assert len(results) <= 2
        assert all(r.score >= 0 for r in results)

    async def test_empty_index(self, tool_index: ToolIndex) -> None:
        query = np.zeros(DIM, dtype=np.float32)
        results = await tool_index.hybrid_search(
            query_text="anything", query_embedding=query, top_k=5
        )
        assert results == []


# ---------------------------------------------------------------------------
# FTS search
# ---------------------------------------------------------------------------


class TestFTSSearch:
    async def test_keyword_match(self, tool_index: ToolIndex) -> None:
        await tool_index.upsert_tool(
            make_indexed_tool("get_weather", "Get current weather conditions")
        )
        query = np.zeros(DIM, dtype=np.float32)
        results = await tool_index.hybrid_search(
            query_text="weather",
            query_embedding=query,
            top_k=5,
            alpha=0.0,  # FTS only
        )
        assert len(results) >= 1
        assert results[0].name == "get_weather"


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


class TestHybridSearch:
    async def test_tag_filter(self, tool_index: ToolIndex) -> None:
        await tool_index.upsert_tool(make_indexed_tool("a", "tool a", ["alpha"]))
        await tool_index.upsert_tool(make_indexed_tool("b", "tool b", ["beta"]))
        query = np.random.default_rng(1).standard_normal(DIM).astype(np.float32)
        results = await tool_index.hybrid_search(
            query_text="tool",
            query_embedding=query,
            top_k=10,
            tags=["alpha"],
        )
        names = {r.name for r in results}
        assert "b" not in names

    async def test_min_score_filter(self, tool_index: ToolIndex) -> None:
        await tool_index.upsert_tool(make_indexed_tool("a", "tool a"))
        query = np.zeros(DIM, dtype=np.float32)
        results = await tool_index.hybrid_search(
            query_text="zzzzz",
            query_embedding=query,
            top_k=5,
            min_score=999.0,
        )
        assert results == []


# ---------------------------------------------------------------------------
# RRF + FTS sanitisation
# ---------------------------------------------------------------------------


class TestRRF:
    def test_combines_scores(self) -> None:
        vec = [("a", 0.9), ("b", 0.5)]
        fts = [("b", 10.0), ("c", 5.0)]
        combined = _rrf_combine(vec, fts, alpha=0.5)
        names = [n for n, _ in combined]
        # 'b' appears in both → should rank high
        assert "b" in names

    def test_empty_inputs(self) -> None:
        assert _rrf_combine([], []) == []


class TestSanitiseFTS:
    def test_strips_special(self) -> None:
        assert _sanitise_fts('"hello" (world)*') == "hello world"

    def test_empty(self) -> None:
        assert _sanitise_fts("") == ""
