"""Tests for the Redis context store (using fakeredis)."""

from __future__ import annotations

import pytest

from mcp_tool_router.context.redis_store import ContextOverflowError, RedisContextStore
from mcp_tool_router.models.schemas import ContextEntry


class TestAccumulate:
    async def test_stores_and_returns_call_id(
        self, redis_store: RedisContextStore, sample_context_entry: ContextEntry
    ) -> None:
        call_id = await redis_store.accumulate(sample_context_entry)
        assert call_id == sample_context_entry.call_id

    async def test_overflow_raises(self, redis_store: RedisContextStore) -> None:
        # max_context_bytes=1024 in fixture
        big = ContextEntry(
            session_id="s1",
            tool_name="t",
            result={"data": "x" * 2000},
        )
        with pytest.raises(ContextOverflowError):
            await redis_store.accumulate(big)

    async def test_multiple_entries(self, redis_store: RedisContextStore) -> None:
        for i in range(3):
            entry = ContextEntry(session_id="s1", tool_name=f"tool_{i}", result={"i": i})
            await redis_store.accumulate(entry)

        entries = await redis_store.get_entries("s1")
        assert len(entries) == 3


class TestGetEntries:
    async def test_filter_by_call_ids(self, redis_store: RedisContextStore) -> None:
        ids = []
        for i in range(3):
            e = ContextEntry(session_id="s1", tool_name=f"t{i}", result=i)
            cid = await redis_store.accumulate(e)
            ids.append(cid)

        subset = await redis_store.get_entries("s1", call_ids=[ids[0], ids[2]])
        assert len(subset) == 2
        assert {e.call_id for e in subset} == {ids[0], ids[2]}

    async def test_empty_session(self, redis_store: RedisContextStore) -> None:
        entries = await redis_store.get_entries("nonexistent")
        assert entries == []

    async def test_sorted_by_timestamp(self, redis_store: RedisContextStore) -> None:
        for i in range(5):
            await redis_store.accumulate(ContextEntry(session_id="s1", tool_name=f"t{i}", result=i))
        entries = await redis_store.get_entries("s1")
        timestamps = [e.timestamp for e in entries]
        assert timestamps == sorted(timestamps)


class TestSessionInfo:
    async def test_returns_counts(self, redis_store: RedisContextStore) -> None:
        await redis_store.accumulate(ContextEntry(session_id="s1", tool_name="t", result="x"))
        info = await redis_store.get_session_info("s1")
        assert info["total_entries"] == 1
        assert info["total_size_bytes"] > 0

    async def test_empty_session_info(self, redis_store: RedisContextStore) -> None:
        info = await redis_store.get_session_info("empty")
        assert info["total_entries"] == 0


class TestEntryMetadata:
    async def test_returns_lightweight_data(self, redis_store: RedisContextStore) -> None:
        await redis_store.accumulate(
            ContextEntry(
                session_id="s1",
                tool_name="weather",
                result={"temp": 20},
                snippet="temp: 20",
            )
        )
        meta = await redis_store.get_entry_metadata("s1")
        assert len(meta) == 1
        assert meta[0]["tool_name"] == "weather"
        assert "snippet" in meta[0]
