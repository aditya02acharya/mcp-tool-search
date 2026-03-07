"""Tests for the background sync worker."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from mcp.types import Tool

from mcp_tool_router.embeddings.client import EmbeddingClient
from mcp_tool_router.index.store import ToolIndex
from mcp_tool_router.mcp_client.factory import MCPClientFactory
from mcp_tool_router.settings import (
    IndexSettings,
    LLMSettings,
    RegistrySettings,
    TDWASettings,
)
from mcp_tool_router.sync.worker import SyncWorker
from tests.conftest import DIM


def _make_mcp_tool(name: str, desc: str = "d") -> Tool:
    return Tool(name=name, description=desc, inputSchema={"type": "object"})


@pytest.fixture
def mcp_client() -> AsyncMock:
    client = AsyncMock(spec=MCPClientFactory)
    client.list_all_tools.return_value = {}
    return client


@pytest.fixture
async def index(tmp_path: object) -> ToolIndex:  # type: ignore[override]
    settings = IndexSettings(
        db_path=str(tmp_path) + "/sync_test.db",  # type: ignore[operator]
        use_vec_extension=False,
        similarity_chunk_size=100,
    )
    idx = ToolIndex(settings)
    await idx.initialize()
    return idx


@pytest.fixture
def embeddings() -> AsyncMock:
    client = AsyncMock(spec=EmbeddingClient)
    client.embed_tool_tdwa.return_value = (
        np.random.default_rng(0).standard_normal(DIM).astype(np.float32)
    )
    return client


@pytest.fixture
def worker(mcp_client: AsyncMock, index: ToolIndex, embeddings: AsyncMock) -> SyncWorker:
    return SyncWorker(
        mcp_client=mcp_client,
        index=index,
        embeddings=embeddings,
        registry_settings=RegistrySettings(),
        tdwa_settings=TDWASettings(num_synthetic_questions=0),
        llm_settings=LLMSettings(),
    )


class TestSyncOnce:
    async def test_no_tools_no_crash(self, worker: SyncWorker) -> None:
        await worker.sync_once()

    async def test_adds_new_tools(
        self, worker: SyncWorker, mcp_client: AsyncMock, index: ToolIndex
    ) -> None:
        mcp_client.list_all_tools.return_value = {
            "server-1": [_make_mcp_tool("a"), _make_mcp_tool("b")],
        }
        await worker.sync_once()
        assert await index.tool_count() == 2

    async def test_removes_deleted_tools(
        self, worker: SyncWorker, mcp_client: AsyncMock, index: ToolIndex
    ) -> None:
        mcp_client.list_all_tools.return_value = {
            "server-1": [_make_mcp_tool("a"), _make_mcp_tool("b")],
        }
        await worker.sync_once()

        mcp_client.list_all_tools.return_value = {
            "server-1": [_make_mcp_tool("a")],
        }
        # Reset global hash so diff runs
        worker._last_global_hash = None
        await worker.sync_once()
        assert await index.tool_count() == 1

    async def test_updates_modified_tools(
        self, worker: SyncWorker, mcp_client: AsyncMock, index: ToolIndex
    ) -> None:
        mcp_client.list_all_tools.return_value = {
            "server-1": [_make_mcp_tool("a", "original")],
        }
        await worker.sync_once()

        mcp_client.list_all_tools.return_value = {
            "server-1": [_make_mcp_tool("a", "changed")],
        }
        worker._last_global_hash = None
        await worker.sync_once()
        t = await index.get_tool("a")
        assert t is not None
        assert t.description == "changed"

    async def test_skips_when_hash_matches(
        self, worker: SyncWorker, mcp_client: AsyncMock, index: ToolIndex
    ) -> None:
        mcp_client.list_all_tools.return_value = {
            "server-1": [_make_mcp_tool("a")],
        }
        await worker.sync_once()

        # Second call with same tools – should skip
        await worker.sync_once()
        # embed_tool_tdwa called only once (first sync)
        assert worker._embeddings.embed_tool_tdwa.call_count == 1  # type: ignore[union-attr]

    async def test_stores_server_id(
        self, worker: SyncWorker, mcp_client: AsyncMock, index: ToolIndex
    ) -> None:
        mcp_client.list_all_tools.return_value = {
            "server-1": [_make_mcp_tool("a")],
        }
        await worker.sync_once()
        t = await index.get_tool("a")
        assert t is not None
        assert t.server_id == "server-1"

    async def test_tools_from_multiple_servers(
        self, worker: SyncWorker, mcp_client: AsyncMock, index: ToolIndex
    ) -> None:
        mcp_client.list_all_tools.return_value = {
            "server-1": [_make_mcp_tool("a")],
            "server-2": [_make_mcp_tool("b")],
        }
        await worker.sync_once()
        assert await index.tool_count() == 2
        t_a = await index.get_tool("a")
        t_b = await index.get_tool("b")
        assert t_a is not None and t_a.server_id == "server-1"
        assert t_b is not None and t_b.server_id == "server-2"


class TestSyntheticQuestions:
    async def test_generates_when_enabled(
        self, mcp_client: AsyncMock, index: ToolIndex, embeddings: AsyncMock
    ) -> None:
        worker = SyncWorker(
            mcp_client=mcp_client,
            index=index,
            embeddings=embeddings,
            registry_settings=RegistrySettings(),
            tdwa_settings=TDWASettings(num_synthetic_questions=3),
            llm_settings=LLMSettings(),
        )
        mcp_client.list_all_tools.return_value = {
            "server-1": [_make_mcp_tool("a")],
        }

        llm_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=json.dumps(["q1", "q2", "q3"])))
            ]
        )
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=llm_resp):
            await worker.sync_once()

        t = await index.get_tool("a")
        assert t is not None
        assert len(t.synthetic_questions) == 3

    async def test_handles_llm_failure(
        self, mcp_client: AsyncMock, index: ToolIndex, embeddings: AsyncMock
    ) -> None:
        worker = SyncWorker(
            mcp_client=mcp_client,
            index=index,
            embeddings=embeddings,
            registry_settings=RegistrySettings(),
            tdwa_settings=TDWASettings(num_synthetic_questions=3),
            llm_settings=LLMSettings(),
        )
        mcp_client.list_all_tools.return_value = {
            "server-1": [_make_mcp_tool("a")],
        }

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
            await worker.sync_once()

        t = await index.get_tool("a")
        assert t is not None
        assert t.synthetic_questions == []
