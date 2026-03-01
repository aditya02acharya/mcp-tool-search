"""Tests for the background sync worker."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from mcp_tool_router.embeddings.client import EmbeddingClient
from mcp_tool_router.index.store import ToolIndex
from mcp_tool_router.models.schemas import ToolRecord
from mcp_tool_router.registry.client import RegistryClient
from mcp_tool_router.settings import (
    IndexSettings,
    LLMSettings,
    RegistrySettings,
    TDWASettings,
)
from mcp_tool_router.sync.worker import SyncWorker
from tests.conftest import DIM


def _make_tool(name: str, desc: str = "d") -> ToolRecord:
    return ToolRecord(name=name, description=desc, input_schema={})


@pytest.fixture
def registry() -> AsyncMock:
    r = AsyncMock(spec=RegistryClient)
    r.list_tools.return_value = []
    return r


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
def worker(registry: AsyncMock, index: ToolIndex, embeddings: AsyncMock) -> SyncWorker:
    return SyncWorker(
        registry=registry,
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
        self, worker: SyncWorker, registry: AsyncMock, index: ToolIndex
    ) -> None:
        registry.list_tools.return_value = [_make_tool("a"), _make_tool("b")]
        await worker.sync_once()
        assert await index.tool_count() == 2

    async def test_removes_deleted_tools(
        self, worker: SyncWorker, registry: AsyncMock, index: ToolIndex
    ) -> None:
        registry.list_tools.return_value = [_make_tool("a"), _make_tool("b")]
        await worker.sync_once()

        registry.list_tools.return_value = [_make_tool("a")]
        # Reset global hash so diff runs
        worker._last_global_hash = None
        await worker.sync_once()
        assert await index.tool_count() == 1

    async def test_updates_modified_tools(
        self, worker: SyncWorker, registry: AsyncMock, index: ToolIndex
    ) -> None:
        registry.list_tools.return_value = [_make_tool("a", "original")]
        await worker.sync_once()

        registry.list_tools.return_value = [_make_tool("a", "changed")]
        worker._last_global_hash = None
        await worker.sync_once()
        t = await index.get_tool("a")
        assert t is not None
        assert t.description == "changed"

    async def test_skips_when_hash_matches(
        self, worker: SyncWorker, registry: AsyncMock, index: ToolIndex
    ) -> None:
        tools = [_make_tool("a")]
        registry.list_tools.return_value = tools
        await worker.sync_once()

        # Second call with same tools – should skip
        await worker.sync_once()
        # embed_tool_tdwa called only once (first sync)
        assert worker._embeddings.embed_tool_tdwa.call_count == 1  # type: ignore[union-attr]


class TestSyntheticQuestions:
    async def test_generates_when_enabled(
        self, registry: AsyncMock, index: ToolIndex, embeddings: AsyncMock
    ) -> None:
        worker = SyncWorker(
            registry=registry,
            index=index,
            embeddings=embeddings,
            registry_settings=RegistrySettings(),
            tdwa_settings=TDWASettings(num_synthetic_questions=3),
            llm_settings=LLMSettings(),
        )
        registry.list_tools.return_value = [_make_tool("a")]

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
        self, registry: AsyncMock, index: ToolIndex, embeddings: AsyncMock
    ) -> None:
        worker = SyncWorker(
            registry=registry,
            index=index,
            embeddings=embeddings,
            registry_settings=RegistrySettings(),
            tdwa_settings=TDWASettings(num_synthetic_questions=3),
            llm_settings=LLMSettings(),
        )
        registry.list_tools.return_value = [_make_tool("a")]

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
            await worker.sync_once()

        t = await index.get_tool("a")
        assert t is not None
        assert t.synthetic_questions == []
