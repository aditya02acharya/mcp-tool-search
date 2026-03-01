"""Shared fixtures for the test suite."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import fakeredis.aioredis  # type: ignore[import-untyped]
import numpy as np
import pytest

from mcp_tool_router.context.compressor import ContextCompressor
from mcp_tool_router.context.redis_store import RedisContextStore
from mcp_tool_router.embeddings.client import EmbeddingClient
from mcp_tool_router.index.store import ToolIndex
from mcp_tool_router.models.schemas import ContextEntry, IndexedServer, IndexedTool, ServerRecord, ToolRecord
from mcp_tool_router.settings import (
    AppSettings,
    EmbeddingSettings,
    IndexSettings,
    LLMSettings,
    RedisSettings,
    RegistrySettings,
    TDWASettings,
)

# Force dev environment
os.environ.setdefault("ENVIRONMENT", "dev")


# ---------------------------------------------------------------------------
# Settings fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedding_settings() -> EmbeddingSettings:
    return EmbeddingSettings(
        base_url="http://test:8000",
        model_name="test-model",
        dimension=8,
        batch_size=4,
    )


@pytest.fixture
def index_settings(tmp_path: Any) -> IndexSettings:
    return IndexSettings(
        db_path=str(tmp_path / "test.db"),
        use_vec_extension=False,
        similarity_chunk_size=10,
    )


@pytest.fixture
def redis_settings() -> RedisSettings:
    return RedisSettings(
        host="localhost",
        port=6379,
        session_ttl_seconds=60,
        max_context_bytes=1024,
    )


@pytest.fixture
def llm_settings() -> LLMSettings:
    return LLMSettings(model_name="test-llm")


@pytest.fixture
def registry_settings() -> RegistrySettings:
    return RegistrySettings(base_url="http://test:4000")


@pytest.fixture
def tdwa_settings() -> TDWASettings:
    return TDWASettings(
        name_weight=0.10,
        description_weight=0.30,
        params_weight=0.20,
        questions_weight=0.30,
        server_description_weight=0.10,
        num_synthetic_questions=3,
    )


@pytest.fixture
def app_settings(index_settings: IndexSettings) -> AppSettings:
    return AppSettings(
        environment="dev",
        index=index_settings,
    )


# ---------------------------------------------------------------------------
# Service fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def tool_index(index_settings: IndexSettings) -> AsyncIterator[ToolIndex]:
    idx = ToolIndex(index_settings)
    await idx.initialize()
    yield idx
    idx.close()


@pytest.fixture
async def redis_store(redis_settings: RedisSettings) -> AsyncIterator[RedisContextStore]:
    store = RedisContextStore(redis_settings)
    store._client = fakeredis.aioredis.FakeRedis(decode_responses=True)  # type: ignore[assignment]
    yield store
    await store.close()


@pytest.fixture
def compressor(llm_settings: LLMSettings) -> ContextCompressor:
    return ContextCompressor(llm_settings)


@pytest.fixture
def embedding_client(embedding_settings: EmbeddingSettings) -> EmbeddingClient:
    return EmbeddingClient(embedding_settings)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

DIM = 8


@pytest.fixture
def sample_tool_record() -> ToolRecord:
    return ToolRecord.model_validate(
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "inputSchema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
            "outputSchema": {"type": "object"},
            "meta": {"_fastmcp": {"tags": ["weather", "api"]}},
        }
    )


@pytest.fixture
def sample_indexed_tool() -> IndexedTool:
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(DIM).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return IndexedTool(
        name="get_weather",
        description="Get current weather for a location",
        input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
        tags=["weather", "api"],
        content_hash="abc123",
        embedding=emb.tolist(),
        synthetic_questions=["What is the weather in London?"],
    )


@pytest.fixture
def sample_context_entry() -> ContextEntry:
    return ContextEntry(
        session_id="sess-1",
        tool_name="get_weather",
        arguments={"location": "London"},
        result={"temp": 15, "conditions": "cloudy"},
        snippet='{"temp": 15, "conditions": "cloudy"}',
    )


def make_indexed_tool(name: str, desc: str, tags: list[str] | None = None) -> IndexedTool:
    """Helper to create an IndexedTool with a random embedding."""
    rng = np.random.default_rng(hash(name) % 2**31)
    emb = rng.standard_normal(DIM).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return IndexedTool(
        name=name,
        description=desc,
        input_schema={},
        tags=tags or [],
        content_hash=f"hash-{name}",
        embedding=emb.tolist(),
    )


@pytest.fixture
def sample_server_record() -> ServerRecord:
    return ServerRecord(
        server_id="srv-1",
        server_name="weather-server",
        alias="weather",
        description="Weather data and forecast APIs",
    )


@pytest.fixture
def sample_indexed_server() -> IndexedServer:
    return IndexedServer(
        server_id="srv-1",
        server_name="weather-server",
        alias="weather",
        description="Weather data and forecast APIs",
        content_hash="srv-hash-1",
    )
