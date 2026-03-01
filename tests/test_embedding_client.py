"""Tests for the vLLM embedding client and TDWA strategy."""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np
import pytest
import respx  # type: ignore[import-untyped]

from mcp_tool_router.embeddings.client import EmbeddingClient
from mcp_tool_router.settings import EmbeddingSettings, TDWASettings

DIM = 8


def _embedding_response(inputs: int, dim: int = DIM) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    return {
        "data": [
            {"index": i, "embedding": rng.standard_normal(dim).tolist()} for i in range(inputs)
        ]
    }


@pytest.fixture
def settings() -> EmbeddingSettings:
    return EmbeddingSettings(
        base_url="http://test:8000",
        model_name="test-model",
        dimension=DIM,
        batch_size=2,
        timeout_seconds=5.0,
    )


@pytest.fixture
def tdwa() -> TDWASettings:
    return TDWASettings()


@pytest.fixture
def client(settings: EmbeddingSettings) -> EmbeddingClient:
    return EmbeddingClient(settings)


class TestEmbedQuery:
    @respx.mock
    async def test_returns_vector(self, client: EmbeddingClient) -> None:
        respx.post("http://test:8000/v1/embeddings").mock(
            return_value=httpx.Response(200, json=_embedding_response(1))
        )
        vec = await client.embed_query("test query")
        assert vec.shape == (DIM,)
        assert vec.dtype == np.float32

    @respx.mock
    async def test_uses_instruction_prefix(self, client: EmbeddingClient) -> None:
        route = respx.post("http://test:8000/v1/embeddings").mock(
            return_value=httpx.Response(200, json=_embedding_response(1))
        )
        await client.embed_query("hello")
        body = json.loads(route.calls[0].request.content)
        assert body["input"][0].startswith("Instruct:")
        assert "hello" in body["input"][0]


class TestEmbedDocuments:
    @respx.mock
    async def test_empty_input(self, client: EmbeddingClient) -> None:
        result = await client.embed_documents([])
        assert result.shape == (0, DIM)

    @respx.mock
    async def test_batching(self, client: EmbeddingClient) -> None:
        # batch_size=2, 3 docs → 2 requests
        route = respx.post("http://test:8000/v1/embeddings")
        route.side_effect = [
            httpx.Response(200, json=_embedding_response(2)),
            httpx.Response(200, json=_embedding_response(1)),
        ]
        result = await client.embed_documents(["a", "b", "c"])
        assert result.shape == (3, DIM)
        assert route.call_count == 2


class TestTDWA:
    @respx.mock
    async def test_with_questions(self, client: EmbeddingClient, tdwa: TDWASettings) -> None:
        # 4 components → 1 request (batch_size=2 → 2 requests)
        respx.post("http://test:8000/v1/embeddings").mock(
            side_effect=[
                httpx.Response(200, json=_embedding_response(2)),
                httpx.Response(200, json=_embedding_response(2)),
            ]
        )
        vec = await client.embed_tool_tdwa(
            name="tool",
            description="does stuff",
            params_text="{}",
            synthetic_questions=["what?", "how?"],
            weights=tdwa,
        )
        assert vec.shape == (DIM,)
        # Should be L2-normalised
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    @respx.mock
    async def test_without_questions(self, client: EmbeddingClient, tdwa: TDWASettings) -> None:
        # 3 components, no questions → weights redistributed
        respx.post("http://test:8000/v1/embeddings").mock(
            side_effect=[
                httpx.Response(200, json=_embedding_response(2)),
                httpx.Response(200, json=_embedding_response(1)),
            ]
        )
        vec = await client.embed_tool_tdwa(
            name="tool",
            description="desc",
            params_text="{}",
            synthetic_questions=[],
            weights=tdwa,
        )
        assert vec.shape == (DIM,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5
