"""Tests for the LLM context compressor (mocked LiteLLM)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from mcp_tool_router.context.compressor import (
    ContextCompressor,
    _build_prompt,
    _format_entry,
    _parse_response,
)
from mcp_tool_router.models.schemas import ContextEntry
from mcp_tool_router.settings import LLMSettings


def _llm_response(content: str) -> SimpleNamespace:
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


@pytest.fixture
def compressor(llm_settings: LLMSettings) -> ContextCompressor:
    return ContextCompressor(llm_settings)


@pytest.fixture
def entries() -> list[ContextEntry]:
    return [
        ContextEntry(
            call_id="aaa",
            session_id="s1",
            tool_name="weather",
            arguments={"loc": "London"},
            result={"temp": 15},
            content_size_bytes=50,
        ),
        ContextEntry(
            call_id="bbb",
            session_id="s1",
            tool_name="stock",
            arguments={"sym": "AAPL"},
            result={"price": 180},
            content_size_bytes=45,
        ),
    ]


class TestSummarise:
    async def test_returns_summary(
        self, compressor: ContextCompressor, entries: list[ContextEntry]
    ) -> None:
        resp_json = json.dumps(
            {
                "summary": "Weather is 15C, AAPL is 180",
                "gaps": ["no forecast"],
                "citations": [
                    {
                        "call_id": "aaa",
                        "tool_name": "weather",
                        "excerpt": "temp: 15",
                        "relevance_score": 0.9,
                    }
                ],
            }
        )
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = _llm_response(resp_json)
            result = await compressor.summarise(entries, "what's the weather?")

        assert "15" in result.summary
        assert len(result.citations) == 1
        assert result.total_entries == 2
        assert result.total_size_bytes == 95

    async def test_empty_entries(self, compressor: ContextCompressor) -> None:
        result = await compressor.summarise([], "anything")
        assert "No context" in result.summary
        assert result.gaps == ["anything"]


class TestQuickSummarise:
    async def test_returns_dict(self, compressor: ContextCompressor) -> None:
        entry = ContextEntry(session_id="s1", tool_name="t", result={"k": "v"})
        resp = json.dumps({"summary": "ok", "gaps": []})
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = _llm_response(resp)
            result = await compressor.quick_summarise(entry)

        assert result["summary"] == "ok"


class TestHelpers:
    def test_format_entry(self) -> None:
        e = ContextEntry(call_id="xyz", session_id="s", tool_name="t", result={"a": 1})
        text = _format_entry(e)
        assert "[xyz]" in text
        assert "Tool: t" in text

    def test_build_prompt(self) -> None:
        prompt = _build_prompt("my query", "some context")
        assert "my query" in prompt
        assert "some context" in prompt

    def test_parse_valid_json(self) -> None:
        data = _parse_response('{"summary": "ok"}')
        assert data["summary"] == "ok"

    def test_parse_invalid_json(self) -> None:
        data = _parse_response("not json at all")
        assert data["summary"] == "not json at all"
