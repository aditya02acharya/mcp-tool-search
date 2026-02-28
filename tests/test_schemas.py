"""Tests for Pydantic data models."""

from __future__ import annotations

from mcp_tool_router.models.schemas import (
    Citation,
    ContextEntry,
    ContextSummary,
    IndexedTool,
    SearchResult,
    ToolRecord,
)


class TestToolRecord:
    def test_parse_with_alias(self) -> None:
        raw = {
            "name": "foo",
            "description": "does foo",
            "inputSchema": {"type": "object"},
            "outputSchema": None,
            "meta": {"_fastmcp": {"tags": ["a", "b"]}},
        }
        t = ToolRecord.model_validate(raw)
        assert t.name == "foo"
        assert t.input_schema == {"type": "object"}
        assert t.tags == ["a", "b"]

    def test_tags_from_fastmcp_key(self) -> None:
        t = ToolRecord(
            name="x",
            description="d",
            meta={"fastmcp": {"tags": ["t1"]}},
        )
        assert t.tags == ["t1"]

    def test_tags_empty_when_no_meta(self) -> None:
        t = ToolRecord(name="x", description="d", meta=None)
        assert t.tags == []

    def test_tags_empty_when_null_tags(self) -> None:
        t = ToolRecord(name="x", description="d", meta={"_fastmcp": {"tags": None}})
        assert t.tags == []

    def test_content_hash_deterministic(self) -> None:
        t = ToolRecord(name="a", description="b", input_schema={"x": 1})
        assert t.content_hash() == t.content_hash()

    def test_content_hash_changes(self) -> None:
        t1 = ToolRecord(name="a", description="b")
        t2 = ToolRecord(name="a", description="c")
        assert t1.content_hash() != t2.content_hash()


class TestIndexedTool:
    def test_defaults(self) -> None:
        t = IndexedTool(name="x", description="d")
        assert t.tags == []
        assert t.embedding is None
        assert t.synthetic_questions == []


class TestSearchResult:
    def test_fields(self) -> None:
        r = SearchResult(name="x", description="d", score=0.9)
        assert r.score == 0.9


class TestContextEntry:
    def test_auto_call_id(self) -> None:
        e = ContextEntry(session_id="s1", tool_name="t")
        assert len(e.call_id) == 12

    def test_timestamp_set(self) -> None:
        e = ContextEntry(session_id="s1", tool_name="t")
        assert e.timestamp is not None

    def test_serialisation_roundtrip(self) -> None:
        e = ContextEntry(session_id="s1", tool_name="t", result={"k": "v"})
        raw = e.model_dump_json()
        e2 = ContextEntry.model_validate_json(raw)
        assert e2.call_id == e.call_id
        assert e2.result == {"k": "v"}


class TestCitation:
    def test_fields(self) -> None:
        c = Citation(call_id="abc", tool_name="t", excerpt="e")
        assert c.relevance_score == 0.0


class TestContextSummary:
    def test_defaults(self) -> None:
        s = ContextSummary(summary="ok")
        assert s.gaps == []
        assert s.citations == []
