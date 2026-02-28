"""Pydantic data models for tools, context entries, search results, and citations."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ToolRecord(BaseModel):
    """A tool fetched from the upstream registry."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str
    input_schema: dict[str, Any] = Field(alias="inputSchema", default_factory=dict)
    output_schema: dict[str, Any] | None = Field(alias="outputSchema", default=None)
    meta: dict[str, Any] | None = None

    @property
    def tags(self) -> list[str]:
        if not self.meta:
            return []
        for key in ("_fastmcp", "fastmcp"):
            sub = self.meta.get(key)
            if isinstance(sub, dict):
                tags = sub.get("tags")
                if isinstance(tags, list):
                    return [str(t) for t in tags if t]
        return []

    def content_hash(self) -> str:
        payload = json.dumps(
            {
                "name": self.name,
                "description": self.description,
                "input_schema": self.input_schema,
                "tags": self.tags,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class IndexedTool(BaseModel):
    """A tool that has been indexed locally with its TDWA embedding."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)
    content_hash: str = ""
    embedding: list[float] | None = None
    synthetic_questions: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    """A ranked search result returned to the agent."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)
    score: float = 0.0


class ContextEntry(BaseModel):
    """A single accumulated tool-call result stored in Redis."""

    call_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    session_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    snippet: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    content_size_bytes: int = 0


class Citation(BaseModel):
    """A reference to a specific context entry for the agent."""

    call_id: str
    tool_name: str
    excerpt: str
    relevance_score: float = 0.0


class ContextSummary(BaseModel):
    """Summary of accumulated session context with citation metadata."""

    summary: str
    gaps: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    total_entries: int = 0
    total_size_bytes: int = 0
