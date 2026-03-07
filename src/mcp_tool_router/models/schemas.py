"""Pydantic data models for tools, context entries, search results, and citations."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ToolRecord(BaseModel):
    """A tool fetched from a remote MCP server."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str
    server_id: str = ""
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
    server_id: str = ""
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
    server_id: str = ""
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


# ---------------------------------------------------------------------------
# MCP server registry models
# ---------------------------------------------------------------------------


class MCPServerCredentials(BaseModel):
    """Credential reference for an MCP server."""

    auth_value: str = ""


class MCPServerRecord(BaseModel):
    """An MCP server descriptor returned by the registry ``GET /mcp/server``."""

    model_config = ConfigDict(populate_by_name=True)

    server_id: str
    server_name: str
    description: str = ""
    transport: str = "streamable_http"
    auth_type: str | None = None
    credentials: MCPServerCredentials | None = None
    url: str
    env: dict[str, str] = Field(default_factory=dict)
    static_headers: dict[str, str] = Field(default_factory=dict)

    @property
    def has_remote_secret(self) -> bool:
        """True when the secret is stored in AWS Secrets Manager."""
        return bool(self.env.get("secret_url")) and bool(self.env.get("role_name"))

    @property
    def rotation_frequency_days(self) -> int | None:
        """Return rotation frequency if declared, else None."""
        val = self.env.get("rotation_frequency_days")
        if val is None:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None
