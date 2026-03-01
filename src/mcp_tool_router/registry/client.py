"""REST client for the upstream LiteLLM tool registry.

Fetches the canonical tool list and MCP server metadata, proxies tool
calls, and provides two-tier content hashing for change detection.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import httpx

from mcp_tool_router.models.schemas import ServerRecord, ToolRecord
from mcp_tool_router.settings import RegistrySettings

logger = logging.getLogger(__name__)


class RegistryClient:
    """Async HTTP client for the upstream tool registry."""

    def __init__(self, settings: RegistrySettings) -> None:
        self._settings = settings
        headers: dict[str, str] = {}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            headers=headers,
        )

    async def list_tools(self) -> list[ToolRecord]:
        """Fetch all tools from the upstream registry."""
        resp = await self._client.get("/v1/tools")
        resp.raise_for_status()
        data = resp.json()
        items: list[dict[str, Any]] = (
            data if isinstance(data, list) else data.get("tools", data.get("data", []))
        )
        return [ToolRecord.model_validate(t) for t in items]

    async def list_servers(self) -> list[ServerRecord]:
        """Fetch MCP server metadata from the upstream registry."""
        resp = await self._client.get("/mcp/")
        resp.raise_for_status()
        data = resp.json()
        items: list[dict[str, Any]] = (
            data if isinstance(data, list) else data.get("data", data.get("servers", []))
        )
        servers: list[ServerRecord] = []
        for item in items:
            mcp_info = item.get("mcp_info") or {}
            servers.append(
                ServerRecord(
                    server_id=item.get("server_id", ""),
                    server_name=item.get("server_name", mcp_info.get("server_name", "")),
                    alias=item.get("alias"),
                    description=mcp_info.get("description", item.get("description", "")),
                )
            )
        return servers

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Proxy a tool call to the upstream registry."""
        resp = await self._client.post(
            f"/v1/tools/{tool_name}/call",
            json={"arguments": arguments},
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Server-tool resolution
    # ------------------------------------------------------------------

    @staticmethod
    def build_server_lookup(servers: list[ServerRecord]) -> dict[str, ServerRecord]:
        """Build an alias -> ServerRecord lookup for tool-to-server resolution."""
        lookup: dict[str, ServerRecord] = {}
        for s in servers:
            key = s.alias or s.server_name
            if key:
                lookup[key] = s
        return lookup

    @staticmethod
    def resolve_tool_server(
        tool_name: str, server_lookup: dict[str, ServerRecord]
    ) -> ServerRecord | None:
        """Resolve a tool's parent server via LiteLLM alias prefix matching.

        LiteLLM namespaces tools as ``{server_alias}_{tool_name}``.  We try
        longest-prefix matching to handle aliases that contain underscores.
        """
        # Sort by alias length descending for longest-prefix-first matching
        for alias in sorted(server_lookup, key=len, reverse=True):
            if tool_name.startswith(f"{alias}_"):
                return server_lookup[alias]
        return None

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    @staticmethod
    def compute_global_hash(tools: list[ToolRecord]) -> str:
        """SHA-256 of the entire sorted, serialised tool list (quick equality check)."""
        payload = json.dumps(
            [t.model_dump(by_alias=True) for t in sorted(tools, key=lambda t: t.name)],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()
