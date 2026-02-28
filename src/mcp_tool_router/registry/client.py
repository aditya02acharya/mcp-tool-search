"""REST client for the upstream LiteLLM tool registry.

Fetches the canonical tool list, proxies tool calls, and provides
two-tier content hashing for change detection.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx

from mcp_tool_router.models.schemas import ToolRecord
from mcp_tool_router.settings import RegistrySettings


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
