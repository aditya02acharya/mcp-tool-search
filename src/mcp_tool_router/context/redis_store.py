"""Redis-backed session context store for tool-call result accumulation.

Each session's data is partitioned as:
  ``mcp:ctx:{session_id}``         – SET of call_ids
  ``mcp:ctx:{session_id}:{cid}``   – STRING with serialised ContextEntry
  ``mcp:ctx:{session_id}:size``    – INT tracking total byte usage

All keys share the same TTL which is reset on every write.
"""

from __future__ import annotations

import logging
from typing import Any

import redis.asyncio as aioredis

from mcp_tool_router.models.schemas import ContextEntry
from mcp_tool_router.settings import RedisSettings

logger = logging.getLogger(__name__)


class ContextOverflowError(Exception):
    """Raised when a session exceeds ``max_context_bytes``."""


class RedisContextStore:
    """Async Redis store for accumulated tool-call results."""

    def __init__(self, settings: RedisSettings) -> None:
        self._settings = settings
        self._client: aioredis.Redis | None = None  # type: ignore[type-arg]

    async def connect(self) -> None:
        self._client = aioredis.Redis(
            host=self._settings.host,
            port=self._settings.port,
            db=self._settings.db,
            password=self._settings.password,
            ssl=self._settings.ssl,
            decode_responses=True,
        )

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    @staticmethod
    def _session_key(session_id: str) -> str:
        return f"mcp:ctx:{session_id}"

    @staticmethod
    def _entry_key(session_id: str, call_id: str) -> str:
        return f"mcp:ctx:{session_id}:{call_id}"

    @staticmethod
    def _size_key(session_id: str) -> str:
        return f"mcp:ctx:{session_id}:size"

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def accumulate(self, entry: ContextEntry) -> str:
        """Store a tool-call result; returns the ``call_id``."""
        assert self._client is not None
        data = entry.model_dump_json()
        size = len(data.encode())
        entry.content_size_bytes = size

        current = await self._get_session_size(entry.session_id)
        if current + size > self._settings.max_context_bytes:
            raise ContextOverflowError(
                f"Session {entry.session_id} would exceed "
                f"{self._settings.max_context_bytes} bytes"
            )

        ttl = self._settings.session_ttl_seconds
        sk = self._session_key(entry.session_id)
        ek = self._entry_key(entry.session_id, entry.call_id)
        szk = self._size_key(entry.session_id)

        pipe = self._client.pipeline()
        pipe.set(ek, data, ex=ttl)
        pipe.sadd(sk, entry.call_id)
        pipe.expire(sk, ttl)
        pipe.incrby(szk, size)
        pipe.expire(szk, ttl)
        await pipe.execute()

        return entry.call_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_entries(
        self,
        session_id: str,
        call_ids: list[str] | None = None,
    ) -> list[ContextEntry]:
        """Retrieve context entries, optionally filtered by *call_ids*."""
        assert self._client is not None
        if call_ids is None:
            call_ids = list(await self._client.smembers(self._session_key(session_id)))
        if not call_ids:
            return []

        pipe = self._client.pipeline()
        for cid in call_ids:
            pipe.get(self._entry_key(session_id, cid))
        raw: list[Any] = await pipe.execute()

        entries = [ContextEntry.model_validate_json(data) for data in raw if data is not None]
        return sorted(entries, key=lambda e: e.timestamp)

    async def get_entry_metadata(self, session_id: str) -> list[dict[str, Any]]:
        """Lightweight metadata (call_id, tool_name, snippet) without full results."""
        entries = await self.get_entries(session_id)
        return [
            {
                "call_id": e.call_id,
                "tool_name": e.tool_name,
                "snippet": e.snippet,
                "timestamp": e.timestamp.isoformat(),
                "size": e.content_size_bytes,
            }
            for e in entries
        ]

    async def get_session_info(self, session_id: str) -> dict[str, int]:
        assert self._client is not None
        total_entries = await self._client.scard(self._session_key(session_id))
        total_size = await self._get_session_size(session_id)
        return {"total_entries": total_entries, "total_size_bytes": total_size}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_session_size(self, session_id: str) -> int:
        assert self._client is not None
        val = await self._client.get(self._size_key(session_id))
        return int(val) if val else 0
