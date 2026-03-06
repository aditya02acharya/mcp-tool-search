"""Credential resolver with TTL-based caching.

Supports two backends:
1. AWS Secrets Manager – when ``secret_url`` and ``role_name`` are present in
   the server env list.
2. Local environment variables – fallback using the variable name declared in
   ``credentials.auth_value`` or ``static_headers``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from mcp_tool_router.models.schemas import MCPServerRecord
from mcp_tool_router.settings import MCPClientSettings

logger = logging.getLogger(__name__)


class _CacheEntry:
    """A cached secret value with TTL."""

    __slots__ = ("expires_at", "value")

    def __init__(self, value: str, ttl_seconds: int) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl_seconds

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self.expires_at


class CredentialResolver:
    """Resolve and cache authentication tokens for MCP servers."""

    def __init__(self, settings: MCPClientSettings) -> None:
        self._settings = settings
        self._cache: dict[str, _CacheEntry] = {}
        self._sm_client: Any | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def resolve(self, server: MCPServerRecord) -> str | None:
        """Return the auth token for *server*, or ``None`` if unauthenticated."""
        if not server.auth_type:
            return None

        cache_key = server.server_id
        entry = self._cache.get(cache_key)
        if entry and not entry.expired:
            return entry.value

        token = await self._fetch_token(server)
        if token:
            ttl = self._ttl_for(server)
            self._cache[cache_key] = _CacheEntry(token, ttl)
        return token

    def invalidate(self, server_id: str) -> None:
        """Remove a cached credential, forcing a fresh fetch next time."""
        self._cache.pop(server_id, None)

    async def close(self) -> None:
        """Clean up any resources."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _fetch_token(self, server: MCPServerRecord) -> str | None:
        """Fetch a token from the appropriate backend."""
        if server.has_remote_secret:
            return await self._fetch_from_aws(server)
        return self._fetch_from_env(server)

    async def _fetch_from_aws(self, server: MCPServerRecord) -> str | None:
        """Retrieve a secret from AWS Secrets Manager."""
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "boto3 is not installed; cannot fetch secret for server %s from AWS",
                server.server_id,
            )
            return None

        secret_url = os.environ.get("SECRET_URL", "")
        role_name = os.environ.get("ROLE_NAME", "")
        if not secret_url or not role_name:
            logger.warning(
                "secret_url or role_name env var not set for server %s",
                server.server_id,
            )
            return None

        try:
            if self._sm_client is None:
                sts = boto3.client("sts", region_name=self._settings.aws_region)
                assumed = sts.assume_role(
                    RoleArn=role_name,
                    RoleSessionName="mcp-tool-router",
                )
                creds = assumed["Credentials"]
                self._sm_client = boto3.client(
                    "secretsmanager",
                    region_name=self._settings.aws_region,
                    aws_access_key_id=creds["AccessKeyId"],
                    aws_secret_access_key=creds["SecretAccessKey"],
                    aws_session_token=creds["SessionToken"],
                )

            resp = self._sm_client.get_secret_value(SecretId=secret_url)
            secret_str: str = resp.get("SecretString", "")
            # Secret may be a JSON blob; try to extract an "auth_value" key
            try:
                data = json.loads(secret_str)
                if isinstance(data, dict):
                    var_name = (
                        server.credentials.auth_value if server.credentials else "auth_value"
                    )
                    return str(data.get(var_name, data.get("token", secret_str)))
            except (json.JSONDecodeError, TypeError):
                pass
            return secret_str
        except Exception:
            logger.exception("Failed to fetch secret from AWS for server %s", server.server_id)
            return None

    def _fetch_from_env(self, server: MCPServerRecord) -> str | None:
        """Resolve a token from a local environment variable."""
        # Try credentials.auth_value first
        if server.credentials and server.credentials.auth_value:
            value = os.environ.get(server.credentials.auth_value)
            if value:
                return value

        # Try extracting variable name from static_headers
        for header_value in server.static_headers.values():
            parts = header_value.split()
            if len(parts) == 2:
                # e.g. "Token <VARIABLE_NAME>" or "Bearer <VARIABLE_NAME>"
                var_name = parts[1]
                env_val = os.environ.get(var_name)
                if env_val:
                    return env_val

        logger.warning(
            "No credential found in environment for server %s (auth_type=%s)",
            server.server_id,
            server.auth_type,
        )
        return None

    def _ttl_for(self, server: MCPServerRecord) -> int:
        """Determine TTL in seconds for the credential cache entry."""
        if server.rotation_frequency_days is not None:
            return server.rotation_frequency_days * 86400
        return self._settings.credential_ttl_seconds
