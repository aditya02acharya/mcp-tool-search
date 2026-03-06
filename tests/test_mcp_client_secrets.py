"""Tests for the credential resolver with TTL-based caching."""

from __future__ import annotations

import os
import time
from unittest.mock import patch

import pytest

from mcp_tool_router.mcp_client.secrets import CredentialResolver, _CacheEntry
from mcp_tool_router.models.schemas import MCPServerCredentials, MCPServerRecord
from mcp_tool_router.settings import MCPClientSettings


@pytest.fixture
def mcp_settings() -> MCPClientSettings:
    return MCPClientSettings(credential_ttl_seconds=60)


@pytest.fixture
def resolver(mcp_settings: MCPClientSettings) -> CredentialResolver:
    return CredentialResolver(mcp_settings)


def _make_server(
    *,
    server_id: str = "srv-1",
    auth_type: str | None = "bearer_token",
    credentials: MCPServerCredentials | None = None,
    env: list[str] | None = None,
    static_headers: dict[str, str] | None = None,
) -> MCPServerRecord:
    return MCPServerRecord(
        server_id=server_id,
        server_name="Test Server",
        url="http://test:8080",
        auth_type=auth_type,
        credentials=credentials,
        env=env or [],
        static_headers=static_headers or {},
    )


class TestCacheEntry:
    def test_not_expired(self) -> None:
        entry = _CacheEntry("tok", ttl_seconds=10)
        assert not entry.expired

    def test_expired(self) -> None:
        entry = _CacheEntry("tok", ttl_seconds=0)
        # Immediately expired with 0 ttl
        time.sleep(0.01)
        assert entry.expired


class TestResolveNoAuth:
    async def test_returns_none_for_no_auth_type(self, resolver: CredentialResolver) -> None:
        server = _make_server(auth_type=None)
        assert await resolver.resolve(server) is None


class TestResolveFromEnv:
    async def test_resolves_from_credentials_auth_value(
        self, resolver: CredentialResolver
    ) -> None:
        server = _make_server(
            credentials=MCPServerCredentials(auth_value="MY_TOKEN"),
        )
        with patch.dict(os.environ, {"MY_TOKEN": "secret-123"}):
            token = await resolver.resolve(server)
        assert token == "secret-123"

    async def test_resolves_from_static_headers_variable(
        self, resolver: CredentialResolver
    ) -> None:
        server = _make_server(
            auth_type="authorization",
            static_headers={"Authorization": "Token MY_HEADER_VAR"},
        )
        with patch.dict(os.environ, {"MY_HEADER_VAR": "hdr-456"}):
            token = await resolver.resolve(server)
        assert token == "hdr-456"

    async def test_returns_none_when_env_missing(self, resolver: CredentialResolver) -> None:
        server = _make_server(
            credentials=MCPServerCredentials(auth_value="MISSING_VAR"),
        )
        with patch.dict(os.environ, {}, clear=False):
            # Ensure MISSING_VAR is not set
            os.environ.pop("MISSING_VAR", None)
            token = await resolver.resolve(server)
        assert token is None


class TestCaching:
    async def test_caches_token(self, resolver: CredentialResolver) -> None:
        server = _make_server(
            credentials=MCPServerCredentials(auth_value="CACHED_TOK"),
        )
        with patch.dict(os.environ, {"CACHED_TOK": "val-1"}):
            t1 = await resolver.resolve(server)

        # Even after removing the env var, cache should return the value
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CACHED_TOK", None)
            t2 = await resolver.resolve(server)

        assert t1 == "val-1"
        assert t2 == "val-1"

    async def test_invalidate_clears_cache(self, resolver: CredentialResolver) -> None:
        server = _make_server(
            server_id="srv-inv",
            credentials=MCPServerCredentials(auth_value="INV_TOK"),
        )
        with patch.dict(os.environ, {"INV_TOK": "old"}):
            await resolver.resolve(server)

        resolver.invalidate("srv-inv")

        with patch.dict(os.environ, {"INV_TOK": "new"}):
            token = await resolver.resolve(server)
        assert token == "new"


class TestTTL:
    async def test_default_ttl(self, resolver: CredentialResolver) -> None:
        server = _make_server()
        ttl = resolver._ttl_for(server)
        assert ttl == 60  # from settings

    async def test_ttl_without_rotation(self, resolver: CredentialResolver) -> None:
        server = _make_server(env=[])
        ttl = resolver._ttl_for(server)
        assert ttl == 60


class TestRemoteSecret:
    async def test_has_remote_secret_true(self) -> None:
        server = _make_server(env=["secret_url", "role_name", "rotation_frequency_days"])
        assert server.has_remote_secret

    async def test_has_remote_secret_false(self) -> None:
        server = _make_server(env=[])
        assert not server.has_remote_secret

    async def test_falls_back_when_boto3_missing(self, resolver: CredentialResolver) -> None:
        server = _make_server(
            env=["secret_url", "role_name"],
        )
        with (
            patch.dict(os.environ, {"SECRET_URL": "arn:aws:sm:us-east-1:123:secret/test",
                                     "ROLE_NAME": "arn:aws:iam::123:role/test"}),
            patch.dict("sys.modules", {"boto3": None}),
        ):
            token = await resolver.resolve(server)
        assert token is None


class TestClose:
    async def test_close_clears_cache(self, resolver: CredentialResolver) -> None:
        resolver._cache["test"] = _CacheEntry("v", 100)
        await resolver.close()
        assert len(resolver._cache) == 0
