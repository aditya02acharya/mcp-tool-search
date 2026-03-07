"""Tests for the MCP client factory."""

from __future__ import annotations

import asyncio
from typing import Any
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx  # type: ignore[import-untyped]

from mcp_tool_router.mcp_client.factory import (
    MCPAuthError,
    MCPClientError,
    MCPClientFactory,
    MCPTransientError,
    _classify_error,
    _is_auth_error,
    _is_transient_error,
)
from mcp_tool_router.models.schemas import MCPServerCredentials, MCPServerRecord
from mcp_tool_router.settings import MCPClientSettings, RegistrySettings

REGISTRY_BASE = "http://test:4000"

SERVER_PAYLOAD = [
    {
        "server_id": "srv-1",
        "server_name": "Weather Server",
        "description": "Weather tools",
        "transport": "streamable_http",
        "auth_type": "bearer_token",
        "credentials": {"auth_value": "WEATHER_KEY"},
        "url": "http://weather:9090/mcp",
        "env": {},
        "static_headers": {},
    },
    {
        "server_id": "srv-2",
        "server_name": "Stock Server",
        "description": "Stock tools",
        "transport": "streamable_http",
        "auth_type": None,
        "credentials": None,
        "url": "http://stock:9090/mcp",
        "env": {},
        "static_headers": {},
    },
]


def _make_settings(**overrides: Any) -> MCPClientSettings:
    defaults = {
        "server_list_url": "/mcp/server",
        "connect_timeout_seconds": 5.0,
        "call_timeout_seconds": 10.0,
        "credential_ttl_seconds": 60,
        "pool_max_size": 10,
        # Fast retries for tests — no real waiting
        "retry_max_attempts": 3,
        "retry_wait_multiplier": 0.0,
        "retry_wait_min": 0.0,
        "retry_wait_max": 0.0,
    }
    defaults.update(overrides)
    return MCPClientSettings(**defaults)


@pytest.fixture
def mcp_settings() -> MCPClientSettings:
    return _make_settings()


@pytest.fixture
def registry_settings() -> RegistrySettings:
    return RegistrySettings(base_url=REGISTRY_BASE, api_key="sk-test")


@pytest.fixture
def factory(
    mcp_settings: MCPClientSettings, registry_settings: RegistrySettings
) -> MCPClientFactory:
    return MCPClientFactory(mcp_settings, registry_settings)


class TestInitialize:
    @respx.mock
    async def test_loads_servers(self, factory: MCPClientFactory) -> None:
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=SERVER_PAYLOAD)
        )
        await factory.initialize()
        servers = factory.get_servers()
        assert len(servers) == 2
        assert servers[0].server_id in ("srv-1", "srv-2")
        await factory.close()

    @respx.mock
    async def test_skips_non_streamable_http(self, factory: MCPClientFactory) -> None:
        payload = [
            {**SERVER_PAYLOAD[0], "transport": "stdio"},
            SERVER_PAYLOAD[1],
        ]
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=payload)
        )
        await factory.initialize()
        assert len(factory.get_servers()) == 1
        await factory.close()

    @respx.mock
    async def test_handles_wrapped_response(self, factory: MCPClientFactory) -> None:
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json={"servers": SERVER_PAYLOAD})
        )
        await factory.initialize()
        assert len(factory.get_servers()) == 2
        await factory.close()

    @respx.mock
    async def test_raises_on_http_error(self, factory: MCPClientFactory) -> None:
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(500, json={"error": "boom"})
        )
        with pytest.raises(httpx.HTTPStatusError):
            await factory.initialize()
        await factory.close()

    async def test_raises_when_not_initialized(self, factory: MCPClientFactory) -> None:
        with pytest.raises(MCPClientError, match="not been initialized"):
            await factory.refresh_servers()


class TestBuildHeaders:
    @respx.mock
    async def test_bearer_token_header(self, factory: MCPClientFactory) -> None:
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=[])
        )
        await factory.initialize()

        server = MCPServerRecord(
            server_id="s1",
            server_name="Test",
            url="http://test:8080",
            auth_type="bearer_token",
            credentials=MCPServerCredentials(auth_value="MY_TOK"),
        )
        with patch.dict(os.environ, {"MY_TOK": "abc123"}):
            headers = await factory._build_headers(server)
        assert headers["Authorization"] == "Bearer abc123"
        await factory.close()

    @respx.mock
    async def test_x_api_key_header(self, factory: MCPClientFactory) -> None:
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=[])
        )
        await factory.initialize()

        server = MCPServerRecord(
            server_id="s2",
            server_name="Test",
            url="http://test:8080",
            auth_type="x-api-key",
            credentials=MCPServerCredentials(auth_value="API_KEY_VAR"),
        )
        with patch.dict(os.environ, {"API_KEY_VAR": "key-xyz"}):
            headers = await factory._build_headers(server)
        assert headers["x-api-key"] == "key-xyz"
        await factory.close()

    @respx.mock
    async def test_authorization_with_token_prefix(self, factory: MCPClientFactory) -> None:
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=[])
        )
        await factory.initialize()

        server = MCPServerRecord(
            server_id="s3",
            server_name="Test",
            url="http://test:8080",
            auth_type="authorization",
            credentials=MCPServerCredentials(auth_value="AUTH_VAR"),
            static_headers={"Authorization": "Token AUTH_VAR"},
        )
        with patch.dict(os.environ, {"AUTH_VAR": "tok-789"}):
            headers = await factory._build_headers(server)
        assert headers["Authorization"] == "Token tok-789"
        await factory.close()

    @respx.mock
    async def test_no_auth_returns_empty(self, factory: MCPClientFactory) -> None:
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=[])
        )
        await factory.initialize()

        server = MCPServerRecord(
            server_id="s4",
            server_name="Test",
            url="http://test:8080",
            auth_type=None,
        )
        headers = await factory._build_headers(server)
        assert headers == {}
        await factory.close()


class TestValidation:
    def test_rejects_empty_server_id(self, factory: MCPClientFactory) -> None:
        with pytest.raises(MCPClientError, match="server_id"):
            factory._validate_call_input("", "tool", {})

    def test_rejects_empty_tool_name(self, factory: MCPClientFactory) -> None:
        with pytest.raises(MCPClientError, match="tool_name"):
            factory._validate_call_input("srv", "", {})

    def test_rejects_non_dict_arguments(self, factory: MCPClientFactory) -> None:
        with pytest.raises(MCPClientError, match="arguments must be a dict"):
            factory._validate_call_input("srv", "tool", "not-a-dict")  # type: ignore[arg-type]

    def test_rejects_oversized_argument(self, factory: MCPClientFactory) -> None:
        huge = "x" * 1_100_000
        with pytest.raises(MCPClientError, match="exceeds maximum length"):
            factory._validate_call_input("srv", "tool", {"data": huge})

    def test_accepts_valid_input(self, factory: MCPClientFactory) -> None:
        factory._validate_call_input("srv-1", "tool-a", {"key": "value"})

    def test_accepts_none_arguments(self, factory: MCPClientFactory) -> None:
        factory._validate_call_input("srv-1", "tool-a", None)


class TestCallToolError:
    @respx.mock
    async def test_unknown_server_raises(self, factory: MCPClientFactory) -> None:
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=[])
        )
        await factory.initialize()
        with pytest.raises(MCPClientError, match="Unknown server"):
            await factory.call_tool("nonexistent", "tool", {})
        await factory.close()


class TestRefreshServers:
    @respx.mock
    async def test_removes_stale_servers(self, factory: MCPClientFactory) -> None:
        # First load: 2 servers
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=SERVER_PAYLOAD)
        )
        await factory.initialize()
        assert len(factory.get_servers()) == 2

        # Second load: only 1 server
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=[SERVER_PAYLOAD[0]])
        )
        await factory.refresh_servers()
        assert len(factory.get_servers()) == 1
        assert factory.get_servers()[0].server_id == "srv-1"
        await factory.close()


class TestClose:
    @respx.mock
    async def test_close_cleans_up(self, factory: MCPClientFactory) -> None:
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=SERVER_PAYLOAD)
        )
        await factory.initialize()
        await factory.close()
        assert len(factory._servers) == 0
        assert factory._http is None


class TestIsAuthError:
    def test_detects_401(self) -> None:
        assert _is_auth_error(Exception("HTTP 401 Unauthorized"))

    def test_detects_403(self) -> None:
        assert _is_auth_error(Exception("HTTP 403 Forbidden"))

    def test_detects_unauthorized_keyword(self) -> None:
        assert _is_auth_error(Exception("unauthorized access"))

    def test_detects_forbidden_keyword(self) -> None:
        assert _is_auth_error(Exception("request forbidden"))

    def test_non_auth_error(self) -> None:
        assert not _is_auth_error(Exception("Connection refused"))


class TestIsTransientError:
    def test_detects_timeout(self) -> None:
        assert _is_transient_error(Exception("Connection timed out"))

    def test_detects_502(self) -> None:
        assert _is_transient_error(Exception("HTTP 502 Bad Gateway"))

    def test_detects_503(self) -> None:
        assert _is_transient_error(Exception("HTTP 503 Service Unavailable"))

    def test_detects_connection_reset(self) -> None:
        assert _is_transient_error(Exception("connection reset by peer"))

    def test_detects_connection_refused(self) -> None:
        assert _is_transient_error(Exception("Connection refused"))

    def test_non_transient_error(self) -> None:
        assert not _is_transient_error(Exception("Invalid JSON"))


class TestClassifyError:
    def test_auth_error_classified(self) -> None:
        result = _classify_error(Exception("HTTP 401 Unauthorized"))
        assert isinstance(result, MCPAuthError)

    def test_transient_error_classified(self) -> None:
        result = _classify_error(Exception("Connection timed out"))
        assert isinstance(result, MCPTransientError)

    def test_unknown_error_classified(self) -> None:
        result = _classify_error(Exception("Something unexpected"))
        assert isinstance(result, MCPClientError)
        assert not isinstance(result, (MCPAuthError, MCPTransientError))


class TestLRUPool:
    @respx.mock
    async def test_evicts_lru_when_pool_full(self) -> None:
        """When pool is at capacity, the least-recently-used client is evicted."""
        settings = _make_settings(pool_max_size=2)
        registry_settings = RegistrySettings(base_url=REGISTRY_BASE, api_key="sk-test")
        factory = MCPClientFactory(settings, registry_settings)

        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=[])
        )
        await factory.initialize()

        # Manually insert mock clients to simulate pool state
        mock_a = MagicMock()
        mock_a.is_connected.return_value = True
        mock_b = MagicMock()
        mock_b.is_connected.return_value = True
        mock_c_ctx = MagicMock()
        mock_c_ctx.is_connected.return_value = True

        factory._clients["a"] = mock_a
        factory._clients["b"] = mock_b
        factory._servers["c"] = MCPServerRecord(
            server_id="c",
            server_name="C",
            url="http://c:8080/mcp",
        )

        # Mock _connect to insert mock_c_ctx directly
        async def fake_connect(server: MCPServerRecord) -> MagicMock:
            async with factory._pool_lock:
                while len(factory._clients) >= factory._pool_max_size:
                    _evict_sid, _ = factory._clients.popitem(last=False)
                factory._clients[server.server_id] = mock_c_ctx
            return mock_c_ctx

        factory._connect = fake_connect  # type: ignore[assignment]

        result = await factory._get_or_connect("c")
        assert result is mock_c_ctx
        # "a" was the LRU and should have been evicted
        assert "a" not in factory._clients
        assert "b" in factory._clients
        assert "c" in factory._clients
        await factory.close()


class TestTenacityRetryCallTool:
    """Test tenacity-based retry logic for call_tool."""

    @respx.mock
    async def test_retries_on_auth_error_then_succeeds(self, factory: MCPClientFactory) -> None:
        """call_tool retries on 401 with credential invalidation and succeeds."""
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=SERVER_PAYLOAD)
        )
        await factory.initialize()

        mock_result = MagicMock()
        mock_result.content = []
        call_count = 0

        async def fake_call_tool(name: str, args: dict, timeout: float = 60) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("HTTP 401 Unauthorized")
            return mock_result

        # Insert a mock client; after disconnect (from before_sleep), _get_or_connect
        # needs to return a fresh client. We mock _connect for the reconnect path.
        mock_client = AsyncMock()
        mock_client.call_tool = fake_call_tool
        mock_client.is_connected.return_value = True
        factory._clients["srv-1"] = mock_client

        fresh_client = AsyncMock()
        fresh_client.call_tool = fake_call_tool
        fresh_client.is_connected.return_value = True

        async def fake_connect(server: MCPServerRecord) -> Any:
            async with factory._pool_lock:
                factory._clients[server.server_id] = fresh_client
            return fresh_client

        factory._connect = fake_connect  # type: ignore[assignment]

        # Track credential invalidation
        invalidated: list[str] = []
        orig_invalidate = factory._credentials.invalidate

        def tracking_invalidate(server_id: str) -> None:
            invalidated.append(server_id)
            orig_invalidate(server_id)

        factory._credentials.invalidate = tracking_invalidate  # type: ignore[assignment]

        result = await factory.call_tool("srv-1", "some-tool", {})
        assert result is mock_result
        assert call_count == 2
        # Verify credential cache was invalidated before retry
        assert "srv-1" in invalidated
        await factory.close()

    @respx.mock
    async def test_raises_after_all_retries_exhausted(self, factory: MCPClientFactory) -> None:
        """call_tool raises MCPAuthError when all retry attempts fail."""
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=SERVER_PAYLOAD)
        )
        await factory.initialize()

        async def always_fail(name: str, args: dict, timeout: float = 60) -> None:
            raise Exception("HTTP 401 Unauthorized")

        mock_client = AsyncMock()
        mock_client.call_tool = always_fail
        mock_client.is_connected.return_value = True
        factory._clients["srv-1"] = mock_client

        async def fake_connect(server: MCPServerRecord) -> Any:
            fresh = AsyncMock()
            fresh.call_tool = always_fail
            fresh.is_connected.return_value = True
            async with factory._pool_lock:
                factory._clients[server.server_id] = fresh
            return fresh

        factory._connect = fake_connect  # type: ignore[assignment]

        with pytest.raises(MCPAuthError):
            await factory.call_tool("srv-1", "some-tool", {})
        await factory.close()

    @respx.mock
    async def test_retries_on_transient_error_then_succeeds(
        self, factory: MCPClientFactory
    ) -> None:
        """call_tool retries transient errors with backoff and succeeds."""
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=SERVER_PAYLOAD)
        )
        await factory.initialize()

        mock_result = MagicMock()
        mock_result.content = []
        call_count = 0

        async def flaky_call(name: str, args: dict, timeout: float = 60) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Connection timed out")
            return mock_result

        mock_client = AsyncMock()
        mock_client.call_tool = flaky_call
        mock_client.is_connected.return_value = True
        factory._clients["srv-1"] = mock_client

        async def fake_connect(server: MCPServerRecord) -> Any:
            fresh = AsyncMock()
            fresh.call_tool = flaky_call
            fresh.is_connected.return_value = True
            async with factory._pool_lock:
                factory._clients[server.server_id] = fresh
            return fresh

        factory._connect = fake_connect  # type: ignore[assignment]

        result = await factory.call_tool("srv-1", "some-tool", {})
        assert result is mock_result
        assert call_count == 3
        await factory.close()

    @respx.mock
    async def test_non_retryable_error_raises_immediately(
        self, factory: MCPClientFactory
    ) -> None:
        """Non-auth, non-transient errors raise immediately without retry."""
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=SERVER_PAYLOAD)
        )
        await factory.initialize()

        call_count = 0

        async def bad_call(name: str, args: dict, timeout: float = 60) -> None:
            nonlocal call_count
            call_count += 1
            raise Exception("Invalid JSON response")

        mock_client = AsyncMock()
        mock_client.call_tool = bad_call
        mock_client.is_connected.return_value = True
        factory._clients["srv-1"] = mock_client

        with pytest.raises(MCPClientError):
            await factory.call_tool("srv-1", "some-tool", {})
        # Should have been called exactly once — no retries
        assert call_count == 1
        await factory.close()

    @respx.mock
    async def test_credential_invalidated_on_each_auth_retry(
        self, factory: MCPClientFactory
    ) -> None:
        """Credential cache is invalidated before each auth retry attempt."""
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=SERVER_PAYLOAD)
        )
        await factory.initialize()

        async def always_401(name: str, args: dict, timeout: float = 60) -> None:
            raise Exception("HTTP 401 Unauthorized")

        mock_client = AsyncMock()
        mock_client.call_tool = always_401
        mock_client.is_connected.return_value = True
        factory._clients["srv-1"] = mock_client

        async def fake_connect(server: MCPServerRecord) -> Any:
            fresh = AsyncMock()
            fresh.call_tool = always_401
            fresh.is_connected.return_value = True
            async with factory._pool_lock:
                factory._clients[server.server_id] = fresh
            return fresh

        factory._connect = fake_connect  # type: ignore[assignment]

        invalidation_count = 0
        orig_invalidate = factory._credentials.invalidate

        def counting_invalidate(server_id: str) -> None:
            nonlocal invalidation_count
            invalidation_count += 1
            orig_invalidate(server_id)

        factory._credentials.invalidate = counting_invalidate  # type: ignore[assignment]

        with pytest.raises(MCPAuthError):
            await factory.call_tool("srv-1", "some-tool", {})

        # With retry_max_attempts=3: 3 attempts, 2 sleeps (before_sleep fires
        # between attempts), so invalidation should happen twice.
        assert invalidation_count == 2
        await factory.close()


class TestTenacityRetryListTools:
    """Test tenacity-based retry logic for list_tools."""

    @respx.mock
    async def test_list_tools_retries_on_auth_error(self, factory: MCPClientFactory) -> None:
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=SERVER_PAYLOAD)
        )
        await factory.initialize()

        call_count = 0

        async def fake_list_tools() -> list:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("HTTP 401 Unauthorized")
            return []

        mock_client = AsyncMock()
        mock_client.list_tools = fake_list_tools
        mock_client.is_connected.return_value = True
        factory._clients["srv-1"] = mock_client

        fresh_client = AsyncMock()
        fresh_client.list_tools = fake_list_tools
        fresh_client.is_connected.return_value = True

        async def fake_connect(server: MCPServerRecord) -> Any:
            async with factory._pool_lock:
                factory._clients[server.server_id] = fresh_client
            return fresh_client

        factory._connect = fake_connect  # type: ignore[assignment]

        result = await factory.list_tools("srv-1")
        assert result == []
        assert call_count == 2
        await factory.close()

    @respx.mock
    async def test_list_tools_retries_on_transient_error(
        self, factory: MCPClientFactory
    ) -> None:
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=SERVER_PAYLOAD)
        )
        await factory.initialize()

        call_count = 0

        async def flaky_list() -> list:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("HTTP 503 Service Unavailable")
            return []

        mock_client = AsyncMock()
        mock_client.list_tools = flaky_list
        mock_client.is_connected.return_value = True
        factory._clients["srv-1"] = mock_client

        async def fake_connect(server: MCPServerRecord) -> Any:
            fresh = AsyncMock()
            fresh.list_tools = flaky_list
            fresh.is_connected.return_value = True
            async with factory._pool_lock:
                factory._clients[server.server_id] = fresh
            return fresh

        factory._connect = fake_connect  # type: ignore[assignment]

        result = await factory.list_tools("srv-1")
        assert result == []
        assert call_count == 2
        await factory.close()


class TestSafeCloseClient:
    async def test_close_connected_client(self) -> None:
        mock = MagicMock()
        mock.is_connected.return_value = True
        mock.__aexit__ = AsyncMock()
        await MCPClientFactory._safe_close_client(mock)
        mock.__aexit__.assert_awaited_once()

    async def test_close_disconnected_client(self) -> None:
        mock = MagicMock()
        mock.is_connected.return_value = False
        mock.__aexit__ = AsyncMock()
        await MCPClientFactory._safe_close_client(mock)
        mock.__aexit__.assert_not_awaited()

    async def test_close_ignores_errors(self) -> None:
        mock = MagicMock()
        mock.is_connected.return_value = True
        mock.__aexit__ = AsyncMock(side_effect=RuntimeError("boom"))
        # Should not raise
        await MCPClientFactory._safe_close_client(mock)


class TestListAllToolsParallel:
    @respx.mock
    async def test_gathers_in_parallel(self, factory: MCPClientFactory) -> None:
        """list_all_tools calls list_tools for all servers concurrently."""
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=SERVER_PAYLOAD)
        )
        await factory.initialize()

        call_order: list[str] = []

        async def fake_list_tools(sid: str) -> list:
            call_order.append(sid)
            await asyncio.sleep(0)  # yield to event loop
            return []

        factory.list_tools = fake_list_tools  # type: ignore[assignment]
        result = await factory.list_all_tools()
        assert len(result) == 2
        assert set(result.keys()) == {"srv-1", "srv-2"}
        await factory.close()

    @respx.mock
    async def test_handles_errors_gracefully(self, factory: MCPClientFactory) -> None:
        """list_all_tools returns empty list for failing servers."""
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=SERVER_PAYLOAD)
        )
        await factory.initialize()

        async def failing_list_tools(sid: str) -> list:
            if sid == "srv-1":
                raise MCPClientError("boom")
            return []

        factory.list_tools = failing_list_tools  # type: ignore[assignment]
        result = await factory.list_all_tools()
        assert result["srv-1"] == []
        assert result["srv-2"] == []
        await factory.close()


class TestDisconnectClient:
    @respx.mock
    async def test_disconnect_nonexistent_is_noop(self, factory: MCPClientFactory) -> None:
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=[])
        )
        await factory.initialize()
        # Should not raise
        await factory._disconnect_client("nonexistent")
        await factory.close()

    @respx.mock
    async def test_disconnect_removes_from_pool(self, factory: MCPClientFactory) -> None:
        respx.get(f"{REGISTRY_BASE}/mcp/server").mock(
            return_value=httpx.Response(200, json=[])
        )
        await factory.initialize()
        mock = MagicMock()
        mock.is_connected.return_value = False
        mock.__aexit__ = AsyncMock()
        factory._clients["srv-x"] = mock
        await factory._disconnect_client("srv-x")
        assert "srv-x" not in factory._clients
        await factory.close()
