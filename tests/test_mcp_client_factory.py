"""Tests for the MCP client factory."""

from __future__ import annotations

import os
from unittest.mock import patch

import httpx
import pytest
import respx  # type: ignore[import-untyped]

from mcp_tool_router.mcp_client.factory import MCPClientError, MCPClientFactory
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
        "env": [],
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
        "env": [],
        "static_headers": {},
    },
]


@pytest.fixture
def mcp_settings() -> MCPClientSettings:
    return MCPClientSettings(
        server_list_url="/mcp/server",
        connect_timeout_seconds=5.0,
        call_timeout_seconds=10.0,
        credential_ttl_seconds=60,
    )


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
