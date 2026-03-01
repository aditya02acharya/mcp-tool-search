"""Tests for the upstream registry REST client."""

from __future__ import annotations

import httpx
import pytest
import respx  # type: ignore[import-untyped]

from mcp_tool_router.models.schemas import ServerRecord, ToolRecord
from mcp_tool_router.registry.client import RegistryClient
from mcp_tool_router.settings import RegistrySettings


@pytest.fixture
def settings() -> RegistrySettings:
    return RegistrySettings(base_url="http://test:4000", api_key="sk-test")


@pytest.fixture
def client(settings: RegistrySettings) -> RegistryClient:
    return RegistryClient(settings)


TOOLS_PAYLOAD = [
    {
        "name": "weather",
        "description": "Get weather",
        "inputSchema": {"type": "object"},
        "outputSchema": None,
        "meta": {"_fastmcp": {"tags": ["api"]}},
    },
    {
        "name": "stock",
        "description": "Get stock price",
        "inputSchema": {"type": "object"},
        "outputSchema": None,
        "meta": None,
    },
]

SERVERS_PAYLOAD = [
    {
        "server_id": "srv-1",
        "server_name": "weather-mcp",
        "alias": "weather",
        "mcp_info": {
            "server_name": "weather-mcp",
            "description": "Weather data and forecast APIs",
        },
    },
    {
        "server_id": "srv-2",
        "server_name": "github-mcp",
        "alias": "github",
        "mcp_info": {
            "server_name": "github-mcp",
            "description": "GitHub integration for code management",
        },
    },
]


class TestListTools:
    @respx.mock
    async def test_list_array_response(self, client: RegistryClient) -> None:
        respx.get("http://test:4000/v1/tools").mock(
            return_value=httpx.Response(200, json=TOOLS_PAYLOAD)
        )
        tools = await client.list_tools()
        assert len(tools) == 2
        assert tools[0].name == "weather"

    @respx.mock
    async def test_list_wrapped_response(self, client: RegistryClient) -> None:
        respx.get("http://test:4000/v1/tools").mock(
            return_value=httpx.Response(200, json={"tools": TOOLS_PAYLOAD})
        )
        tools = await client.list_tools()
        assert len(tools) == 2

    @respx.mock
    async def test_auth_header(self, client: RegistryClient) -> None:
        route = respx.get("http://test:4000/v1/tools").mock(
            return_value=httpx.Response(200, json=[])
        )
        await client.list_tools()
        assert route.calls[0].request.headers["authorization"] == "Bearer sk-test"


class TestListServers:
    @respx.mock
    async def test_list_array_response(self, client: RegistryClient) -> None:
        respx.get("http://test:4000/mcp/").mock(
            return_value=httpx.Response(200, json=SERVERS_PAYLOAD)
        )
        servers = await client.list_servers()
        assert len(servers) == 2
        assert servers[0].server_id == "srv-1"
        assert servers[0].server_name == "weather-mcp"
        assert servers[0].alias == "weather"
        assert servers[0].description == "Weather data and forecast APIs"

    @respx.mock
    async def test_list_wrapped_response(self, client: RegistryClient) -> None:
        respx.get("http://test:4000/mcp/").mock(
            return_value=httpx.Response(200, json={"data": SERVERS_PAYLOAD})
        )
        servers = await client.list_servers()
        assert len(servers) == 2

    @respx.mock
    async def test_description_from_mcp_info(self, client: RegistryClient) -> None:
        respx.get("http://test:4000/mcp/").mock(
            return_value=httpx.Response(200, json=SERVERS_PAYLOAD)
        )
        servers = await client.list_servers()
        assert servers[1].description == "GitHub integration for code management"


class TestCallTool:
    @respx.mock
    async def test_proxies_call(self, client: RegistryClient) -> None:
        respx.post("http://test:4000/v1/tools/weather/call").mock(
            return_value=httpx.Response(200, json={"temp": 15})
        )
        result = await client.call_tool("weather", {"loc": "London"})
        assert result == {"temp": 15}


class TestServerToolResolution:
    def test_build_server_lookup(self) -> None:
        servers = [
            ServerRecord(server_id="s1", server_name="weather-mcp", alias="weather", description="d"),
            ServerRecord(server_id="s2", server_name="github-mcp", alias="github", description="d"),
        ]
        lookup = RegistryClient.build_server_lookup(servers)
        assert "weather" in lookup
        assert "github" in lookup
        assert lookup["weather"].server_id == "s1"

    def test_build_lookup_uses_server_name_when_no_alias(self) -> None:
        servers = [
            ServerRecord(server_id="s1", server_name="myserver", description="d"),
        ]
        lookup = RegistryClient.build_server_lookup(servers)
        assert "myserver" in lookup

    def test_resolve_tool_server(self) -> None:
        servers = [
            ServerRecord(server_id="s1", server_name="weather-mcp", alias="weather", description="d"),
            ServerRecord(server_id="s2", server_name="github-mcp", alias="github", description="d"),
        ]
        lookup = RegistryClient.build_server_lookup(servers)
        result = RegistryClient.resolve_tool_server("weather_get_forecast", lookup)
        assert result is not None
        assert result.server_id == "s1"

    def test_resolve_tool_server_no_match(self) -> None:
        lookup = RegistryClient.build_server_lookup([
            ServerRecord(server_id="s1", server_name="weather-mcp", alias="weather", description="d"),
        ])
        result = RegistryClient.resolve_tool_server("unknown_tool", lookup)
        assert result is None

    def test_resolve_longest_prefix(self) -> None:
        """Aliases with underscores should match longest prefix first."""
        servers = [
            ServerRecord(server_id="s1", server_name="a", alias="my", description="d"),
            ServerRecord(server_id="s2", server_name="b", alias="my_server", description="d"),
        ]
        lookup = RegistryClient.build_server_lookup(servers)
        result = RegistryClient.resolve_tool_server("my_server_do_thing", lookup)
        assert result is not None
        assert result.server_id == "s2"


class TestGlobalHash:
    def test_deterministic(self) -> None:
        tools = [
            ToolRecord(name="a", description="d1"),
            ToolRecord(name="b", description="d2"),
        ]
        h1 = RegistryClient.compute_global_hash(tools)
        h2 = RegistryClient.compute_global_hash(tools)
        assert h1 == h2

    def test_order_independent(self) -> None:
        t1 = ToolRecord(name="a", description="d1")
        t2 = ToolRecord(name="b", description="d2")
        h1 = RegistryClient.compute_global_hash([t1, t2])
        h2 = RegistryClient.compute_global_hash([t2, t1])
        assert h1 == h2

    def test_changes_on_diff(self) -> None:
        t1 = [ToolRecord(name="a", description="d1")]
        t2 = [ToolRecord(name="a", description="d2")]
        assert RegistryClient.compute_global_hash(t1) != RegistryClient.compute_global_hash(t2)
