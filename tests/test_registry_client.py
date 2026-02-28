"""Tests for the upstream registry REST client."""

from __future__ import annotations

import httpx
import pytest
import respx  # type: ignore[import-untyped]

from mcp_tool_router.models.schemas import ToolRecord
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


class TestCallTool:
    @respx.mock
    async def test_proxies_call(self, client: RegistryClient) -> None:
        respx.post("http://test:4000/v1/tools/weather/call").mock(
            return_value=httpx.Response(200, json={"temp": 15})
        )
        result = await client.call_tool("weather", {"loc": "London"})
        assert result == {"temp": 15}


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
