"""MCP Client Factory – manages connections to remote MCP servers.

Uses ``fastmcp.Client`` with ``StreamableHttpTransport`` to interact with
remote MCP servers listed by the registry via ``GET /mcp/server``.

Features:
- Lazy connection establishment per server
- Authentication header injection (bearer_token, authorization, x-api-key)
- AsyncExitStack-based resource management
- Server input/output validation
- Graceful disconnection handling
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any

import httpx
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import CallToolResult, Tool

from mcp_tool_router.mcp_client.secrets import CredentialResolver
from mcp_tool_router.models.schemas import MCPServerRecord
from mcp_tool_router.settings import MCPClientSettings, RegistrySettings

logger = logging.getLogger(__name__)

# Maximum allowed length for a single tool argument value (bytes).
_MAX_ARG_VALUE_LEN = 1_000_000
# Maximum allowed response content length (chars) before truncation.
_MAX_RESPONSE_LEN = 5_000_000


class MCPClientError(Exception):
    """Raised when an MCP client operation fails."""


class MCPClientFactory:
    """Manage connections to remote MCP servers discovered from the registry.

    Usage::

        factory = MCPClientFactory(mcp_settings, registry_settings)
        await factory.initialize()
        tools = await factory.list_all_tools()
        result = await factory.call_tool("server-id", "tool-name", {"arg": "val"})
        await factory.close()
    """

    def __init__(
        self,
        mcp_settings: MCPClientSettings,
        registry_settings: RegistrySettings,
    ) -> None:
        self._mcp_settings = mcp_settings
        self._registry_settings = registry_settings
        self._credentials = CredentialResolver(mcp_settings)
        self._http: httpx.AsyncClient | None = None
        self._servers: dict[str, MCPServerRecord] = {}
        self._clients: dict[str, Client] = {}
        self._exit_stack = AsyncExitStack()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Fetch the server list from the registry and prepare the HTTP client."""
        headers: dict[str, str] = {}
        if self._registry_settings.api_key:
            headers["Authorization"] = f"Bearer {self._registry_settings.api_key}"
        self._http = httpx.AsyncClient(
            base_url=self._registry_settings.base_url,
            timeout=self._registry_settings.timeout_seconds,
            headers=headers,
        )
        await self.refresh_servers()

    async def refresh_servers(self) -> None:
        """Re-fetch the server list from the registry."""
        if self._http is None:
            msg = "MCPClientFactory has not been initialized"
            raise MCPClientError(msg)
        try:
            resp = await self._http.get(self._mcp_settings.server_list_url)
            resp.raise_for_status()
            data = resp.json()
            items: list[dict[str, Any]] = (
                data if isinstance(data, list) else data.get("servers", data.get("data", []))
            )
            new_servers: dict[str, MCPServerRecord] = {}
            for item in items:
                try:
                    record = MCPServerRecord.model_validate(item)
                    if record.transport != "streamable_http":
                        logger.debug(
                            "Skipping server %s: unsupported transport %s",
                            record.server_id,
                            record.transport,
                        )
                        continue
                    new_servers[record.server_id] = record
                except Exception:
                    logger.warning("Skipping invalid server record: %s", item, exc_info=True)
            # Disconnect clients for servers no longer in the list
            removed = set(self._servers) - set(new_servers)
            for sid in removed:
                await self._disconnect_client(sid)
            self._servers = new_servers
            logger.info("Loaded %d MCP server(s) from registry", len(self._servers))
        except httpx.HTTPStatusError:
            logger.exception("Failed to fetch MCP server list (HTTP error)")
            raise
        except Exception:
            logger.exception("Failed to fetch MCP server list")
            raise

    async def close(self) -> None:
        """Disconnect all clients and release resources."""
        for sid in list(self._clients):
            await self._disconnect_client(sid)
        await self._exit_stack.aclose()
        if self._http:
            await self._http.aclose()
            self._http = None
        await self._credentials.close()
        self._servers.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_servers(self) -> list[MCPServerRecord]:
        """Return the current list of known servers."""
        return list(self._servers.values())

    async def list_tools(self, server_id: str) -> list[Tool]:
        """List all tools available on a specific server."""
        client = await self._get_or_connect(server_id)
        try:
            return await client.list_tools()
        except Exception as exc:
            logger.exception("Failed to list tools for server %s", server_id)
            await self._disconnect_client(server_id)
            msg = f"Failed to list tools for server {server_id}: {exc}"
            raise MCPClientError(msg) from exc

    async def list_all_tools(self) -> dict[str, list[Tool]]:
        """List tools from every registered server.

        Returns a mapping of ``server_id`` -> tool list.
        """
        result: dict[str, list[Tool]] = {}
        for sid in list(self._servers):
            try:
                result[sid] = await self.list_tools(sid)
            except MCPClientError:
                logger.warning("Skipping server %s due to connection error", sid)
                result[sid] = []
        return result

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Execute a tool on a remote MCP server.

        Args:
            server_id: The target server identifier.
            tool_name: Name of the tool to invoke.
            arguments: Tool arguments as a JSON-serialisable dict.

        Returns:
            The ``CallToolResult`` from the remote server.

        Raises:
            MCPClientError: On validation failure, connection, or execution errors.
        """
        self._validate_call_input(server_id, tool_name, arguments)
        client = await self._get_or_connect(server_id)
        try:
            result = await client.call_tool(
                tool_name,
                arguments or {},
                timeout=self._mcp_settings.call_timeout_seconds,
            )
            self._validate_call_output(result)
            return result
        except MCPClientError:
            raise
        except Exception as exc:
            logger.exception(
                "Tool call failed: server=%s tool=%s", server_id, tool_name
            )
            await self._disconnect_client(server_id)
            msg = f"Tool call failed on server {server_id}: {exc}"
            raise MCPClientError(msg) from exc

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _get_or_connect(self, server_id: str) -> Client:
        """Return an existing client or create a new connection."""
        client = self._clients.get(server_id)
        if client and client.is_connected():
            return client

        server = self._servers.get(server_id)
        if server is None:
            msg = f"Unknown server: {server_id}"
            raise MCPClientError(msg)

        return await self._connect(server)

    async def _connect(self, server: MCPServerRecord) -> Client:
        """Establish a new connection to a remote MCP server."""
        headers = await self._build_headers(server)
        transport = StreamableHttpTransport(
            url=server.url,
            headers=headers,
        )
        client = Client(
            transport=transport,
            timeout=self._mcp_settings.connect_timeout_seconds,
        )
        try:
            ctx = await self._exit_stack.enter_async_context(client)
            self._clients[server.server_id] = ctx
            logger.info("Connected to MCP server %s at %s", server.server_id, server.url)
            return ctx
        except Exception as exc:
            logger.exception("Failed to connect to server %s at %s", server.server_id, server.url)
            msg = f"Connection to server {server.server_id} failed: {exc}"
            raise MCPClientError(msg) from exc

    async def _disconnect_client(self, server_id: str) -> None:
        """Gracefully disconnect a client if it exists."""
        client = self._clients.pop(server_id, None)
        if client is not None:
            try:
                if client.is_connected():
                    await client.close()
            except Exception:
                logger.debug("Error closing client for server %s", server_id, exc_info=True)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def _build_headers(self, server: MCPServerRecord) -> dict[str, str]:
        """Build request headers including authentication for a server."""
        headers: dict[str, str] = {}

        if not server.auth_type:
            return headers

        token = await self._credentials.resolve(server)
        if not token:
            logger.warning("No credential resolved for server %s", server.server_id)
            return headers

        auth_type = server.auth_type.lower()
        if auth_type == "bearer_token":
            headers["Authorization"] = f"Bearer {token}"
        elif auth_type == "authorization":
            # Use the static_headers pattern: "Token <value>" or custom
            auth_header = server.static_headers.get("Authorization", "")
            if auth_header:
                # Replace the variable placeholder with the resolved token
                parts = auth_header.split()
                if len(parts) == 2:
                    headers["Authorization"] = f"{parts[0]} {token}"
                else:
                    headers["Authorization"] = token
            else:
                headers["Authorization"] = f"Bearer {token}"
        elif auth_type == "x-api-key":
            headers["x-api-key"] = token
        else:
            logger.warning("Unsupported auth_type '%s' for server %s", auth_type, server.server_id)

        return headers

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_call_input(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any] | None,
    ) -> None:
        """Basic input validation before proxying a tool call."""
        if not server_id or not isinstance(server_id, str):
            msg = "server_id must be a non-empty string"
            raise MCPClientError(msg)
        if not tool_name or not isinstance(tool_name, str):
            msg = "tool_name must be a non-empty string"
            raise MCPClientError(msg)
        if arguments is not None:
            if not isinstance(arguments, dict):
                msg = "arguments must be a dict"
                raise MCPClientError(msg)
            for key, val in arguments.items():
                if isinstance(val, str) and len(val) > _MAX_ARG_VALUE_LEN:
                    msg = f"Argument '{key}' exceeds maximum length ({_MAX_ARG_VALUE_LEN} bytes)"
                    raise MCPClientError(msg)

    def _validate_call_output(self, result: CallToolResult) -> None:
        """Validate the response from a remote MCP server."""
        if result is None:
            msg = "Received null result from MCP server"
            raise MCPClientError(msg)
        for content in result.content:
            text = getattr(content, "text", None)
            if isinstance(text, str) and len(text) > _MAX_RESPONSE_LEN:
                logger.warning(
                    "Response content exceeds %d chars; it will be truncated downstream",
                    _MAX_RESPONSE_LEN,
                )
