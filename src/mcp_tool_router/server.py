"""FastMCP server entrypoint.

Wires together all components via a lifespan context manager and
registers the three MCP tools (search_tools, execute_tool, retrieve_context).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

from mcp_tool_router.context.compressor import ContextCompressor
from mcp_tool_router.context.redis_store import RedisContextStore
from mcp_tool_router.embeddings.client import EmbeddingClient
from mcp_tool_router.index.store import ToolIndex
from mcp_tool_router.registry.client import RegistryClient
from mcp_tool_router.settings import AppSettings
from mcp_tool_router.sync.worker import SyncWorker
from mcp_tool_router.tools.execute_tool import execute_tool
from mcp_tool_router.tools.retrieve_context import retrieve_context
from mcp_tool_router.tools.search_tools import search_tools

logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Initialise shared services and start the background sync worker."""
    settings = AppSettings()
    logging.basicConfig(level=logging.DEBUG if settings.environment == "dev" else logging.INFO)

    embeddings = EmbeddingClient(settings.embedding)
    index = ToolIndex(settings.index)
    await index.initialize()
    redis_store = RedisContextStore(settings.redis)
    await redis_store.connect()
    compressor = ContextCompressor(settings.llm)
    registry = RegistryClient(settings.registry)

    sync = SyncWorker(
        registry=registry,
        index=index,
        embeddings=embeddings,
        registry_settings=settings.registry,
        tdwa_settings=settings.tdwa,
        llm_settings=settings.llm,
    )
    sync_task = asyncio.create_task(sync.start())

    try:
        yield {
            "settings": settings,
            "embeddings": embeddings,
            "index": index,
            "redis_store": redis_store,
            "compressor": compressor,
            "registry": registry,
        }
    finally:
        await sync.stop()
        sync_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sync_task
        await embeddings.close()
        await registry.close()
        await redis_store.close()
        index.close()


def create_server() -> FastMCP:
    """Build and return the configured FastMCP server."""
    mcp = FastMCP("MCP Tool Router", lifespan=app_lifespan)
    mcp.tool(search_tools)
    mcp.tool(execute_tool)
    mcp.tool(retrieve_context)
    return mcp


mcp = create_server()


def main() -> None:
    """CLI entrypoint."""
    mcp.run()


if __name__ == "__main__":
    main()
