"""Background sync worker – polls upstream registry, diffs via two-tier
content hashing, and incrementally re-embeds changed tools with TDWA.

Change detection
----------------
1. **Global hash** – SHA-256 of full sorted tool list → quick short-circuit.
2. **Per-tool hash** – SHA-256 of ``(name, description, schema, tags)`` →
   identifies specific adds / removes / modifications for incremental update.
"""

from __future__ import annotations

import asyncio
import json
import logging

import litellm

from mcp_tool_router.embeddings.client import EmbeddingClient
from mcp_tool_router.index.store import ToolIndex
from mcp_tool_router.models.schemas import IndexedTool, ToolRecord
from mcp_tool_router.registry.client import RegistryClient
from mcp_tool_router.settings import LLMSettings, RegistrySettings, TDWASettings

logger = logging.getLogger(__name__)


class SyncWorker:
    """Polls the upstream registry and keeps the local index in sync."""

    def __init__(
        self,
        *,
        registry: RegistryClient,
        index: ToolIndex,
        embeddings: EmbeddingClient,
        registry_settings: RegistrySettings,
        tdwa_settings: TDWASettings,
        llm_settings: LLMSettings,
    ) -> None:
        self._registry = registry
        self._index = index
        self._embeddings = embeddings
        self._reg_settings = registry_settings
        self._tdwa = tdwa_settings
        self._llm = llm_settings
        self._last_global_hash: str | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        while self._running:
            try:
                await self.sync_once()
            except Exception:
                logger.exception("Sync cycle failed")
            await asyncio.sleep(self._reg_settings.sync_interval_seconds)

    async def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Core sync
    # ------------------------------------------------------------------

    async def sync_once(self) -> None:
        """Execute a single sync cycle."""
        tools = await self._registry.list_tools()
        global_hash = RegistryClient.compute_global_hash(tools)

        if global_hash == self._last_global_hash:
            logger.debug("No changes (global hash match)")
            return

        remote_hashes = {t.name: t.content_hash() for t in tools}
        local_hashes = await self._index.get_content_hashes()

        added = set(remote_hashes) - set(local_hashes)
        removed = set(local_hashes) - set(remote_hashes)
        modified = {
            n for n in set(remote_hashes) & set(local_hashes) if remote_hashes[n] != local_hashes[n]
        }

        if not (added or removed or modified):
            self._last_global_hash = global_hash
            return

        logger.info("Sync diff: +%d -%d ~%d", len(added), len(removed), len(modified))

        if removed:
            await self._index.delete_tools(list(removed))

        tools_by_name = {t.name: t for t in tools}
        for name in added | modified:
            tool = tools_by_name[name]
            await self._index_tool(tool)

        self._last_global_hash = global_hash

    # ------------------------------------------------------------------
    # Indexing a single tool
    # ------------------------------------------------------------------

    async def _index_tool(self, tool: ToolRecord) -> None:
        questions = await self._generate_synthetic_questions(tool)
        params_text = json.dumps(tool.input_schema)

        embedding = await self._embeddings.embed_tool_tdwa(
            name=tool.name,
            description=tool.description,
            params_text=params_text,
            synthetic_questions=questions,
            weights=self._tdwa,
        )

        await self._index.upsert_tool(
            IndexedTool(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
                output_schema=tool.output_schema,
                tags=tool.tags,
                content_hash=tool.content_hash(),
                embedding=embedding.tolist(),
                synthetic_questions=questions,
            )
        )

    async def _generate_synthetic_questions(self, tool: ToolRecord) -> list[str]:
        """Use LLM to generate diverse natural-language queries (reverse-HyDE)."""
        n = self._tdwa.num_synthetic_questions
        if n <= 0:
            return []

        prompt = (
            f"Generate exactly {n} diverse natural-language questions that a user "
            f"might ask when they need the following tool.\n\n"
            f"Tool name: {tool.name}\n"
            f"Description: {tool.description}\n"
            f"Parameters: {json.dumps(tool.input_schema)}\n\n"
            f'Return ONLY a JSON array of strings, e.g. ["q1", "q2"].'
        )
        try:
            resp = await litellm.acompletion(
                model=self._llm.model_name,
                messages=[{"role": "user", "content": prompt}],
                api_base=self._llm.api_base,
                api_key=self._llm.api_key,
                max_tokens=512,
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            raw: str = resp.choices[0].message.content
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(q) for q in parsed[:n]]
            if isinstance(parsed, dict) and "questions" in parsed:
                return [str(q) for q in parsed["questions"][:n]]
        except Exception:
            logger.warning("Synthetic question generation failed for %s", tool.name)
        return []
