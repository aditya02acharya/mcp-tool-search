"""LLM-based context compression via LiteLLM.

Summarises accumulated tool-call results, identifies information gaps,
and returns citations keyed by ``call_id`` so the agent can request
verbatim content when needed.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import litellm

from mcp_tool_router.models.schemas import Citation, ContextEntry, ContextSummary
from mcp_tool_router.settings import LLMSettings

logger = logging.getLogger(__name__)

_MAX_ENTRY_CHARS = 2000  # truncate individual entries for the LLM prompt


class ContextCompressor:
    """Compress / summarise session context using an LLM proxy."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings

    async def summarise(
        self,
        entries: list[ContextEntry],
        query: str,
    ) -> ContextSummary:
        """Summarise *entries* relevant to *query* and cite sources."""
        if not entries:
            return ContextSummary(summary="No context accumulated yet.", gaps=[query])

        context_block = "\n---\n".join(map(_format_entry, entries))

        prompt = _build_prompt(query, context_block)

        response = await litellm.acompletion(
            model=self._settings.model_name,
            messages=[{"role": "user", "content": prompt}],
            api_base=self._settings.api_base,
            api_key=self._settings.api_key,
            max_tokens=self._settings.max_tokens,
            temperature=self._settings.temperature,
            response_format={"type": "json_object"},
        )
        raw: str = response.choices[0].message.content
        parsed = _parse_response(raw)

        return ContextSummary(
            summary=parsed.get("summary", ""),
            gaps=parsed.get("gaps", []),
            citations=[Citation(**c) for c in parsed.get("citations", [])],
            total_entries=len(entries),
            total_size_bytes=sum(e.content_size_bytes for e in entries),
        )

    async def quick_summarise(self, entry: ContextEntry) -> dict[str, Any]:
        """Fast single-entry summary returned immediately after execute_tool."""
        result_text = json.dumps(entry.result, default=str)[:_MAX_ENTRY_CHARS]

        prompt = (
            f"Summarise this tool result concisely. Identify any information gaps.\n\n"
            f"Tool: {entry.tool_name}\nArguments: {json.dumps(entry.arguments)}\n"
            f"Result:\n{result_text}\n\n"
            f'Respond in JSON: {{"summary": "...", "gaps": ["..."]}}'
        )

        response = await litellm.acompletion(
            model=self._settings.model_name,
            messages=[{"role": "user", "content": prompt}],
            api_base=self._settings.api_base,
            api_key=self._settings.api_key,
            max_tokens=512,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw: str = response.choices[0].message.content
        return _parse_response(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_entry(entry: ContextEntry) -> str:
    result_str = json.dumps(entry.result, default=str)[:_MAX_ENTRY_CHARS]
    return (
        f"[{entry.call_id}] Tool: {entry.tool_name}\n"
        f"Args: {json.dumps(entry.arguments)}\n"
        f"Result: {result_str}"
    )


def _build_prompt(query: str, context: str) -> str:
    return (
        "Analyse the following tool-call results and provide:\n"
        "1. A concise summary relevant to this query\n"
        "2. Key information gaps that might need additional tool calls\n"
        "3. For each relevant piece of information, cite the call_id\n\n"
        f"Query: {query}\n\n"
        f"Tool call results:\n{context}\n\n"
        "Respond **only** in JSON:\n"
        '{"summary": "...", "gaps": ["..."], '
        '"citations": [{"call_id": "...", "tool_name": "...", '
        '"excerpt": "...", "relevance_score": 0.0}]}'
    )


def _parse_response(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON: %s", raw[:200])
        return {"summary": raw, "gaps": [], "citations": []}
