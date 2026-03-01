"""vLLM REST client for Qwen3 embeddings with TDWA (Tool Document Weighted Average) support.

Qwen3 requires an ``Instruct: ...\\nQuery:`` prefix for *query* embeddings only.
Documents are embedded without the prefix.

TDWA (ScaleMCP paper) independently embeds tool document components
(name, description, params, synthetic questions) and produces a single
L2-normalised weighted-average vector.
"""

from __future__ import annotations

from operator import itemgetter

import httpx
import numpy as np

from mcp_tool_router.settings import EmbeddingSettings, TDWASettings


class EmbeddingClient:
    """Async embedding client targeting a vLLM ``/v1/embeddings`` endpoint."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a *query* with Qwen3 instruction prefix."""
        formatted = f"Instruct: {self._settings.query_instruction}\nQuery:{query}"
        vecs = await self._embed_texts([formatted])
        return np.asarray(vecs[0])

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed plain documents (no instruction prefix)."""
        if not texts:
            return np.empty((0, self._settings.dimension), dtype=np.float32)
        return await self._embed_texts(texts)

    async def embed_tool_tdwa(
        self,
        *,
        name: str,
        description: str,
        params_text: str,
        synthetic_questions: list[str],
        weights: TDWASettings,
        server_description: str = "",
    ) -> np.ndarray:
        """Compute TDWA embedding for a tool document.

        z_TDWA = normalise( Σ wᵢ · Embed(cᵢ) )

        When *server_description* is provided its weight is taken from
        ``weights.server_description_weight``.  When absent the weight is
        redistributed to the description component so the total always
        sums to 1.
        """
        components = [name, description, params_text]
        w = np.array(
            [weights.name_weight, weights.description_weight, weights.params_weight],
            dtype=np.float32,
        )

        if synthetic_questions:
            components.append(" ".join(synthetic_questions))
            w = np.append(w, weights.questions_weight)

        if server_description:
            components.append(server_description)
            w = np.append(w, weights.server_description_weight)

        # Normalise weights so they sum to 1 (handles missing components)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum

        embeddings = await self.embed_documents(components)
        weighted: np.ndarray = np.einsum("i,ij->j", w, embeddings)
        norm: float = float(np.linalg.norm(weighted))
        if norm > 0:
            weighted = weighted / norm
        return weighted

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Batch-embed via vLLM, honouring ``batch_size``."""
        all_embeddings: list[list[float]] = []
        bs = self._settings.batch_size

        for start in range(0, len(texts), bs):
            batch = texts[start : start + bs]
            resp = await self._client.post(
                "/v1/embeddings",
                json={"model": self._settings.model_name, "input": batch},
            )
            resp.raise_for_status()
            items = sorted(resp.json()["data"], key=itemgetter("index"))
            all_embeddings.extend(item["embedding"] for item in items)

        return np.array(all_embeddings, dtype=np.float32)
