"""SQLite vector + FTS5 index with sqlite-vec extension and Python fallback.

Key design choices
------------------
* **sqlite-vec first** – if the C extension is loadable we use ``vec0`` virtual
  tables for KNN queries (SIMD-accelerated cosine distance).
* **Python fallback** – when the extension is unavailable, embeddings are stored
  as BLOBs and cosine similarity is computed with NumPy in configurable chunks
  to cap memory on resource-constrained pods.
* **FTS5** – full-text search on name / description / tags / synthetic
  questions for the keyword leg of hybrid search.
* **Hybrid ranking** uses weighted linear interpolation of max-normalised
  vector and FTS5 scores, producing final scores in [0, 1].
* **aiosqlite** – native async SQLite access with a lightweight bounded
  connection pool.  Writes are serialised via an ``asyncio.Lock``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite
import numpy as np

from mcp_tool_router.models.schemas import IndexedTool, SearchResult
from mcp_tool_router.settings import IndexSettings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _serialize_f32(vec: np.ndarray | list[float]) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _deserialize_f32(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# Async connection pool
# ---------------------------------------------------------------------------


class _AsyncConnectionPool:
    """Bounded pool of aiosqlite connections for concurrent WAL readers."""

    def __init__(
        self,
        factory: Callable[[], Awaitable[aiosqlite.Connection]],
        max_size: int = 5,
    ) -> None:
        self._factory = factory
        self._max_size = max_size
        self._idle: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=max_size)
        self._all: list[aiosqlite.Connection] = []
        self._created = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Check out a connection, return it when done."""
        conn = await self._acquire()
        try:
            yield conn
        except Exception:
            try:
                await conn.rollback()
            except Exception:
                pass
            raise
        finally:
            self._idle.put_nowait(conn)

    async def _acquire(self) -> aiosqlite.Connection:
        # Try to grab an idle connection
        try:
            return self._idle.get_nowait()
        except asyncio.QueueEmpty:
            pass
        # Create a new one if capacity allows
        async with self._lock:
            if self._created < self._max_size:
                self._created += 1
                conn = await self._factory()
                self._all.append(conn)
                return conn
        # Pool full – block until one is returned
        return await self._idle.get()

    async def close(self) -> None:
        """Close all connections (idle and checked-out)."""
        for conn in self._all:
            try:
                await conn.close()
            except Exception:
                pass
        self._all.clear()
        self._created = 0
        # Drain the idle queue
        while not self._idle.empty():
            try:
                self._idle.get_nowait()
            except asyncio.QueueEmpty:
                break


# ---------------------------------------------------------------------------
# ToolIndex
# ---------------------------------------------------------------------------


class ToolIndex:
    """Manages the local SQLite tool index (metadata + vectors + FTS)."""

    def __init__(self, settings: IndexSettings) -> None:
        self._settings = settings
        self._pool: _AsyncConnectionPool | None = None
        self._write_lock = asyncio.Lock()
        self._vec_available = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        import pathlib

        pathlib.Path(self._settings.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Detect vec capability with a probe connection
        if self._settings.use_vec_extension:
            try:
                import sqlite_vec

                probe = await aiosqlite.connect(self._settings.db_path)
                await probe.enable_load_extension(True)
                await probe.load_extension(sqlite_vec.loadable_path())
                await probe.enable_load_extension(False)
                await probe.close()
                self._vec_available = True
                logger.info("sqlite-vec extension loaded")
            except Exception:
                logger.warning("sqlite-vec unavailable - using Python cosine fallback")

        # Build connection factory
        db_path = self._settings.db_path
        vec_available = self._vec_available

        async def _factory() -> aiosqlite.Connection:
            conn = await aiosqlite.connect(db_path)
            conn.row_factory = sqlite3.Row
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=5000")
            if vec_available:
                try:
                    import sqlite_vec

                    await conn.enable_load_extension(True)
                    await conn.load_extension(sqlite_vec.loadable_path())
                    await conn.enable_load_extension(False)
                except Exception:
                    logger.warning("Failed to load sqlite-vec on pooled connection")
            return conn

        self._pool = _AsyncConnectionPool(
            factory=_factory,
            max_size=self._settings.pool_size,
        )

        # Create tables on the first connection
        async with self._pool.connection() as conn:
            await self._create_tables(conn)

    async def _create_tables(self, conn: aiosqlite.Connection) -> None:
        await conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tools (
                name            TEXT PRIMARY KEY,
                description     TEXT NOT NULL,
                input_schema    TEXT NOT NULL DEFAULT '{}',
                output_schema   TEXT,
                tags            TEXT NOT NULL DEFAULT '[]',
                content_hash    TEXT NOT NULL DEFAULT '',
                synthetic_questions TEXT NOT NULL DEFAULT '[]',
                embedding       BLOB
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS tools_fts USING fts5(
                name, description, tags, synthetic_questions,
                content='tools', content_rowid='rowid'
            );
            """
        )
        if self._vec_available:
            dim = self._settings.dimension
            await conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS tools_vec "
                f"USING vec0(embedding float[{dim}])"
            )
        await conn.commit()

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    @property
    def dimension(self) -> int:
        return self._settings.dimension

    @property
    def vec_available(self) -> bool:
        return self._vec_available

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def upsert_tool(self, tool: IndexedTool) -> None:
        assert self._pool is not None
        emb_blob = _serialize_f32(tool.embedding) if tool.embedding else None

        async with self._write_lock, self._pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT rowid FROM tools WHERE name = ?", (tool.name,)
            )
            existing = await cursor.fetchone()

            if existing:
                rowid = existing["rowid"]
                await conn.execute(
                    """UPDATE tools SET description=?, input_schema=?, output_schema=?,
                       tags=?, content_hash=?, synthetic_questions=?, embedding=?
                       WHERE name=?""",
                    (
                        tool.description,
                        json.dumps(tool.input_schema),
                        json.dumps(tool.output_schema) if tool.output_schema else None,
                        json.dumps(tool.tags),
                        tool.content_hash,
                        json.dumps(tool.synthetic_questions),
                        emb_blob,
                        tool.name,
                    ),
                )
                # Update FTS
                await conn.execute(
                    "INSERT INTO tools_fts(tools_fts, rowid, name, description, tags, "
                    "synthetic_questions) VALUES('delete', ?, ?, ?, ?, ?)",
                    (
                        rowid,
                        tool.name,
                        tool.description,
                        " ".join(tool.tags),
                        " ".join(tool.synthetic_questions),
                    ),
                )
                await conn.execute(
                    "INSERT INTO tools_fts(rowid, name, description, tags, synthetic_questions) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (
                        rowid,
                        tool.name,
                        tool.description,
                        " ".join(tool.tags),
                        " ".join(tool.synthetic_questions),
                    ),
                )
                # Update vec
                if self._vec_available and emb_blob:
                    await conn.execute("DELETE FROM tools_vec WHERE rowid = ?", (rowid,))
                    await conn.execute(
                        "INSERT INTO tools_vec(rowid, embedding) VALUES(?, ?)",
                        (rowid, emb_blob),
                    )
            else:
                cursor = await conn.execute(
                    """INSERT INTO tools(name, description, input_schema, output_schema,
                       tags, content_hash, synthetic_questions, embedding)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        tool.name,
                        tool.description,
                        json.dumps(tool.input_schema),
                        json.dumps(tool.output_schema) if tool.output_schema else None,
                        json.dumps(tool.tags),
                        tool.content_hash,
                        json.dumps(tool.synthetic_questions),
                        emb_blob,
                    ),
                )
                rowid = cursor.lastrowid
                # FTS
                await conn.execute(
                    "INSERT INTO tools_fts(rowid, name, description, tags, synthetic_questions) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (
                        rowid,
                        tool.name,
                        tool.description,
                        " ".join(tool.tags),
                        " ".join(tool.synthetic_questions),
                    ),
                )
                # Vec
                if self._vec_available and emb_blob:
                    await conn.execute(
                        "INSERT INTO tools_vec(rowid, embedding) VALUES(?, ?)",
                        (rowid, emb_blob),
                    )

            await conn.commit()

    async def delete_tools(self, names: list[str]) -> None:
        assert self._pool is not None
        async with self._write_lock, self._pool.connection() as conn:
            for name in names:
                cursor = await conn.execute(
                    "SELECT rowid FROM tools WHERE name = ?", (name,)
                )
                row = await cursor.fetchone()
                if not row:
                    continue
                rowid = row["rowid"]
                cursor = await conn.execute(
                    "SELECT name, description, tags, synthetic_questions FROM tools WHERE rowid=?",
                    (rowid,),
                )
                existing = await cursor.fetchone()
                if existing:
                    await conn.execute(
                        "INSERT INTO tools_fts(tools_fts, rowid, name, description, tags, "
                        "synthetic_questions) VALUES('delete', ?, ?, ?, ?, ?)",
                        (
                            rowid,
                            existing["name"],
                            existing["description"],
                            existing["tags"],
                            existing["synthetic_questions"],
                        ),
                    )
                if self._vec_available:
                    await conn.execute("DELETE FROM tools_vec WHERE rowid = ?", (rowid,))
                await conn.execute("DELETE FROM tools WHERE name = ?", (name,))
            await conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_content_hashes(self) -> dict[str, str]:
        assert self._pool is not None
        async with self._pool.connection() as conn:
            cursor = await conn.execute("SELECT name, content_hash FROM tools")
            rows = await cursor.fetchall()
        return {r["name"]: r["content_hash"] for r in rows}

    async def get_tool(self, name: str) -> IndexedTool | None:
        assert self._pool is not None
        async with self._pool.connection() as conn:
            cursor = await conn.execute("SELECT * FROM tools WHERE name = ?", (name,))
            row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_indexed_tool(row)

    async def tool_count(self) -> int:
        assert self._pool is not None
        async with self._pool.connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) AS c FROM tools")
            row = await cursor.fetchone()
        return int(row["c"]) if row else 0

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
        alpha: float = 0.7,
        tags: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        assert self._pool is not None
        async with self._pool.connection() as conn:
            fetch_k = top_k * 3
            vec_ranked = await self._vector_search(conn, query_embedding, fetch_k)
            fts_ranked = await self._fts_search(conn, query_text, fetch_k)
            combined = _weighted_combine(vec_ranked, fts_ranked, alpha=alpha)

            # Tag filter
            if tags:
                tag_set = set(tags)
                filtered: list[tuple[str, float]] = []
                for n, s in combined:
                    if await self._tool_has_tags(conn, n, tag_set):
                        filtered.append((n, s))
                combined = filtered

            # Min score filter + top-k
            combined = [(n, s) for n, s in combined if s >= min_score][:top_k]

            results: list[SearchResult] = []
            for n, s in combined:
                results.append(await self._to_search_result(conn, n, s))
            return results

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    async def _vector_search(
        self, conn: aiosqlite.Connection, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[str, float]]:
        if self._vec_available:
            return await self._vec_ext_search(conn, query_embedding, top_k)
        return await self._python_cosine_search(conn, query_embedding, top_k)

    async def _vec_ext_search(
        self, conn: aiosqlite.Connection, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[str, float]]:
        """KNN via sqlite-vec ``vec0`` virtual table."""
        blob = _serialize_f32(query_embedding)
        cursor = await conn.execute(
            """SELECT v.rowid, v.distance, t.name
               FROM tools_vec v
               JOIN tools t ON t.rowid = v.rowid
               WHERE v.embedding MATCH ? AND k = ?
               ORDER BY v.distance""",
            (blob, top_k),
        )
        rows = await cursor.fetchall()
        # distance → similarity (cosine distance ∈ [0,2], similarity = 1 - dist/2)
        return [(r["name"], max(0.0, 1.0 - r["distance"] / 2.0)) for r in rows]

    async def _python_cosine_search(
        self, conn: aiosqlite.Connection, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[str, float]]:
        """Chunked NumPy cosine similarity – CPU / memory friendly."""
        chunk = self._settings.similarity_chunk_size
        q_norm = np.linalg.norm(query_embedding)
        q_unit = query_embedding / q_norm if q_norm > 0 else query_embedding

        all_scores: list[tuple[str, float]] = []
        offset = 0

        while True:
            cursor = await conn.execute(
                "SELECT name, embedding FROM tools WHERE embedding IS NOT NULL "
                "LIMIT ? OFFSET ?",
                (chunk, offset),
            )
            rows = await cursor.fetchall()
            if not rows:
                break
            names = [r["name"] for r in rows]
            mat = np.array([_deserialize_f32(r["embedding"]) for r in rows], dtype=np.float32)
            # Vectorised normalisation
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            mat /= norms
            sims: np.ndarray = mat @ q_unit
            all_scores.extend(zip(names, sims.tolist(), strict=True))
            offset += chunk

        all_scores.sort(key=lambda t: t[1], reverse=True)
        return all_scores[:top_k]

    # ------------------------------------------------------------------
    # FTS5 search
    # ------------------------------------------------------------------

    async def _fts_search(
        self, conn: aiosqlite.Connection, query: str, top_k: int
    ) -> list[tuple[str, float]]:
        safe_query = _sanitise_fts(query)
        if not safe_query:
            return []
        try:
            cursor = await conn.execute(
                """SELECT t.name, -tools_fts.rank AS score
                   FROM tools_fts
                   JOIN tools t ON t.rowid = tools_fts.rowid
                   WHERE tools_fts MATCH ?
                   ORDER BY tools_fts.rank
                   LIMIT ?""",
                (safe_query, top_k),
            )
            rows = await cursor.fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r["name"], float(r["score"])) for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _tool_has_tags(
        self, conn: aiosqlite.Connection, name: str, tag_set: set[str]
    ) -> bool:
        cursor = await conn.execute("SELECT tags FROM tools WHERE name = ?", (name,))
        row = await cursor.fetchone()
        if not row:
            return False
        tool_tags: list[str] = json.loads(row["tags"])
        return bool(tag_set & set(tool_tags))

    async def _to_search_result(
        self, conn: aiosqlite.Connection, name: str, score: float
    ) -> SearchResult:
        cursor = await conn.execute(
            "SELECT name, description, input_schema, output_schema, tags FROM tools WHERE name=?",
            (name,),
        )
        row = await cursor.fetchone()
        if not row:
            return SearchResult(name=name, description="", score=score)
        return SearchResult(
            name=row["name"],
            description=row["description"],
            input_schema=json.loads(row["input_schema"]),
            output_schema=json.loads(row["output_schema"]) if row["output_schema"] else None,
            tags=json.loads(row["tags"]),
            score=score,
        )

    @staticmethod
    def _row_to_indexed_tool(row: Any) -> IndexedTool:
        return IndexedTool(
            name=row["name"],
            description=row["description"],
            input_schema=json.loads(row["input_schema"]),
            output_schema=json.loads(row["output_schema"]) if row["output_schema"] else None,
            tags=json.loads(row["tags"]),
            content_hash=row["content_hash"],
            embedding=_deserialize_f32(row["embedding"]).tolist() if row["embedding"] else None,
            synthetic_questions=json.loads(row["synthetic_questions"]),
        )


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------


def _normalise_scores(
    results: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Normalise scores to [0, 1] by dividing by the maximum."""
    if not results:
        return results
    max_score = max(s for _, s in results)
    if max_score <= 0:
        return results
    return [(n, s / max_score) for n, s in results]


def _weighted_combine(
    vec_results: list[tuple[str, float]],
    fts_results: list[tuple[str, float]],
    *,
    alpha: float = 0.7,
) -> list[tuple[str, float]]:
    """Weighted linear interpolation of normalised score lists.

    Both inputs are max-normalised to [0, 1] before blending so that
    ``alpha * vec_score + (1 - alpha) * fts_score`` produces a final
    score in [0, 1].
    """
    vec_norm = dict(_normalise_scores(vec_results))
    fts_norm = dict(_normalise_scores(fts_results))
    all_names = set(vec_norm) | set(fts_norm)
    combined = [
        (name, alpha * vec_norm.get(name, 0.0) + (1 - alpha) * fts_norm.get(name, 0.0))
        for name in all_names
    ]
    return sorted(combined, key=lambda t: t[1], reverse=True)


def _sanitise_fts(query: str) -> str:
    """Strip FTS5 special chars to prevent syntax errors."""
    return " ".join(
        filter(
            None,
            (tok.strip("\"'*():-+") for tok in query.split()),
        )
    )
