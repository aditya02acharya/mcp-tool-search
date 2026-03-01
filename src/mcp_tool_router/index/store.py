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
* **Hybrid ranking** uses weighted Reciprocal Rank Fusion (RRF) so we never
  need to normalise heterogeneous score distributions.
* **Connection pool** – a bounded pool of SQLite connections allows concurrent
  reads under WAL mode.  Writes are serialised via a dedicated lock.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import sqlite3
import struct
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

import numpy as np

from mcp_tool_router.models.schemas import IndexedTool, SearchResult
from mcp_tool_router.settings import IndexSettings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _serialize_f32(vec: np.ndarray | list[float]) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    return struct.pack(f"{len(arr)}f", *arr)


def _deserialize_f32(blob: bytes) -> np.ndarray:
    count = len(blob) // 4
    return np.array(struct.unpack(f"{count}f", blob), dtype=np.float32)


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------


class _ConnectionPool:
    """Lazy, bounded pool of SQLite connections for concurrent WAL readers."""

    def __init__(
        self,
        db_path: str,
        *,
        max_size: int = 32,
    ) -> None:
        self._db_path = db_path
        self._max_size = max_size
        self._idle: queue.SimpleQueue[sqlite3.Connection] = queue.SimpleQueue()
        self._created = 0
        self._created_lock = threading.Lock()
        self.vec_available = False

    def _new_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        if self.vec_available:
            try:
                import sqlite_vec

                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
            except Exception:
                logger.warning("Failed to load sqlite-vec on pooled connection")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Check out a connection, return it when done."""
        conn = self._acquire()
        try:
            yield conn
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            self._idle.put(conn)

    def _acquire(self) -> sqlite3.Connection:
        try:
            return self._idle.get_nowait()
        except queue.Empty:
            with self._created_lock:
                if self._created < self._max_size:
                    self._created += 1
                    return self._new_conn()
            # Pool full – block until one is returned
            return self._idle.get()

    def close_all(self) -> None:
        while True:
            try:
                self._idle.get_nowait().close()
            except queue.Empty:
                break


# ---------------------------------------------------------------------------
# ToolIndex
# ---------------------------------------------------------------------------


class ToolIndex:
    """Manages the local SQLite tool index (metadata + vectors + FTS)."""

    def __init__(self, settings: IndexSettings) -> None:
        self._settings = settings
        self._pool: _ConnectionPool | None = None
        self._write_lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        await asyncio.to_thread(self._init_sync)

    def _init_sync(self) -> None:
        import pathlib

        pathlib.Path(self._settings.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._pool = _ConnectionPool(
            self._settings.db_path,
            max_size=self._settings.pool_size,
        )

        # Bootstrap: detect vec capability and create tables on the first connection
        with self._pool.connection() as conn:
            if self._settings.use_vec_extension:
                try:
                    import sqlite_vec

                    conn.enable_load_extension(True)
                    sqlite_vec.load(conn)
                    conn.enable_load_extension(False)
                    self._pool.vec_available = True
                    logger.info("sqlite-vec extension loaded")
                except Exception:
                    logger.warning("sqlite-vec unavailable - using Python cosine fallback")

            self._create_tables(conn)

        self._executor = ThreadPoolExecutor(max_workers=self._settings.pool_size)

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
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
        if self.vec_available:
            dim = self._settings.dimension
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS tools_vec "
                f"USING vec0(embedding float[{dim}])"
            )
        conn.commit()

    def close(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._pool:
            self._pool.close_all()
            self._pool = None

    @property
    def dimension(self) -> int:
        return self._settings.dimension

    @property
    def vec_available(self) -> bool:
        return self._pool.vec_available if self._pool else False

    # ------------------------------------------------------------------
    # Thread dispatch helper
    # ------------------------------------------------------------------

    async def _in_thread(self, fn: Any, *args: Any) -> Any:
        """Run *fn* on the dedicated thread-pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def upsert_tool(self, tool: IndexedTool) -> None:
        await self._in_thread(self._upsert_sync, tool)

    def _upsert_sync(self, tool: IndexedTool) -> None:
        assert self._pool is not None
        emb_blob = _serialize_f32(tool.embedding) if tool.embedding else None

        with self._write_lock, self._pool.connection() as conn:
            # Check if existing
            existing = conn.execute(
                "SELECT rowid FROM tools WHERE name = ?", (tool.name,)
            ).fetchone()

            if existing:
                rowid = existing["rowid"]
                conn.execute(
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
                conn.execute(
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
                conn.execute(
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
                if self.vec_available and emb_blob:
                    conn.execute("DELETE FROM tools_vec WHERE rowid = ?", (rowid,))
                    conn.execute(
                        "INSERT INTO tools_vec(rowid, embedding) VALUES(?, ?)",
                        (rowid, emb_blob),
                    )
            else:
                cur = conn.execute(
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
                rowid = cur.lastrowid
                # FTS
                conn.execute(
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
                if self.vec_available and emb_blob:
                    conn.execute(
                        "INSERT INTO tools_vec(rowid, embedding) VALUES(?, ?)",
                        (rowid, emb_blob),
                    )

            conn.commit()

    async def delete_tools(self, names: list[str]) -> None:
        await self._in_thread(self._delete_sync, names)

    def _delete_sync(self, names: list[str]) -> None:
        assert self._pool is not None
        with self._write_lock, self._pool.connection() as conn:
            for name in names:
                row = conn.execute(
                    "SELECT rowid FROM tools WHERE name = ?", (name,)
                ).fetchone()
                if not row:
                    continue
                rowid = row["rowid"]
                existing = conn.execute(
                    "SELECT name, description, tags, synthetic_questions FROM tools WHERE rowid=?",
                    (rowid,),
                ).fetchone()
                if existing:
                    conn.execute(
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
                if self.vec_available:
                    conn.execute("DELETE FROM tools_vec WHERE rowid = ?", (rowid,))
                conn.execute("DELETE FROM tools WHERE name = ?", (name,))
            conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_content_hashes(self) -> dict[str, str]:
        return await self._in_thread(self._hashes_sync)

    def _hashes_sync(self) -> dict[str, str]:
        assert self._pool is not None
        with self._pool.connection() as conn:
            rows = conn.execute("SELECT name, content_hash FROM tools").fetchall()
        return {r["name"]: r["content_hash"] for r in rows}

    async def get_tool(self, name: str) -> IndexedTool | None:
        return await self._in_thread(self._get_tool_sync, name)

    def _get_tool_sync(self, name: str) -> IndexedTool | None:
        assert self._pool is not None
        with self._pool.connection() as conn:
            row = conn.execute("SELECT * FROM tools WHERE name = ?", (name,)).fetchone()
        if not row:
            return None
        return self._row_to_indexed_tool(row)

    async def tool_count(self) -> int:
        return await self._in_thread(self._count_sync)

    def _count_sync(self) -> int:
        assert self._pool is not None
        with self._pool.connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM tools").fetchone()
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
        return await self._in_thread(
            self._hybrid_sync, query_text, query_embedding, top_k, alpha, tags, min_score
        )

    def _hybrid_sync(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int,
        alpha: float,
        tags: list[str] | None,
        min_score: float,
    ) -> list[SearchResult]:
        assert self._pool is not None
        with self._pool.connection() as conn:
            fetch_k = top_k * 3
            vec_ranked = self._vector_search(conn, query_embedding, fetch_k)
            fts_ranked = self._fts_search(conn, query_text, fetch_k)
            combined = _rrf_combine(vec_ranked, fts_ranked, alpha=alpha)

            # Tag filter
            if tags:
                tag_set = set(tags)
                combined = [(n, s) for n, s in combined if self._tool_has_tags(conn, n, tag_set)]

            # Min score filter + top-k
            combined = [(n, s) for n, s in combined if s >= min_score][:top_k]

            return [self._to_search_result(conn, n, s) for n, s in combined]

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def _vector_search(
        self, conn: sqlite3.Connection, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[str, float]]:
        if self.vec_available:
            return self._vec_ext_search(conn, query_embedding, top_k)
        return self._python_cosine_search(conn, query_embedding, top_k)

    def _vec_ext_search(
        self, conn: sqlite3.Connection, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[str, float]]:
        """KNN via sqlite-vec ``vec0`` virtual table."""
        blob = _serialize_f32(query_embedding)
        rows = conn.execute(
            """SELECT v.rowid, v.distance, t.name
               FROM tools_vec v
               JOIN tools t ON t.rowid = v.rowid
               WHERE v.embedding MATCH ? AND k = ?
               ORDER BY v.distance""",
            (blob, top_k),
        ).fetchall()
        # distance → similarity (cosine distance ∈ [0,2], similarity = 1 - dist/2)
        return [(r["name"], max(0.0, 1.0 - r["distance"] / 2.0)) for r in rows]

    def _python_cosine_search(
        self, conn: sqlite3.Connection, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[str, float]]:
        """Chunked NumPy cosine similarity – CPU / memory friendly."""
        chunk = self._settings.similarity_chunk_size
        q_norm = np.linalg.norm(query_embedding)
        q_unit = query_embedding / q_norm if q_norm > 0 else query_embedding

        all_scores: list[tuple[str, float]] = []
        offset = 0

        while True:
            rows = conn.execute(
                "SELECT name, embedding FROM tools WHERE embedding IS NOT NULL "
                "LIMIT ? OFFSET ?",
                (chunk, offset),
            ).fetchall()
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

    def _fts_search(
        self, conn: sqlite3.Connection, query: str, top_k: int
    ) -> list[tuple[str, float]]:
        safe_query = _sanitise_fts(query)
        if not safe_query:
            return []
        try:
            rows = conn.execute(
                """SELECT t.name, -tools_fts.rank AS score
                   FROM tools_fts
                   JOIN tools t ON t.rowid = tools_fts.rowid
                   WHERE tools_fts MATCH ?
                   ORDER BY tools_fts.rank
                   LIMIT ?""",
                (safe_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r["name"], float(r["score"])) for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tool_has_tags(self, conn: sqlite3.Connection, name: str, tag_set: set[str]) -> bool:
        row = conn.execute("SELECT tags FROM tools WHERE name = ?", (name,)).fetchone()
        if not row:
            return False
        tool_tags: list[str] = json.loads(row["tags"])
        return bool(tag_set & set(tool_tags))

    def _to_search_result(self, conn: sqlite3.Connection, name: str, score: float) -> SearchResult:
        row = conn.execute(
            "SELECT name, description, input_schema, output_schema, tags FROM tools WHERE name=?",
            (name,),
        ).fetchone()
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


def _rrf_combine(
    vec_results: list[tuple[str, float]],
    fts_results: list[tuple[str, float]],
    *,
    k: int = 60,
    alpha: float = 0.7,
) -> list[tuple[str, float]]:
    """Weighted Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for rank, (name, _) in enumerate(vec_results):
        scores[name] = scores.get(name, 0.0) + alpha / (k + rank + 1)
    for rank, (name, _) in enumerate(fts_results):
        scores[name] = scores.get(name, 0.0) + (1.0 - alpha) / (k + rank + 1)
    return sorted(scores.items(), key=lambda t: t[1], reverse=True)


def _sanitise_fts(query: str) -> str:
    """Strip FTS5 special chars to prevent syntax errors."""
    return " ".join(
        filter(
            None,
            (tok.strip("\"'*():-+") for tok in query.split()),
        )
    )
