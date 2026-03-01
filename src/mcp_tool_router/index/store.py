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
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import struct
from typing import Any

import numpy as np

from mcp_tool_router.models.schemas import IndexedServer, IndexedTool, SearchResult
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
# ToolIndex
# ---------------------------------------------------------------------------


class ToolIndex:
    """Manages the local SQLite tool index (metadata + vectors + FTS)."""

    def __init__(self, settings: IndexSettings) -> None:
        self._settings = settings
        self._conn: sqlite3.Connection | None = None
        self._vec_available = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        await asyncio.to_thread(self._init_sync)

    def _init_sync(self) -> None:
        import pathlib

        pathlib.Path(self._settings.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            self._settings.db_path,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")

        # Attempt to load sqlite-vec
        if self._settings.use_vec_extension:
            try:
                import sqlite_vec

                self._conn.enable_load_extension(True)
                sqlite_vec.load(self._conn)
                self._conn.enable_load_extension(False)
                self._vec_available = True
                logger.info("sqlite-vec extension loaded")
            except Exception:
                logger.warning("sqlite-vec unavailable - using Python cosine fallback")

        self._create_tables()

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS servers (
                server_id       TEXT PRIMARY KEY,
                server_name     TEXT NOT NULL,
                alias           TEXT,
                description     TEXT NOT NULL DEFAULT '',
                content_hash    TEXT NOT NULL DEFAULT '',
                embedding       BLOB
            );

            CREATE TABLE IF NOT EXISTS tools (
                name            TEXT PRIMARY KEY,
                description     TEXT NOT NULL,
                input_schema    TEXT NOT NULL DEFAULT '{}',
                output_schema   TEXT,
                tags            TEXT NOT NULL DEFAULT '[]',
                content_hash    TEXT NOT NULL DEFAULT '',
                synthetic_questions TEXT NOT NULL DEFAULT '[]',
                embedding       BLOB,
                server_id       TEXT,
                server_description TEXT NOT NULL DEFAULT ''
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS tools_fts USING fts5(
                name, description, tags, synthetic_questions,
                content='tools', content_rowid='rowid'
            );
            """
        )
        # Migrate existing tools tables that lack the new columns
        self._migrate_tools_table()

        if self._vec_available:
            dim = self._settings.dimension
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS tools_vec "
                f"USING vec0(embedding float[{dim}])"
            )
        self._conn.commit()

    def _migrate_tools_table(self) -> None:
        """Add server_id and server_description columns if missing (idempotent)."""
        assert self._conn is not None
        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(tools)").fetchall()
        }
        if "server_id" not in existing:
            self._conn.execute("ALTER TABLE tools ADD COLUMN server_id TEXT")
        if "server_description" not in existing:
            self._conn.execute(
                "ALTER TABLE tools ADD COLUMN server_description TEXT NOT NULL DEFAULT ''"
            )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def dimension(self) -> int:
        return self._settings.dimension

    @property
    def vec_available(self) -> bool:
        return self._vec_available

    # ------------------------------------------------------------------
    # Server write operations
    # ------------------------------------------------------------------

    async def upsert_server(self, server: IndexedServer) -> None:
        await asyncio.to_thread(self._upsert_server_sync, server)

    def _upsert_server_sync(self, server: IndexedServer) -> None:
        assert self._conn is not None
        emb_blob = _serialize_f32(server.embedding) if server.embedding else None
        self._conn.execute(
            """INSERT INTO servers(server_id, server_name, alias, description,
               content_hash, embedding)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(server_id) DO UPDATE SET
               server_name=excluded.server_name, alias=excluded.alias,
               description=excluded.description, content_hash=excluded.content_hash,
               embedding=excluded.embedding""",
            (
                server.server_id,
                server.server_name,
                server.alias,
                server.description,
                server.content_hash,
                emb_blob,
            ),
        )
        self._conn.commit()

    async def delete_servers(self, server_ids: list[str]) -> None:
        await asyncio.to_thread(self._delete_servers_sync, server_ids)

    def _delete_servers_sync(self, server_ids: list[str]) -> None:
        assert self._conn is not None
        for sid in server_ids:
            self._conn.execute("DELETE FROM servers WHERE server_id = ?", (sid,))
        self._conn.commit()

    async def get_server_hashes(self) -> dict[str, str]:
        return await asyncio.to_thread(self._server_hashes_sync)

    def _server_hashes_sync(self) -> dict[str, str]:
        assert self._conn is not None
        rows = self._conn.execute("SELECT server_id, content_hash FROM servers").fetchall()
        return {r["server_id"]: r["content_hash"] for r in rows}

    async def get_server(self, server_id: str) -> IndexedServer | None:
        return await asyncio.to_thread(self._get_server_sync, server_id)

    def _get_server_sync(self, server_id: str) -> IndexedServer | None:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM servers WHERE server_id = ?", (server_id,)
        ).fetchone()
        if not row:
            return None
        return IndexedServer(
            server_id=row["server_id"],
            server_name=row["server_name"],
            alias=row["alias"],
            description=row["description"],
            content_hash=row["content_hash"],
            embedding=_deserialize_f32(row["embedding"]).tolist() if row["embedding"] else None,
        )

    # ------------------------------------------------------------------
    # Tool write operations
    # ------------------------------------------------------------------

    async def upsert_tool(self, tool: IndexedTool) -> None:
        await asyncio.to_thread(self._upsert_sync, tool)

    def _upsert_sync(self, tool: IndexedTool) -> None:
        assert self._conn is not None
        emb_blob = _serialize_f32(tool.embedding) if tool.embedding else None

        # Check if existing
        existing = self._conn.execute(
            "SELECT rowid FROM tools WHERE name = ?", (tool.name,)
        ).fetchone()

        if existing:
            rowid = existing["rowid"]
            self._conn.execute(
                """UPDATE tools SET description=?, input_schema=?, output_schema=?,
                   tags=?, content_hash=?, synthetic_questions=?, embedding=?,
                   server_id=?, server_description=?
                   WHERE name=?""",
                (
                    tool.description,
                    json.dumps(tool.input_schema),
                    json.dumps(tool.output_schema) if tool.output_schema else None,
                    json.dumps(tool.tags),
                    tool.content_hash,
                    json.dumps(tool.synthetic_questions),
                    emb_blob,
                    tool.server_id,
                    tool.server_description,
                    tool.name,
                ),
            )
            # Update FTS
            self._conn.execute(
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
            self._conn.execute(
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
                self._conn.execute("DELETE FROM tools_vec WHERE rowid = ?", (rowid,))
                self._conn.execute(
                    "INSERT INTO tools_vec(rowid, embedding) VALUES(?, ?)",
                    (rowid, emb_blob),
                )
        else:
            cur = self._conn.execute(
                """INSERT INTO tools(name, description, input_schema, output_schema,
                   tags, content_hash, synthetic_questions, embedding,
                   server_id, server_description)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    tool.name,
                    tool.description,
                    json.dumps(tool.input_schema),
                    json.dumps(tool.output_schema) if tool.output_schema else None,
                    json.dumps(tool.tags),
                    tool.content_hash,
                    json.dumps(tool.synthetic_questions),
                    emb_blob,
                    tool.server_id,
                    tool.server_description,
                ),
            )
            rowid = cur.lastrowid
            # FTS
            self._conn.execute(
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
                self._conn.execute(
                    "INSERT INTO tools_vec(rowid, embedding) VALUES(?, ?)",
                    (rowid, emb_blob),
                )

        self._conn.commit()

    async def delete_tools(self, names: list[str]) -> None:
        await asyncio.to_thread(self._delete_sync, names)

    def _delete_sync(self, names: list[str]) -> None:
        assert self._conn is not None
        for name in names:
            row = self._conn.execute("SELECT rowid FROM tools WHERE name = ?", (name,)).fetchone()
            if not row:
                continue
            rowid = row["rowid"]
            existing = self._conn.execute(
                "SELECT name, description, tags, synthetic_questions FROM tools WHERE rowid=?",
                (rowid,),
            ).fetchone()
            if existing:
                self._conn.execute(
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
                self._conn.execute("DELETE FROM tools_vec WHERE rowid = ?", (rowid,))
            self._conn.execute("DELETE FROM tools WHERE name = ?", (name,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_content_hashes(self) -> dict[str, str]:
        return await asyncio.to_thread(self._hashes_sync)

    def _hashes_sync(self) -> dict[str, str]:
        assert self._conn is not None
        rows = self._conn.execute("SELECT name, content_hash FROM tools").fetchall()
        return {r["name"]: r["content_hash"] for r in rows}

    async def get_tool(self, name: str) -> IndexedTool | None:
        return await asyncio.to_thread(self._get_tool_sync, name)

    def _get_tool_sync(self, name: str) -> IndexedTool | None:
        assert self._conn is not None
        row = self._conn.execute("SELECT * FROM tools WHERE name = ?", (name,)).fetchone()
        if not row:
            return None
        return self._row_to_indexed_tool(row)

    async def tool_count(self) -> int:
        return await asyncio.to_thread(self._count_sync)

    def _count_sync(self) -> int:
        assert self._conn is not None
        row = self._conn.execute("SELECT COUNT(*) AS c FROM tools").fetchone()
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
        return await asyncio.to_thread(
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
        fetch_k = top_k * 3
        vec_ranked = self._vector_search(query_embedding, fetch_k)
        fts_ranked = self._fts_search(query_text, fetch_k)
        combined = _rrf_combine(vec_ranked, fts_ranked, alpha=alpha)

        # Tag filter
        if tags:
            tag_set = set(tags)
            combined = [(n, s) for n, s in combined if self._tool_has_tags(n, tag_set)]

        # Min score filter + top-k
        combined = [(n, s) for n, s in combined if s >= min_score][:top_k]

        return [self._to_search_result(n, s) for n, s in combined]

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def _vector_search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self._vec_available:
            return self._vec_ext_search(query_embedding, top_k)
        return self._python_cosine_search(query_embedding, top_k)

    def _vec_ext_search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """KNN via sqlite-vec ``vec0`` virtual table."""
        assert self._conn is not None
        blob = _serialize_f32(query_embedding)
        rows = self._conn.execute(
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
        self, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[str, float]]:
        """Chunked NumPy cosine similarity – CPU / memory friendly."""
        assert self._conn is not None
        chunk = self._settings.similarity_chunk_size
        q_norm = np.linalg.norm(query_embedding)
        q_unit = query_embedding / q_norm if q_norm > 0 else query_embedding

        all_scores: list[tuple[str, float]] = []
        offset = 0

        while True:
            rows = self._conn.execute(
                "SELECT name, embedding FROM tools WHERE embedding IS NOT NULL " "LIMIT ? OFFSET ?",
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

    def _fts_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        assert self._conn is not None
        safe_query = _sanitise_fts(query)
        if not safe_query:
            return []
        try:
            rows = self._conn.execute(
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

    def _tool_has_tags(self, name: str, tag_set: set[str]) -> bool:
        assert self._conn is not None
        row = self._conn.execute("SELECT tags FROM tools WHERE name = ?", (name,)).fetchone()
        if not row:
            return False
        tool_tags: list[str] = json.loads(row["tags"])
        return bool(tag_set & set(tool_tags))

    def _to_search_result(self, name: str, score: float) -> SearchResult:
        assert self._conn is not None
        row = self._conn.execute(
            """SELECT t.name, t.description, t.input_schema, t.output_schema, t.tags,
                      s.server_name, s.description AS srv_description
               FROM tools t
               LEFT JOIN servers s ON t.server_id = s.server_id
               WHERE t.name=?""",
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
            server_name=row["server_name"],
            server_description=row["srv_description"] if row["srv_description"] else None,
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
            server_id=row["server_id"],
            server_description=row["server_description"] or "",
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
