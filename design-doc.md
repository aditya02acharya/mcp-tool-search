# MCP Tool Router — Design Document

## 1. Overview

**MCP Tool Router** is a FastMCP server that acts as a smart proxy layer between an agent and a large, dynamic tool registry. Its purpose is to reduce agent context window consumption by:

- Exposing only *relevant* tools via semantic/hybrid search instead of loading the full registry.
- Accumulating tool call results in an external Redis store (keyed by session) rather than keeping them in-context.
- Returning LLM-compressed, query-relevant summaries of accumulated context on demand.

A background sync process keeps the local tool index up-to-date with the upstream LiteLLM tool registry.

---


## Project Structure

```
mcp-tool-router/
├── pyproject.toml
├── README.md
├── config/
│   └── settings.py              # Pydantic Settings (env-based config)
│
├── src/
│   └── mcp_tool_router/
│       ├── __init__.py
│       ├── server.py             # FastMCP app entrypoint, tool registration
│       │
│       ├── tools/                # One module per MCP tool
│       │   ├── __init__.py
│       │   ├── search_tools.py   # Tool 1 — semantic/hybrid search
│       │   ├── execute_tool.py   # Tool 2 — proxy call + Redis accumulation
│       │   └── retrieve_context.py  # Tool 3 — context retrieval + LLM compression
│       │
│       ├── embeddings/           # Embedding client
│       │   ├── __init__.py
│       │   └── client.py         # vLLM REST client for Qwen3 embeddings
│       │
│       ├── index/                # Vector index layer
│       │   ├── __init__.py
│       │   └── store.py          # LanceDB read/write/search operations
│       │
│       ├── context/              # Session context management
│       │   ├── __init__.py
│       │   ├── redis_store.py    # Redis operations (accumulate, fetch, expire)
│       │   └── compressor.py     # LLM-based context compression via LiteLLM
│       │
│       ├── registry/             # Upstream registry interaction
│       │   ├── __init__.py
│       │   └── client.py         # Fetch tool list, compute hashes
│       │
│       ├── sync/                 # Background sync worker
│       │   ├── __init__.py
│       │   └── worker.py         # Polling loop, diff, re-index
│       │
│       └── models/               # Shared data models
│           ├── __init__.py
│           └── schemas.py        # Pydantic models (ToolRecord, SessionEntry, etc.)
│
├── tests/
│   ├── conftest.py               # Fixtures (mock Redis, LanceDB, vLLM)
│   ├── test_search_tools.py
│   ├── test_execute_tool.py
│   ├── test_retrieve_context.py
│   ├── test_sync_worker.py
│   └── test_compressor.py
```

---

## 4. Component Design

### 4.1 Tool 1 — `search_tools`

Performs semantic or hybrid search over the tool registry to return a ranked subset of tools relevant to the agent's current intent. Leverages tags and categories as structured filters alongside vector + FTS search.

### 4.2 Tool 2 — `execute_tool`

Proxies a tool call to the upstream registry and accumulates the result in Redis under the caller's session.

Each `execute_tool` call appends a new entry and resets the TTL.

---

### 4.3 Tool 3 — `retrieve_context`

Retrieves accumulated session context and returns an LLM-compressed, query-relevant chunks with metadata. This is the key token-saving mechanism.

---

### 4.4 Background Sync Worker

A long-running async task that polls the upstream tool registry, detects changes via content hashing, and incrementally updates the local index.

**Change Detection** (two-tier hashing):

1. **Global hash** — SHA-256 of the entire sorted, serialised tool list. Quick equality check to short-circuit when nothing changed.
2. **Per-tool hash** — SHA-256 of each tool's `(tool_name, tool_description, tool_schema, tags, categories)` tuple. Used during diff to identify which specific tools were added, removed, or modified — enabling incremental re-embedding.