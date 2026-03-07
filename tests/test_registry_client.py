"""Tests for global hash computation (moved from the removed RegistryClient)."""

from __future__ import annotations

from mcp_tool_router.models.schemas import ToolRecord
from mcp_tool_router.sync.worker import compute_global_hash


class TestGlobalHash:
    def test_deterministic(self) -> None:
        tools = [
            ToolRecord(name="a", description="d1"),
            ToolRecord(name="b", description="d2"),
        ]
        h1 = compute_global_hash(tools)
        h2 = compute_global_hash(tools)
        assert h1 == h2

    def test_order_independent(self) -> None:
        t1 = ToolRecord(name="a", description="d1")
        t2 = ToolRecord(name="b", description="d2")
        h1 = compute_global_hash([t1, t2])
        h2 = compute_global_hash([t2, t1])
        assert h1 == h2

    def test_changes_on_diff(self) -> None:
        t1 = [ToolRecord(name="a", description="d1")]
        t2 = [ToolRecord(name="a", description="d2")]
        assert compute_global_hash(t1) != compute_global_hash(t2)
