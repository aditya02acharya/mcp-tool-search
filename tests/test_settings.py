"""Tests for the settings module – YAML loading, env merging, defaults."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import yaml

from mcp_tool_router.settings import (
    AppSettings,
    EmbeddingSettings,
    IndexSettings,
    LLMSettings,
    RedisSettings,
    RegistrySettings,
    SearchSettings,
    ServerSettings,
    TDWASettings,
    YamlSettingsSource,
    _deep_merge,
    _prefix_alias,
)


class TestDeepMerge:
    def test_flat(self) -> None:
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override(self) -> None:
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested(self) -> None:
        base = {"x": {"a": 1, "b": 2}}
        over = {"x": {"b": 3, "c": 4}}
        assert _deep_merge(base, over) == {"x": {"a": 1, "b": 3, "c": 4}}


class TestPrefixAlias:
    def test_with_prefix(self) -> None:
        gen = _prefix_alias("redis")
        assert gen("host") == "redis_host"

    def test_empty_prefix(self) -> None:
        gen = _prefix_alias("")
        assert gen("host") == "host"


class TestComponentDefaults:
    def test_redis_defaults(self) -> None:
        s = RedisSettings()
        assert s.host == "localhost"
        assert s.port == 6379
        assert s.max_context_bytes == 100 * 1024 * 1024

    def test_embedding_defaults(self) -> None:
        s = EmbeddingSettings()
        assert s.dimension == 4096
        assert "Qwen" in s.model_name

    def test_index_defaults(self) -> None:
        s = IndexSettings()
        assert s.use_vec_extension is True

    def test_llm_defaults(self) -> None:
        s = LLMSettings()
        assert s.temperature == 0.0

    def test_registry_defaults(self) -> None:
        s = RegistrySettings()
        assert s.sync_interval_seconds == 300

    def test_tdwa_weights_sum(self) -> None:
        s = TDWASettings()
        total = s.name_weight + s.description_weight + s.params_weight + s.questions_weight
        assert abs(total - 1.0) < 1e-6

    def test_search_defaults(self) -> None:
        s = SearchSettings()
        assert 0.0 <= s.hybrid_alpha <= 1.0

    def test_server_defaults(self) -> None:
        s = ServerSettings()
        assert s.port == 8080


class TestYamlSettingsSource:
    def test_loads_common_yaml(self, tmp_path: Path) -> None:
        cfg = tmp_path / "common.config.yaml"
        cfg.write_text(yaml.dump({"server": {"port": 9999}}))

        with patch("mcp_tool_router.settings._CONFIG_DIR", tmp_path):
            src = YamlSettingsSource(AppSettings)
            data = src()

        assert data["server"]["port"] == 9999

    def test_env_override(self, tmp_path: Path) -> None:
        (tmp_path / "common.config.yaml").write_text(yaml.dump({"server": {"port": 1111}}))
        (tmp_path / "stage.config.yaml").write_text(yaml.dump({"server": {"port": 2222}}))

        with (
            patch("mcp_tool_router.settings._CONFIG_DIR", tmp_path),
            patch.dict(os.environ, {"ENVIRONMENT": "stage"}),
        ):
            data = YamlSettingsSource(AppSettings)()

        assert data["server"]["port"] == 2222

    def test_missing_files(self, tmp_path: Path) -> None:
        with patch("mcp_tool_router.settings._CONFIG_DIR", tmp_path):
            data = YamlSettingsSource(AppSettings)()
        assert data == {}


class TestAppSettings:
    def test_defaults(self) -> None:
        with patch.dict(os.environ, {"ENVIRONMENT": "dev"}, clear=False):
            s = AppSettings(
                _env_file=None,  # type: ignore[call-arg]
            )
        assert s.environment == "dev"
        assert isinstance(s.redis, RedisSettings)

    def test_env_nested(self) -> None:
        env = {"REDIS__PORT": "1234", "ENVIRONMENT": "dev"}
        with patch.dict(os.environ, env, clear=False):
            s = AppSettings(_env_file=None)  # type: ignore[call-arg]
        assert s.redis.port == 1234
