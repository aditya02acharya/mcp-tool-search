"""Modular Pydantic settings with YAML + env loading and _FIELD_PREFIX alias support."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge *override* into *base*; override values win."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _prefix_alias(prefix: str) -> Callable[[str], str]:
    """Create alias generator that maps ``field`` -> ``{prefix}_{field}``."""

    def _gen(name: str) -> str:
        return f"{prefix}_{name}" if prefix else name

    return _gen


# ---------------------------------------------------------------------------
# YAML settings source
# ---------------------------------------------------------------------------


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Load from ``config/common.config.yaml`` then ``config/{env}.config.yaml``."""

    def get_field_value(
        self,
        field: Any,
        field_name: str,
    ) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        env = os.getenv("ENVIRONMENT", "dev")
        merged: dict[str, Any] = {}
        for name in ("common.config.yaml", f"{env}.config.yaml"):
            path = _CONFIG_DIR / name
            if path.exists():
                with open(path) as fh:
                    data = yaml.safe_load(fh) or {}
                merged = _deep_merge(merged, data)
        return merged


# ---------------------------------------------------------------------------
# Component settings – each has a _FIELD_PREFIX + alias_generator
# ---------------------------------------------------------------------------


class BaseComponentSettings(BaseModel):
    """Base for modular component settings with ``_FIELD_PREFIX`` alias support."""

    _FIELD_PREFIX: ClassVar[str] = ""
    model_config = ConfigDict(populate_by_name=True)


class RedisSettings(BaseComponentSettings):
    _FIELD_PREFIX: ClassVar[str] = "redis"
    model_config = ConfigDict(alias_generator=_prefix_alias("redis"), populate_by_name=True)

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    ssl: bool = False
    session_ttl_seconds: int = 3600
    max_context_bytes: int = 100 * 1024 * 1024  # 100 MB


class EmbeddingSettings(BaseComponentSettings):
    _FIELD_PREFIX: ClassVar[str] = "embedding"
    model_config = ConfigDict(alias_generator=_prefix_alias("embedding"), populate_by_name=True)

    base_url: str = "http://localhost:8000"
    model_name: str = "Qwen/Qwen3-Embedding-8B"
    dimension: int = 4096
    batch_size: int = 32
    query_instruction: str = (
        "Given a tool name and description, retrieve the most relevant tools for the user query"
    )
    timeout_seconds: float = 30.0


class IndexSettings(BaseComponentSettings):
    _FIELD_PREFIX: ClassVar[str] = "index"
    model_config = ConfigDict(alias_generator=_prefix_alias("index"), populate_by_name=True)

    db_path: str = "data/tools.db"
    use_vec_extension: bool = True
    similarity_chunk_size: int = 500
    dimension: int = 4096


class LLMSettings(BaseComponentSettings):
    _FIELD_PREFIX: ClassVar[str] = "llm"
    model_config = ConfigDict(alias_generator=_prefix_alias("llm"), populate_by_name=True)

    model_name: str = "gpt-4o-mini"
    api_base: str | None = None
    api_key: str | None = None
    max_tokens: int = 2048
    temperature: float = 0.0


class RegistrySettings(BaseComponentSettings):
    _FIELD_PREFIX: ClassVar[str] = "registry"
    model_config = ConfigDict(alias_generator=_prefix_alias("registry"), populate_by_name=True)

    base_url: str = "http://localhost:4000"
    api_key: str | None = None
    sync_interval_seconds: int = 300
    timeout_seconds: float = 30.0


class TDWASettings(BaseComponentSettings):
    """Tool Document Weighted Average embedding weights (ScaleMCP paper)."""

    _FIELD_PREFIX: ClassVar[str] = "tdwa"
    model_config = ConfigDict(alias_generator=_prefix_alias("tdwa"), populate_by_name=True)

    name_weight: float = 0.10
    description_weight: float = 0.30
    params_weight: float = 0.20
    questions_weight: float = 0.30
    server_description_weight: float = 0.10
    num_synthetic_questions: int = 5


class SearchSettings(BaseComponentSettings):
    _FIELD_PREFIX: ClassVar[str] = "search"
    model_config = ConfigDict(alias_generator=_prefix_alias("search"), populate_by_name=True)

    top_k: int = 5
    min_score: float = 0.1
    hybrid_alpha: float = 0.7  # vector vs FTS weight


class ServerSettings(BaseComponentSettings):
    _FIELD_PREFIX: ClassVar[str] = "server"
    model_config = ConfigDict(alias_generator=_prefix_alias("server"), populate_by_name=True)

    name: str = "MCP Tool Router"
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8080


# ---------------------------------------------------------------------------
# Root application settings
# ---------------------------------------------------------------------------


class AppSettings(BaseSettings):
    """Root settings – loads from ``.env``, env vars (``__`` nested), and YAML configs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: str = "dev"
    server: ServerSettings = Field(default_factory=ServerSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    registry: RegistrySettings = Field(default_factory=RegistrySettings)
    tdwa: TDWASettings = Field(default_factory=TDWASettings)
    search: SearchSettings = Field(default_factory=SearchSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlSettingsSource(settings_cls),
            file_secret_settings,
        )
