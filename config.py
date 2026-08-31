# config.py
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List, Optional


class LaravelSettings(BaseModel):
    base_url: str = Field(default="http://localhost:8000")
    verify_token_endpoint: str = Field(default="/api/v1/auth/verify-token")
    shared_storage_path: str = Field(default="/app/shared_storage")
    # کلید مشترک Laravel ↔ Python برای ingest/delete (جلوگیری از deadlock)
    internal_api_key: str = Field(
        default="",
        description="Shared secret; env: LARAVEL__INTERNAL_API_KEY",
    )

class AISettings(BaseModel):
    # LLM Settings
    llm_provider: str = Field(default="gapgpt", description="openai, ollama, gapgpt, custom")
    llm_model: str = Field(default="gpt-4o-mini")
    llm_base_url: str = Field(default="https://api.gapgpt.app/v1")
    llm_api_key: str = Field(default="", description="API Key برای LLM")

    # Embedding Settings
    embedding_provider: str = Field(default="gapgpt")
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_base_url: str = Field(default="https://api.gapgpt.app/v1")
    embedding_api_key: str = Field(default="")

    # Vision Settings
    vision_provider: str = Field(default="gapgpt")
    vision_model: str = Field(default="gpt-4o-mini")
    vision_base_url: str = Field(default="https://api.gapgpt.app/v1")
    vision_api_key: str = Field(default="")

    # Generation & Retrieval
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    retrieval_k: int = Field(default=6, ge=1, le=20, description="Number of chunks to retrieve")

    @field_validator(
        "llm_provider",
        "embedding_provider",
        "vision_provider",
        mode="before",
    )
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("provider must be a non-empty string")
        return value.strip().lower()


class DatabaseSettings(BaseModel):
    qdrant_path: str = Field(default="./qdrant_db")
    qdrant_url: str = Field(default="")
    qdrant_collection: str = Field(default="iust_knowledge")
    mysql_dsn: str = Field(default="")


class IngestionSettings(BaseModel):
    """Document ingest / chunking defaults."""

    data_dir: str = Field(default="./data")
    input_dir: str = Field(default="./inputs")
    chunk_size: int = Field(default=1000, ge=100, le=20_000)
    chunk_overlap: int = Field(default=200, ge=0, le=5_000)
    default_folders: List[str] = Field(default_factory=lambda: ["public"])
    default_roles: List[str] = Field(
        default_factory=lambda: ["public"],
        description="Default role tags applied on ingest when none provided",
    )
    default_status: str = Field(default="published")


class RbacSettings(BaseModel):
    """
    Document access control (config-driven).

    Env examples:
      RBAC__ACCESS_MODE=or|and|hybrid
      RBAC__HYBRID_BOTH_MODE=or|and
      RBAC__REQUIRE_PUBLISHED=true
      RBAC__ENABLE_PERMISSION_TAG_MATCH=false
      RBAC__ADMIN_ROLES=["admin","developer","superadmin"]
    """

    access_mode: str = Field(
        default="or",
        description="or | and | hybrid",
    )
    hybrid_both_mode: str = Field(
        default="or",
        description="When access_mode=hybrid and doc has both roles+departments: or | and",
    )
    admin_roles: List[str] = Field(
        default_factory=lambda: ["admin", "developer", "superadmin"]
    )
    admin_bypass_permissions: List[str] = Field(
        default_factory=lambda: ["all", "documents.read.all", "rbac.bypass"]
    )
    require_published: bool = Field(default=True)
    published_status_value: str = Field(default="published")
    enable_permission_tag_match: bool = Field(
        default=False,
        description="If true, also match metadata.permissions against user permissions",
    )
    support_legacy_department: bool = Field(
        default=True,
        description="Also match legacy scalar metadata.department during migration",
    )

    @field_validator("access_mode", "hybrid_both_mode", mode="before")
    @classmethod
    def normalize_mode(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("mode must be a string")
        normalized = value.strip().lower()
        if normalized not in {"or", "and", "hybrid"}:
            if normalized not in {"or", "and"}:
                raise ValueError(f"invalid mode: {value}")
        return normalized

    @field_validator("access_mode")
    @classmethod
    def validate_access_mode(cls, value: str) -> str:
        if value not in {"or", "and", "hybrid"}:
            raise ValueError("access_mode must be one of: or, and, hybrid")
        return value

    @field_validator("hybrid_both_mode")
    @classmethod
    def validate_hybrid_both_mode(cls, value: str) -> str:
        if value not in {"or", "and"}:
            raise ValueError("hybrid_both_mode must be one of: or, and")
        return value

    @field_validator("published_status_value", mode="before")
    @classmethod
    def normalize_status(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            return "published"
        return value.strip().lower()


class Settings(BaseSettings):
    app_name: str = "IUST RAG System AI-Worker"
    debug: bool = Field(default=False)
    api_prefix: str = "/api/v1"

    laravel: LaravelSettings = LaravelSettings()
    ai: AISettings = AISettings()
    db: DatabaseSettings = DatabaseSettings()
    ingestion: IngestionSettings = IngestionSettings()
    rbac: RbacSettings = Field(default_factory=RbacSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore"
    )

    @property
    def vision_provider(self) -> str:
        return self.ai.vision_provider

    @property
    def llm_api_key(self) -> str:
        return self.ai.llm_api_key

    @property
    def llm_base_url(self) -> str:
        return self.ai.llm_base_url

    @property
    def llm_model(self) -> str:
        return self.ai.llm_model


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    """Clear cache (useful in tests when env changes)."""
    get_settings.cache_clear()