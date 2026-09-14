# embeddings.py
"""
Embedding factory — OpenAI-compatible only.

Any provider that exposes /v1/embeddings works by changing .env only:
  Ollama local:  AI__EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1
  GapGPT/OpenAI: AI__EMBEDDING_BASE_URL=https://api.openai.com/v1  (or your gateway)

Do NOT use langchain_community.OllamaEmbeddings (wrong path + /v1 → 404).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)


def _normalize_openai_base_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if not u:
        return "http://127.0.0.1:11434/v1"
    if u.endswith("/v1"):
        return u
    # avoid .../v1/v1 if user already put path fragments
    if u.endswith("/api"):
        u = u[: -len("/api")]
    return u + "/v1"


class EmbeddingFactory:
    _instance: Any = None
    _fingerprint: Optional[str] = None

    @classmethod
    def get_instance(cls):
        settings = get_settings()
        ai = settings.ai
        base_url = _normalize_openai_base_url(ai.embedding_base_url)
        model = (ai.embedding_model or "").strip()
        api_key = (ai.embedding_api_key or "ollama").strip() or "ollama"
        provider = (ai.embedding_provider or "openai").strip().lower()

        fp = f"{provider}|{model}|{base_url}|{api_key}"
        if cls._instance is not None and cls._fingerprint == fp:
            return cls._instance

        if not model:
            raise ValueError("AI__EMBEDDING_MODEL is empty")

        # soft guard: chat models as embedding usually break RAG
        chat_like = ("qwen2.5", "deepseek-r1", "llama3", "mistral", "gpt-4", "gpt-3.5")
        lower = model.lower()
        if any(x in lower for x in chat_like) and "embed" not in lower and "bge" not in lower:
            logger.warning(
                "Embedding model '%s' looks like a chat model. Prefer bge-m3 / nomic-embed-text.",
                model,
            )

        logger.info(
            "Initializing Embedding (openai-compatible) provider=%s model=%s base_url=%s",
            provider,
            model,
            base_url,
        )

        from langchain_openai import OpenAIEmbeddings

        cls._instance = OpenAIEmbeddings(
            model=model,
            base_url=base_url,
            api_key=api_key,
            check_embedding_ctx_length=False,
        )
        cls._fingerprint = fp
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
        cls._fingerprint = None


def get_embedding_function():
    return EmbeddingFactory.get_instance()