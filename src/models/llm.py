# llm.py
"""
LLM factory — OpenAI-compatible only.

Switch providers only via .env (base_url + model + api_key).
Ollama: AI__LLM_BASE_URL=http://127.0.0.1:11434/v1
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)


def _normalize_openai_base_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if not u:
        return "http://127.0.0.1:11434/v1"
    if u.endswith("/v1"):
        return u
    if u.endswith("/api"):
        u = u[: -len("/api")]
    return u + "/v1"


class LLMFactory:
    _instances: Dict[str, Any] = {}

    @classmethod
    def get_instance(cls, temperature: Optional[float] = None):
        settings = get_settings()
        ai = settings.ai
        if temperature is None:
            temperature = float(getattr(ai, "temperature", 0.2) or 0.2)

        base_url = _normalize_openai_base_url(ai.llm_base_url)
        model = (ai.llm_model or "").strip()
        api_key = (ai.llm_api_key or "ollama").strip() or "ollama"
        provider = (ai.llm_provider or "openai").strip().lower()

        if not model:
            raise ValueError("AI__LLM_MODEL is empty")

        key = f"{provider}|{model}|{base_url}|{temperature}|{api_key}"
        if key in cls._instances:
            return cls._instances[key]

        logger.info(
            "Initializing LLM (openai-compatible) provider=%s model=%s base_url=%s temp=%s",
            provider,
            model,
            base_url,
            temperature,
        )

        from langchain_openai import ChatOpenAI

        instance = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
        cls._instances[key] = instance
        return instance

    @classmethod
    def reset(cls) -> None:
        cls._instances.clear()


def get_llm(temperature: Optional[float] = None):
    return LLMFactory.get_instance(temperature)