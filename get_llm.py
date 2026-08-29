# get_llm.py
import logging
from typing import Optional

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMFactory:
    _instances = {}

    @classmethod
    def get_instance(cls, temperature: Optional[float] = None):
        # اگر temperature پاس داده نشود، از config استفاده کن
        if temperature is None:
            temperature = settings.ai.temperature

        key = f"{settings.ai.llm_provider}_{temperature}"

        if key not in cls._instances:
            logger.info(
                f"Initializing LLM: {settings.ai.llm_provider} - {settings.ai.llm_model} "
                f"(temperature={temperature})"
            )

            try:
                if settings.ai.llm_provider == "ollama":
                    from langchain_community.chat_models import ChatOllama
                    instance = ChatOllama(
                        model=settings.ai.llm_model,
                        base_url=settings.ai.llm_base_url,
                        temperature=temperature
                    )
                else:
                    # OpenAI-compatible (GapGPT, GLM, DeepSeek, etc.)
                    from langchain_openai import ChatOpenAI
                    instance = ChatOpenAI(
                        model=settings.ai.llm_model,
                        base_url=settings.ai.llm_base_url,
                        api_key=settings.ai.llm_api_key or "dummy",
                        temperature=temperature
                    )

                cls._instances[key] = instance

            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                raise

        return cls._instances[key]


def get_llm(temperature: Optional[float] = None):
    """
    دریافت instance از LLM.
    اگر temperature مشخص نشود، مقدار settings.ai.temperature استفاده می‌شود.
    """
    return LLMFactory.get_instance(temperature)