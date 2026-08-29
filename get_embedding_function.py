# get_embedding_function.py
import logging
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingFactory:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info(f"Initializing Embedding: {settings.ai.embedding_provider}")

            try:
                if settings.ai.embedding_provider == "ollama":
                    from langchain_community.embeddings import OllamaEmbeddings
                    cls._instance = OllamaEmbeddings(
                        model=settings.ai.embedding_model,
                        base_url=settings.ai.embedding_base_url
                    )
                else:
                    from langchain_openai import OpenAIEmbeddings
                    cls._instance = OpenAIEmbeddings(
                        model=settings.ai.embedding_model,
                        base_url=settings.ai.embedding_base_url,
                        api_key=settings.ai.embedding_api_key or "dummy"
                    )
            except Exception as e:
                logger.error(f"Failed to initialize embeddings: {e}")
                raise

        return cls._instance


def get_embedding_function():
    return EmbeddingFactory.get_instance()