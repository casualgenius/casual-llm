"""
Embeddings module for text embedding providers.

This module provides embedding capabilities for RAG systems, semantic search,
and text similarity tasks.
"""

from casual_llm.config import ModelConfig, Provider
from casual_llm.embeddings.base import EmbeddingProvider
from casual_llm.embeddings.models import EmbeddingData, EmbeddingResponse
from casual_llm.embeddings.ollama import OllamaEmbeddingProvider

try:
    from casual_llm.embeddings.openai import OpenAIEmbeddingProvider
except ImportError:
    OpenAIEmbeddingProvider = None  # type: ignore


def create_embedding_provider(
    model_config: ModelConfig,
    timeout: float = 60.0,
) -> EmbeddingProvider:
    """
    Factory function to create an embedding provider from a ModelConfig.

    Args:
        model_config: Model configuration (name, provider, base_url, api_key)
        timeout: HTTP timeout in seconds (default: 60.0)

    Returns:
        Configured embedding provider (OllamaEmbeddingProvider or OpenAIEmbeddingProvider)

    Raises:
        ValueError: If provider type is not supported
        ImportError: If openai package is not installed for OpenAI provider

    Examples:
        >>> from casual_llm import ModelConfig, Provider, create_embedding_provider
        >>> config = ModelConfig(
        ...     name="text-embedding-3-small",
        ...     provider=Provider.OPENAI,
        ...     api_key="sk-..."
        ... )
        >>> provider = create_embedding_provider(config)

        >>> config = ModelConfig(
        ...     name="nomic-embed-text",
        ...     provider=Provider.OLLAMA,
        ...     base_url="http://localhost:11434"
        ... )
        >>> provider = create_embedding_provider(config)
    """
    if model_config.provider == Provider.OLLAMA:
        host = model_config.base_url or "http://localhost:11434"
        return OllamaEmbeddingProvider(
            model=model_config.name,
            host=host,
            timeout=timeout,
        )

    elif model_config.provider == Provider.OPENAI:
        if OpenAIEmbeddingProvider is None:
            raise ImportError(
                "OpenAI provider requires the 'openai' package. "
                "Install it with: pip install casual-llm[openai]"
            )

        return OpenAIEmbeddingProvider(
            model=model_config.name,
            api_key=model_config.api_key,
            base_url=model_config.base_url,
            timeout=timeout,
        )

    else:
        raise ValueError(f"Unsupported provider: {model_config.provider}")


__all__ = [
    "EmbeddingProvider",
    "EmbeddingData",
    "EmbeddingResponse",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "create_embedding_provider",
]
