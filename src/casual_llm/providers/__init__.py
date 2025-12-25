"""
LLM provider implementations.

This module contains provider-specific implementations of the LLMProvider protocol.
"""

from casual_llm.config import ModelConfig, Provider
from casual_llm.providers.base import LLMProvider
from casual_llm.providers.ollama import OllamaProvider

try:
    from casual_llm.providers.openai import OpenAIProvider
except ImportError:
    OpenAIProvider = None  # type: ignore


def create_provider(
    model_config: ModelConfig,
    timeout: float = 60.0,
) -> LLMProvider:
    """
    Factory function to create an LLM provider from a ModelConfig.

    Args:
        model_config: Model configuration (name, provider, base_url, api_key, temperature, retry_config)
        timeout: HTTP timeout in seconds (default: 60.0)

    Returns:
        Configured LLM provider (OllamaProvider or OpenAIProvider)

    Raises:
        ValueError: If provider type is not supported
        ImportError: If openai package is not installed for OpenAI provider

    Examples:
        >>> from casual_llm import ModelConfig, Provider, create_provider
        >>> config = ModelConfig(
        ...     name="gpt-4o-mini",
        ...     provider=Provider.OPENAI,
        ...     api_key="sk-..."
        ... )
        >>> provider = create_provider(config)

        >>> config = ModelConfig(
        ...     name="qwen2.5:7b-instruct",
        ...     provider=Provider.OLLAMA,
        ...     base_url="http://localhost:11434"
        ... )
        >>> provider = create_provider(config)
    """
    if model_config.provider == Provider.OLLAMA:
        host = model_config.base_url or "http://localhost:11434"
        return OllamaProvider(
            model=model_config.name,
            host=host,
            temperature=model_config.temperature,
            timeout=timeout,
            retry_config=model_config.retry_config,
        )

    elif model_config.provider == Provider.OPENAI:
        if OpenAIProvider is None:
            raise ImportError(
                "OpenAI provider requires the 'openai' package. "
                "Install it with: pip install casual-llm[openai]"
            )

        return OpenAIProvider(
            model=model_config.name,
            api_key=model_config.api_key,
            base_url=model_config.base_url,
            temperature=model_config.temperature,
            timeout=timeout,
            retry_config=model_config.retry_config,
        )

    else:
        raise ValueError(f"Unsupported provider: {model_config.provider}")


__all__ = [
    "LLMProvider",
    "ModelConfig",
    "Provider",
    "OllamaProvider",
    "OpenAIProvider",
    "create_provider",
]
