"""
Factory functions for creating LLM clients and models.

Provides convenient creation of configured client and model instances
from configuration objects, with automatic API key resolution.
"""

from __future__ import annotations

import logging
import os

from casual_llm.config import ClientConfig, ModelConfig, Provider
from casual_llm.model import Model
from casual_llm.providers.base import LLMClient
from casual_llm.providers.ollama import OllamaClient

try:
    from casual_llm.providers.openai import OpenAIClient
except ImportError:
    OpenAIClient = None  # type: ignore

try:
    from casual_llm.providers.anthropic import AnthropicClient
except ImportError:
    AnthropicClient = None  # type: ignore

logger = logging.getLogger(__name__)


def _resolve_api_key(config: ClientConfig) -> str | None:
    """
    Resolve API key from config or name-based env var.

    Resolution order:
    1. config.api_key (explicit key always wins)
    2. {config.name.upper()}_API_KEY env var (if name is set)
    3. None (provider SDK will use its own env var fallback)

    Args:
        config: Client configuration

    Returns:
        Resolved API key or None
    """
    if config.api_key:
        return config.api_key

    if config.name:
        env_var = f"{config.name.upper().replace('-', '_').replace('.', '_')}_API_KEY"
        env_key = os.environ.get(env_var)
        if env_key:
            logger.debug("Found API key from env var %s", env_var)
            return env_key
        else:
            logger.debug("No API key found in env var %s", env_var)

    return None


def create_client(
    config: ClientConfig,
) -> LLMClient:
    """
    Factory function to create an LLM client from a ClientConfig.

    If config.api_key is not set and config.name is provided, will attempt
    to find an API key from the {NAME.upper()}_API_KEY environment variable.

    Args:
        config: Client configuration (provider, name, base_url, api_key, timeout)

    Returns:
        Configured LLM client (OllamaClient, OpenAIClient, or AnthropicClient)

    Raises:
        ValueError: If provider type is not supported
        ImportError: If required package is not installed for the provider

    Examples:
        >>> from casual_llm import ClientConfig, create_client
        >>>
        >>> # Standard usage with string provider
        >>> config = ClientConfig(provider="openai", api_key="sk-...")
        >>> client = create_client(config)
        >>>
        >>> # With name for automatic API key lookup
        >>> config = ClientConfig(
        ...     name="openrouter",
        ...     provider="openai",
        ...     base_url="https://openrouter.ai/api/v1"
        ... )
        >>> client = create_client(config)  # uses OPENROUTER_API_KEY env var
    """
    api_key = _resolve_api_key(config)

    if config.provider == Provider.OLLAMA:
        host = config.base_url or "http://localhost:11434"
        return OllamaClient(
            host=host,
            timeout=config.timeout,
        )

    elif config.provider == Provider.OPENAI:
        if OpenAIClient is None:
            raise ImportError(
                "OpenAI client requires the 'openai' package. "
                "Install it with: pip install casual-llm[openai]"
            )

        return OpenAIClient(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            extra_kwargs=config.extra_kwargs or None,
        )

    elif config.provider == Provider.ANTHROPIC:
        if AnthropicClient is None:
            raise ImportError(
                "Anthropic client requires the 'anthropic' package. "
                "Install it with: pip install casual-llm[anthropic]"
            )

        return AnthropicClient(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            extra_kwargs=config.extra_kwargs or None,
        )

    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


def create_model(
    client: LLMClient,
    config: ModelConfig,
) -> Model:
    """
    Factory function to create a Model from a client and ModelConfig.

    Args:
        client: The LLM client to use
        config: Model configuration (name, temperature, extra_kwargs)

    Returns:
        Configured Model instance

    Examples:
        >>> from casual_llm import ClientConfig, ModelConfig, create_client, create_model
        >>>
        >>> client_config = ClientConfig(provider="openai", api_key="sk-...")
        >>> client = create_client(client_config)
        >>>
        >>> model_config = ModelConfig(name="gpt-4", temperature=0.7)
        >>> model = create_model(client, model_config)
    """
    return Model(
        client=client,
        name=config.name,
        temperature=config.temperature,
        extra_kwargs=config.extra_kwargs or None,
    )
