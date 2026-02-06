"""
LLM client implementations.

This module contains client-specific implementations of the LLMClient protocol.
"""

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


def create_client(
    config: ClientConfig,
) -> LLMClient:
    """
    Factory function to create an LLM client from a ClientConfig.

    Args:
        config: Client configuration (provider, base_url, api_key, timeout)

    Returns:
        Configured LLM client (OllamaClient, OpenAIClient, or AnthropicClient)

    Raises:
        ValueError: If provider type is not supported
        ImportError: If required package is not installed for the provider

    Examples:
        >>> from casual_llm import ClientConfig, Provider, create_client, Model
        >>>
        >>> # Create OpenAI client
        >>> config = ClientConfig(
        ...     provider=Provider.OPENAI,
        ...     api_key="sk-..."
        ... )
        >>> client = create_client(config)
        >>>
        >>> # Create models using the client
        >>> gpt4 = Model(client, name="gpt-4")
        >>> gpt35 = Model(client, name="gpt-3.5-turbo")
    """
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
            api_key=config.api_key,
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
            api_key=config.api_key,
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
        >>> from casual_llm import ClientConfig, ModelConfig, Provider
        >>> from casual_llm import create_client, create_model
        >>>
        >>> # Create client
        >>> client_config = ClientConfig(provider=Provider.OPENAI, api_key="sk-...")
        >>> client = create_client(client_config)
        >>>
        >>> # Create model
        >>> model_config = ModelConfig(name="gpt-4", temperature=0.7)
        >>> model = create_model(client, model_config)
        >>>
        >>> # Use model
        >>> response = await model.chat([UserMessage(content="Hello")])
    """
    return Model(
        client=client,
        name=config.name,
        temperature=config.temperature,
        extra_kwargs=config.extra_kwargs or None,
    )


__all__ = [
    "LLMClient",
    "ClientConfig",
    "ModelConfig",
    "Model",
    "Provider",
    "OllamaClient",
    "OpenAIClient",
    "AnthropicClient",
    "create_client",
    "create_model",
]
