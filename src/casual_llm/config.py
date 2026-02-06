"""
Configuration for LLM clients and models.

This module defines configuration structures for LLM clients (API connections)
and models, allowing unified configuration across different provider backends.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Provider(Enum):
    """Supported LLM providers"""

    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"


@dataclass
class ClientConfig:
    """
    Configuration for an LLM client (API connection).

    Provides a unified way to configure client connections across different providers.

    Attributes:
        provider: Provider type (OPENAI, OLLAMA, or ANTHROPIC)
        base_url: Optional custom API endpoint
        api_key: Optional API key (for OpenAI/Anthropic providers)
        timeout: HTTP request timeout in seconds (default: 60.0)
        extra_kwargs: Additional kwargs passed to the client

    Examples:
        >>> from casual_llm import ClientConfig, Provider
        >>>
        >>> # OpenAI configuration
        >>> config = ClientConfig(
        ...     provider=Provider.OPENAI,
        ...     api_key="sk-..."
        ... )
        >>>
        >>> # Ollama configuration
        >>> config = ClientConfig(
        ...     provider=Provider.OLLAMA,
        ...     base_url="http://localhost:11434"
        ... )
        >>>
        >>> # OpenRouter configuration (OpenAI-compatible)
        >>> config = ClientConfig(
        ...     provider=Provider.OPENAI,
        ...     api_key="sk-or-...",
        ...     base_url="https://openrouter.ai/api/v1"
        ... )
    """

    provider: Provider
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """
    Configuration for a specific LLM model.

    Used with a client to create Model instances.

    Attributes:
        name: Model name (e.g., "gpt-4o-mini", "qwen2.5:7b-instruct", "claude-3-5-sonnet-latest")
        temperature: Sampling temperature (0.0-1.0, optional - uses provider default if not set)
        extra_kwargs: Additional kwargs passed to chat/stream methods

    Examples:
        >>> from casual_llm import ModelConfig
        >>>
        >>> # GPT-4 configuration
        >>> config = ModelConfig(
        ...     name="gpt-4",
        ...     temperature=0.7
        ... )
        >>>
        >>> # Claude configuration
        >>> config = ModelConfig(
        ...     name="claude-3-5-sonnet-latest",
        ...     temperature=0.5
        ... )
    """

    name: str
    temperature: float | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
