"""
Model configuration and provider enums.

This module defines configuration structures for LLM models,
allowing unified configuration across different provider backends.
"""

from dataclasses import dataclass
from enum import Enum


class Provider(Enum):
    """Supported LLM providers"""

    OPENAI = "openai"
    OLLAMA = "ollama"


@dataclass
class ModelConfig:
    """
    Configuration for a specific LLM model.

    Provides a unified way to configure models across different providers.

    Attributes:
        name: Model name (e.g., "gpt-4o-mini", "qwen2.5:7b-instruct")
        provider: Provider type (OPENAI or OLLAMA)
        base_url: Optional custom API endpoint
        api_key: Optional API key (for OpenAI/compatible providers)
        temperature: Sampling temperature (0.0-1.0, default: 0.1)

    Examples:
        >>> from casual_llm import ModelConfig, Provider
        >>>
        >>> # OpenAI configuration
        >>> config = ModelConfig(
        ...     name="gpt-4o-mini",
        ...     provider=Provider.OPENAI,
        ...     api_key="sk-..."
        ... )
        >>>
        >>> # Ollama configuration
        >>> config = ModelConfig(
        ...     name="qwen2.5:7b-instruct",
        ...     provider=Provider.OLLAMA,
        ...     base_url="http://localhost:11434/api/generate"
        ... )
        >>>
        >>> # OpenRouter configuration (OpenAI-compatible)
        >>> config = ModelConfig(
        ...     name="anthropic/claude-3.5-sonnet",
        ...     provider=Provider.OPENAI,
        ...     api_key="sk-or-...",
        ...     base_url="https://openrouter.ai/api/v1"
        ... )
    """

    name: str
    provider: Provider
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.1
