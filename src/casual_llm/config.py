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
class RetryConfig:
    """
    Configuration for retry behavior with exponential backoff.

    Used to configure how API calls should be retried on transient failures.

    Attributes:
        max_attempts: Maximum number of retry attempts (default: 3)
        backoff_factor: Multiplier for exponential backoff delay (default: 2.0)

    Examples:
        >>> from casual_llm.config import RetryConfig
        >>>
        >>> # Default configuration
        >>> config = RetryConfig()
        >>> print(config.max_attempts)  # 3
        >>> print(config.backoff_factor)  # 2.0
        >>>
        >>> # Custom configuration
        >>> config = RetryConfig(max_attempts=5, backoff_factor=1.5)
    """

    max_attempts: int = 3
    backoff_factor: float = 2.0


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
        temperature: Sampling temperature (0.0-1.0, optional - uses provider default if not set)
        retry_config: Optional retry configuration for API calls

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
        ...     base_url="http://localhost:11434"
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
    temperature: float | None = None
    retry_config: RetryConfig | None = None
