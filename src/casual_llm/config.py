"""
Configuration for LLM clients and models.

This module defines configuration structures for LLM clients (API connections),
models, and chat options for unified configuration across different provider backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

    from casual_llm.tools import Tool


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
        provider: Provider type - accepts Provider enum or string (e.g., "openai", "OpenAI")
        name: Optional client name for automatic API key lookup from env vars.
              If set and no api_key is provided, checks {NAME.upper()}_API_KEY env var.
        base_url: Optional custom API endpoint
        api_key: Optional API key (for OpenAI/Anthropic providers)
        timeout: HTTP request timeout in seconds (default: 60.0)
        extra_kwargs: Additional kwargs passed to the client

    Examples:
        >>> from casual_llm import ClientConfig, Provider
        >>>
        >>> # Using Provider enum
        >>> config = ClientConfig(provider=Provider.OPENAI, api_key="sk-...")
        >>>
        >>> # Using string (convenient for JSON configs)
        >>> config = ClientConfig(provider="openai", api_key="sk-...")
        >>>
        >>> # With name for automatic API key lookup
        >>> config = ClientConfig(
        ...     name="openrouter", provider="openai", base_url="https://openrouter.ai/api/v1"
        ... )
        >>> # Will check OPENROUTER_API_KEY env var automatically
    """

    provider: Provider | str
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    system_message_handling: Literal["passthrough", "merge"] | None = None

    def __post_init__(self) -> None:
        """Coerce string provider to Provider enum."""
        if isinstance(self.provider, str):
            try:
                self.provider = Provider(self.provider.lower())
            except ValueError:
                valid = ", ".join(p.value for p in Provider)
                raise ValueError(
                    f"Unknown provider: {self.provider!r}. Valid providers: {valid}"
                ) from None

    def __repr__(self) -> str:
        masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}" if self.api_key else None
        return (
            f"ClientConfig(provider={self.provider!r}, name={self.name!r}, "
            f"base_url={self.base_url!r}, api_key={masked_key!r}, "
            f"timeout={self.timeout!r})"
        )


@dataclass
class ChatOptions:
    """
    Options for chat/stream requests.

    All fields are optional — providers silently ignore params they don't support.
    Use ``extra`` for provider-specific params not covered by first-class fields.

    Attributes:
        response_format: Expected response format ("json", "text", or Pydantic model class)
        max_tokens: Maximum tokens to generate
        tools: List of tools available for the LLM to call
        temperature: Sampling temperature (creativity control)
        top_p: Nucleus sampling probability threshold
        stop: Custom stop sequences
        tool_choice: Tool use control ("auto", "none", "required", or a tool name)
        frequency_penalty: Penalize tokens by repetition frequency (OpenAI, Ollama)
        presence_penalty: Penalize tokens for appearing at all (OpenAI, Ollama)
        seed: Seed for reproducible outputs (OpenAI, Ollama)
        top_k: Top-k sampling (Anthropic, Ollama)
        extra: Additional provider-specific kwargs passed to the API call

    Examples:
        >>> from casual_llm import ChatOptions
        >>>
        >>> # Simple options
        >>> opts = ChatOptions(temperature=0.7, top_p=0.9)
        >>>
        >>> # Reusable presets
        >>> creative = ChatOptions(temperature=0.9, top_p=0.95, frequency_penalty=0.5)
        >>> deterministic = ChatOptions(temperature=0.0, seed=42)
        >>>
        >>> # Provider-specific pass-through
        >>> opts = ChatOptions(temperature=0.7, extra={"logprobs": True})
    """

    response_format: Literal["json", "text"] | type[BaseModel] = "text"
    max_tokens: int | None = None
    tools: list[Tool] | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    tool_choice: Literal["auto", "none", "required"] | str | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    top_k: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    system_message_handling: Literal["passthrough", "merge"] | None = None


@dataclass
class ModelConfig:
    """
    Configuration for a specific LLM model.

    Used with a client to create Model instances.

    Attributes:
        name: Model name (e.g., "gpt-4o-mini", "qwen2.5:7b-instruct", "claude-3-5-sonnet-latest")
        default_options: Default ChatOptions applied to all requests from this model

    Examples:
        >>> from casual_llm import ModelConfig, ChatOptions
        >>>
        >>> # GPT-4 configuration
        >>> config = ModelConfig(name="gpt-4", default_options=ChatOptions(temperature=0.7))
        >>>
        >>> # Claude configuration
        >>> config = ModelConfig(
        ...     name="claude-3-5-sonnet-latest", default_options=ChatOptions(temperature=0.5)
        ... )
    """

    name: str
    default_options: ChatOptions | None = None
    system_message_handling: Literal["passthrough", "merge"] | None = None
