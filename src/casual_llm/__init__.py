"""
casual-llm - Lightweight LLM provider abstraction with standard message models.

A simple, protocol-based library for working with different LLM providers
(OpenAI, Ollama, etc.) using a unified interface and OpenAI-compatible message format.

Part of the casual-* ecosystem of lightweight AI tools.
"""

__version__ = "0.1.0"

# Model configuration
from casual_llm.config import ModelConfig, Provider

# Provider protocol and implementations
from casual_llm.providers import (
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
    create_provider,
)

# OpenAI-compatible message models
from casual_llm.messages import (
    ChatMessage,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    AssistantToolCall,
    AssistantToolCallFunction,
)

# Utilities
from casual_llm.utils import extract_json_from_markdown

__all__ = [
    # Version
    "__version__",
    # Providers
    "LLMProvider",
    "ModelConfig",
    "Provider",
    "OllamaProvider",
    "OpenAIProvider",
    "create_provider",
    # Messages
    "ChatMessage",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "ToolResultMessage",
    "AssistantToolCall",
    "AssistantToolCallFunction",
    # Utils
    "extract_json_from_markdown",
]
