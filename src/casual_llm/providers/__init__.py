"""
LLM client implementations.

This module contains client-specific implementations of the LLMClient protocol.
"""

from casual_llm.providers.base import LLMClient
from casual_llm.providers.ollama import OllamaClient

try:
    from casual_llm.providers.openai import OpenAIClient
except ImportError:
    OpenAIClient = None  # type: ignore[assignment, misc]

try:
    from casual_llm.providers.anthropic import AnthropicClient
except ImportError:
    AnthropicClient = None  # type: ignore[assignment, misc]

__all__ = [
    "LLMClient",
    "OllamaClient",
    "OpenAIClient",
    "AnthropicClient",
]
