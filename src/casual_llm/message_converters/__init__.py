"""
Message converters for different LLM provider formats.

This package provides converters to translate between casual-llm's unified
ChatMessage format and provider-specific formats (OpenAI, Anthropic, Ollama).
"""

from casual_llm.message_converters.openai import (
    convert_messages_to_openai,
    convert_tool_calls_from_openai,
)
from casual_llm.message_converters.anthropic import (
    convert_messages_to_anthropic,
    convert_tool_calls_from_anthropic,
)
from casual_llm.message_converters.ollama import (
    convert_messages_to_ollama,
    convert_tool_calls_from_ollama,
)

__all__ = [
    "convert_messages_to_openai",
    "convert_messages_to_anthropic",
    "convert_messages_to_ollama",
    "convert_tool_calls_from_openai",
    "convert_tool_calls_from_anthropic",
    "convert_tool_calls_from_ollama",
]
