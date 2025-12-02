"""
Tool converters for different LLM provider formats.

This package provides converters to translate casual-llm's Tool format
to provider-specific formats (OpenAI, Ollama).

Note: OpenAI and Ollama currently use the same tool format (JSON Schema based),
but they are kept in separate modules for consistency and potential future divergence.
"""

from casual_llm.tool_converters.openai import (
    tool_to_openai,
    tools_to_openai,
)
from casual_llm.tool_converters.ollama import (
    tool_to_ollama,
    tools_to_ollama,
)

__all__ = [
    "tool_to_ollama",
    "tools_to_ollama",
    "tool_to_openai",
    "tools_to_openai",
]
