"""
casual-llm - Lightweight LLM provider abstraction with standard message models.

A simple, protocol-based library for working with different LLM providers
(OpenAI, Ollama, etc.) using a unified interface and OpenAI-compatible message format.

Part of the casual-* ecosystem of lightweight AI tools.
"""

__version__ = "0.4.3"

# Model configuration
from casual_llm.config import ModelConfig, Provider

# Provider protocol and implementations
from casual_llm.providers import (
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
    AnthropicProvider,
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
    StreamChunk,
    # Multimodal content types
    TextContent,
    ImageContent,
)

# Tool models
from casual_llm.tools import Tool, ToolParameter

# Usage tracking
from casual_llm.usage import Usage

# Tool converters
from casual_llm.tool_converters import (
    tool_to_ollama,
    tools_to_ollama,
    tool_to_openai,
    tools_to_openai,
    tool_to_anthropic,
    tools_to_anthropic,
)

# Message converters
from casual_llm.message_converters import (
    convert_messages_to_openai,
    convert_messages_to_ollama,
    convert_messages_to_anthropic,
    convert_tool_calls_from_openai,
    convert_tool_calls_from_ollama,
    convert_tool_calls_from_anthropic,
    extract_system_message,
)

__all__ = [
    # Version
    "__version__",
    # Providers
    "LLMProvider",
    "ModelConfig",
    "Provider",
    "OllamaProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "create_provider",
    # Messages
    "ChatMessage",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "ToolResultMessage",
    "AssistantToolCall",
    "AssistantToolCallFunction",
    "StreamChunk",
    # Multimodal content types
    "TextContent",
    "ImageContent",
    # Tools
    "Tool",
    "ToolParameter",
    # Usage
    "Usage",
    # Tool converters
    "tool_to_ollama",
    "tools_to_ollama",
    "tool_to_openai",
    "tools_to_openai",
    "tool_to_anthropic",
    "tools_to_anthropic",
    # Message converters
    "convert_messages_to_openai",
    "convert_messages_to_ollama",
    "convert_messages_to_anthropic",
    "convert_tool_calls_from_openai",
    "convert_tool_calls_from_ollama",
    "convert_tool_calls_from_anthropic",
    "extract_system_message",
]
