"""
casual-llm - Lightweight LLM provider abstraction with standard message models.

A simple, protocol-based library for working with different LLM providers
(OpenAI, Ollama, Anthropic) using a unified interface and OpenAI-compatible message format.

Part of the casual-* ecosystem of lightweight AI tools.

Example usage:
    >>> from casual_llm import OpenAIClient, Model, UserMessage
    >>>
    >>> # Create client (configured once)
    >>> client = OpenAIClient(api_key="...")
    >>>
    >>> # Create multiple models using the same client
    >>> gpt4 = Model(client, name="gpt-4", temperature=0.7)
    >>> gpt4o = Model(client, name="gpt-4o")
    >>>
    >>> # Use models
    >>> response = await gpt4.chat([UserMessage(content="Hello")])
    >>> print(response.content)
    >>>
    >>> # Each model tracks its own usage
    >>> print(f"Used {gpt4.get_usage().total_tokens} tokens")
"""

__version__ = "0.6.0"

# Configuration
from casual_llm.config import ChatOptions, ClientConfig, ModelConfig, Provider

# Client protocol and implementations
from casual_llm.providers import (
    LLMClient,
    OllamaClient,
    OpenAIClient,
    AnthropicClient,
)

# Factory functions
from casual_llm.factory import create_client, create_model

# Model class
from casual_llm.model import Model

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

# Utilities
from casual_llm.utils.image import ImageFetchError

# Tool converters (importable for backward compatibility, not public API)
from casual_llm.tool_converters import (  # noqa: F401
    tool_to_ollama,
    tools_to_ollama,
    tool_to_openai,
    tools_to_openai,
    tool_to_anthropic,
    tools_to_anthropic,
)

# Message converters (importable for backward compatibility, not public API)
from casual_llm.message_converters import (  # noqa: F401
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
    # Configuration
    "ChatOptions",
    "ClientConfig",
    "ModelConfig",
    "Provider",
    # Clients
    "LLMClient",
    "OllamaClient",
    "OpenAIClient",
    "AnthropicClient",
    "create_client",
    "create_model",
    # Model
    "Model",
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
    # Utilities
    "ImageFetchError",
]
