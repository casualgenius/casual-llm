"""
Base protocols for LLM clients.

Provides unified interfaces for different LLM backends (OpenAI, Ollama, etc.)
using standard OpenAI-compatible message formats.
"""

from __future__ import annotations

from typing import Protocol, Literal, AsyncIterator

from pydantic import BaseModel

from casual_llm.messages import ChatMessage, AssistantMessage, StreamChunk
from casual_llm.tools import Tool
from casual_llm.usage import Usage


class LLMClient(Protocol):
    """
    Protocol for LLM clients.

    Clients manage API connections and expose internal methods for chat/stream.
    They are typically used via the Model class, which provides a user-friendly
    interface with per-model usage tracking.

    Uses OpenAI-compatible ChatMessage format for all interactions.
    Supports both structured (JSON) and unstructured (text) responses.

    This is a Protocol (PEP 544), meaning any class that implements
    these methods with this signature is compatible - no inheritance required.

    Examples:
        >>> from casual_llm import OpenAIClient, Model, UserMessage
        >>>
        >>> # Create a client (manages API connection)
        >>> client = OpenAIClient(api_key="...")
        >>>
        >>> # Create models using the client
        >>> gpt4 = Model(client, name="gpt-4")
        >>> gpt35 = Model(client, name="gpt-3.5-turbo")
        >>>
        >>> # Use the model
        >>> response = await gpt4.chat([UserMessage(content="Hello")])
    """

    async def _chat(
        self,
        model: str,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> tuple[AssistantMessage, Usage | None]:
        """
        Generate a chat response from the LLM.

        This is an internal method typically called by the Model class.
        Users should prefer using Model.chat() for a cleaner interface.

        Args:
            model: The model identifier to use (e.g., "gpt-4", "llama3.1")
            messages: List of ChatMessage (UserMessage, AssistantMessage, SystemMessage, etc.)
            response_format: Expected response format. Can be "json", "text", or a Pydantic
                BaseModel class for JSON Schema-based structured output.
            max_tokens: Maximum tokens to generate (optional)
            tools: List of tools available for the LLM to call (optional)
            temperature: Temperature for this request (optional)

        Returns:
            Tuple of (AssistantMessage, Usage or None)

        Raises:
            Provider-specific exceptions (httpx.HTTPError, openai.OpenAIError, etc.)
        """
        ...

    def _stream(
        self,
        model: str,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a chat response from the LLM.

        This is an internal method typically called by the Model class.
        Users should prefer using Model.stream() for a cleaner interface.

        Args:
            model: The model identifier to use (e.g., "gpt-4", "llama3.1")
            messages: List of ChatMessage (UserMessage, AssistantMessage, SystemMessage, etc.)
            response_format: Expected response format. Can be "json", "text", or a Pydantic
                BaseModel class for JSON Schema-based structured output.
            max_tokens: Maximum tokens to generate (optional)
            tools: List of tools available for the LLM to call (optional)
            temperature: Temperature for this request (optional)

        Yields:
            StreamChunk objects containing content fragments as tokens are generated.

        Raises:
            Provider-specific exceptions (httpx.HTTPError, openai.OpenAIError, etc.)
        """
        ...
