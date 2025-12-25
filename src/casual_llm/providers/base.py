"""
Base protocol for LLM providers.

Provides a unified interface for different LLM backends (OpenAI, Ollama, etc.)
using standard OpenAI-compatible message formats.
"""

from __future__ import annotations

from typing import Protocol, Literal, AsyncIterator

from pydantic import BaseModel

from casual_llm.messages import ChatMessage, AssistantMessage, StreamChunk
from casual_llm.tools import Tool
from casual_llm.usage import Usage


class LLMProvider(Protocol):
    """
    Protocol for LLM providers.

    Uses OpenAI-compatible ChatMessage format for all interactions.
    Supports both structured (JSON) and unstructured (text) responses.

    This is a Protocol (PEP 544), meaning any class that implements
    the chat() method with this signature is compatible - no
    inheritance required.

    Examples:
        >>> from casual_llm import LLMProvider, ChatMessage, UserMessage
        >>>
        >>> # Any provider implementing this protocol works
        >>> async def get_response(provider: LLMProvider, prompt: str) -> str:
        ...     messages = [UserMessage(content=prompt)]
        ...     return await provider.chat(messages)
    """

    async def chat(
        self,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> AssistantMessage:
        """
        Generate a chat response from the LLM.

        Args:
            messages: List of ChatMessage (UserMessage, AssistantMessage, SystemMessage, etc.)
            response_format: Expected response format. Can be "json", "text", or a Pydantic
                BaseModel class for JSON Schema-based structured output. When a Pydantic model
                is provided, the LLM will be instructed to return JSON matching the schema.
            max_tokens: Maximum tokens to generate (optional)
            tools: List of tools available for the LLM to call (optional)
            temperature: Temperature for this request (optional, overrides instance temperature)

        Returns:
            AssistantMessage with content and optional tool_calls

        Raises:
            Provider-specific exceptions (httpx.HTTPError, openai.OpenAIError, etc.)

        Examples:
            >>> from pydantic import BaseModel
            >>>
            >>> class PersonInfo(BaseModel):
            ...     name: str
            ...     age: int
            >>>
            >>> # Pass Pydantic model for structured output
            >>> response = await provider.chat(
            ...     messages=[UserMessage(content="Tell me about a person")],
            ...     response_format=PersonInfo  # Pass the class, not an instance
            ... )
        """
        ...

    def stream(
        self,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a chat response from the LLM.

        This method yields response chunks in real-time as they are generated,
        enabling progressive display in chat interfaces.

        Args:
            messages: List of ChatMessage (UserMessage, AssistantMessage, SystemMessage, etc.)
            response_format: Expected response format. Can be "json", "text", or a Pydantic
                BaseModel class for JSON Schema-based structured output. When a Pydantic model
                is provided, the LLM will be instructed to return JSON matching the schema.
            max_tokens: Maximum tokens to generate (optional)
            tools: List of tools available for the LLM to call (optional, may not work
                with all providers during streaming)
            temperature: Temperature for this request (optional, overrides instance temperature)

        Yields:
            StreamChunk objects containing content fragments as tokens are generated.
            Each chunk has a `content` attribute with the text fragment.

        Raises:
            Provider-specific exceptions (httpx.HTTPError, openai.OpenAIError, etc.)

        Examples:
            >>> from casual_llm import UserMessage
            >>>
            >>> # Stream response and print tokens as they arrive
            >>> async for chunk in provider.stream([UserMessage(content="Tell me a story")]):
            ...     print(chunk.content, end="", flush=True)
            >>>
            >>> # Collect full response from stream
            >>> chunks = []
            >>> async for chunk in provider.stream([UserMessage(content="Hello")]):
            ...     chunks.append(chunk.content)
            >>> full_response = "".join(chunks)
        """
        ...

    def get_usage(self) -> Usage | None:
        """
        Get token usage statistics from the last chat() call.

        Returns:
            Usage object with prompt_tokens, completion_tokens, and total_tokens,
            or None if no calls have been made yet.

        Examples:
            >>> provider = OllamaProvider(model="llama3.1")
            >>> await provider.chat([UserMessage(content="Hello")])
            >>> usage = provider.get_usage()
            >>> if usage:
            ...     print(f"Used {usage.total_tokens} tokens")
        """
        ...
