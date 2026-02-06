"""
Model class for LLM interactions.

Provides a user-friendly interface for chat and streaming with per-model usage tracking.
"""

from __future__ import annotations

from typing import Literal, AsyncIterator, Any, TYPE_CHECKING

from pydantic import BaseModel

from casual_llm.messages import ChatMessage, AssistantMessage, StreamChunk
from casual_llm.tools import Tool
from casual_llm.usage import Usage

if TYPE_CHECKING:
    from casual_llm.providers.base import LLMClient


class Model:
    """
    User-facing class for LLM interactions.

    A Model wraps an LLMClient with model-specific configuration. This allows
    configuring providers once and creating multiple models that share the
    same connection.

    Examples:
        >>> from casual_llm import OpenAIClient, Model, UserMessage
        >>>
        >>> # Create a client (configured once)
        >>> client = OpenAIClient(api_key="...")
        >>>
        >>> # Create multiple models using the same client
        >>> gpt4 = Model(client, name="gpt-4", temperature=0.7)
        >>> gpt4o = Model(client, name="gpt-4o")
        >>> gpt35 = Model(client, name="gpt-3.5-turbo", temperature=0.5)
        >>>
        >>> # Use models
        >>> response = await gpt4.chat([UserMessage(content="Hello")])
        >>> print(response.content)
        >>>
        >>> # Each model tracks its own usage
        >>> print(f"GPT-4 used {gpt4.get_usage().total_tokens} tokens")
    """

    def __init__(
        self,
        client: LLMClient,
        name: str,
        temperature: float | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ):
        """
        Create a new Model.

        Args:
            client: The LLM client to use (OpenAIClient, OllamaClient, etc.)
            name: The model identifier (e.g., "gpt-4", "llama3.1", "claude-3-opus")
            temperature: Default temperature for this model (can be overridden per-call)
            extra_kwargs: Extra keyword arguments passed to the client methods
        """
        self._client = client
        self.name = name
        self.temperature = temperature
        self.extra_kwargs = extra_kwargs or {}
        self._last_usage: Usage | None = None

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
            temperature: Temperature for this request (optional, overrides model default)

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
            >>> response = await model.chat(
            ...     messages=[UserMessage(content="Tell me about a person")],
            ...     response_format=PersonInfo
            ... )
        """
        temp = temperature if temperature is not None else self.temperature
        result, usage = await self._client._chat(
            model=self.name,
            messages=messages,
            response_format=response_format,
            max_tokens=max_tokens,
            tools=tools,
            temperature=temp,
        )
        self._last_usage = usage
        return result

    async def stream(
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
                BaseModel class for JSON Schema-based structured output.
            max_tokens: Maximum tokens to generate (optional)
            tools: List of tools available for the LLM to call (optional, may not work
                with all providers during streaming)
            temperature: Temperature for this request (optional, overrides model default)

        Yields:
            StreamChunk objects containing content fragments as tokens are generated.
            Each chunk has a `content` attribute with the text fragment.

        Raises:
            Provider-specific exceptions (httpx.HTTPError, openai.OpenAIError, etc.)

        Examples:
            >>> from casual_llm import UserMessage
            >>>
            >>> # Stream response and print tokens as they arrive
            >>> async for chunk in model.stream([UserMessage(content="Tell me a story")]):
            ...     print(chunk.content, end="", flush=True)
            >>>
            >>> # Collect full response from stream
            >>> chunks = []
            >>> async for chunk in model.stream([UserMessage(content="Hello")]):
            ...     chunks.append(chunk.content)
            >>> full_response = "".join(chunks)
        """
        temp = temperature if temperature is not None else self.temperature
        async for chunk in self._client._stream(
            model=self.name,
            messages=messages,
            response_format=response_format,
            max_tokens=max_tokens,
            tools=tools,
            temperature=temp,
        ):
            yield chunk

    def get_usage(self) -> Usage | None:
        """
        Get token usage statistics from the last chat() call.

        Returns:
            Usage object with prompt_tokens, completion_tokens, and total_tokens,
            or None if no calls have been made yet.

        Examples:
            >>> model = Model(client, name="gpt-4")
            >>> await model.chat([UserMessage(content="Hello")])
            >>> usage = model.get_usage()
            >>> if usage:
            ...     print(f"Used {usage.total_tokens} tokens")
        """
        return self._last_usage

    def __repr__(self) -> str:
        return f"Model(name={self.name!r}, temperature={self.temperature})"
