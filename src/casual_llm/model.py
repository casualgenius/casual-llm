"""
Model class for LLM interactions.

Provides a user-friendly interface for chat and streaming with per-model usage tracking.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, AsyncIterator, Literal, TYPE_CHECKING

from casual_llm.config import ChatOptions
from casual_llm.messages import ChatMessage, AssistantMessage, StreamChunk
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
        >>> from casual_llm import OpenAIClient, Model, ChatOptions, UserMessage
        >>>
        >>> # Create a client (configured once)
        >>> client = OpenAIClient(api_key="...")
        >>>
        >>> # Create multiple models using the same client
        >>> gpt4 = Model(client, name="gpt-4", default_options=ChatOptions(temperature=0.7))
        >>> gpt4o = Model(client, name="gpt-4o")
        >>> gpt35 = Model(client, name="gpt-3.5-turbo", default_options=ChatOptions(temperature=0.5))
        >>>
        >>> # Use models
        >>> response = await gpt4.chat([UserMessage(content="Hello")])
        >>> print(response.content)
        >>>
        >>> # Override defaults per-call
        >>> response = await gpt4.chat(
        ...     [UserMessage(content="Be creative")],
        ...     ChatOptions(temperature=0.9, top_p=0.95),
        ... )
        >>>
        >>> # Each model tracks its own usage
        >>> print(f"GPT-4 used {gpt4.get_usage().total_tokens} tokens")
    """

    def __init__(
        self,
        client: LLMClient,
        name: str,
        default_options: ChatOptions | None = None,
        system_message_handling: Literal["passthrough", "merge"] | None = None,
    ):
        """
        Create a new Model.

        Args:
            client: The LLM client to use (OpenAIClient, OllamaClient, etc.)
            name: The model identifier (e.g., "gpt-4", "llama3.1", "claude-3-opus")
            default_options: Default options applied to all requests (can be overridden per-call)
            system_message_handling: How to handle multiple system messages
                ("passthrough" or "merge"). Resolved in order: per-call > model > client > default.
        """
        self._client = client
        self.name = name
        self.default_options = default_options
        self.system_message_handling = system_message_handling
        self._last_usage: Usage | None = None

    def _resolve_system_message_handling(
        self, options: ChatOptions
    ) -> Literal["passthrough", "merge"] | None:
        """
        Resolve the effective system_message_handling setting.

        Resolution chain: per-call ChatOptions > Model > Client > None (provider default).
        """
        if options.system_message_handling is not None:
            return options.system_message_handling
        if self.system_message_handling is not None:
            return self.system_message_handling
        val = getattr(self._client, "system_message_handling", None)
        return val

    def _merge_options(self, per_call: ChatOptions | None) -> ChatOptions:
        """
        Merge default options with per-call options.

        Per-call non-None values override defaults. The ``extra`` dicts are
        merged with per-call values winning on conflicts.
        """
        if self.default_options is None:
            return per_call or ChatOptions()
        if per_call is None:
            return self.default_options

        merged_kwargs: dict[str, Any] = {}
        for f in fields(ChatOptions):
            if f.name == "extra":
                continue
            per_call_val = getattr(per_call, f.name)
            if per_call_val is not None:
                merged_kwargs[f.name] = per_call_val
            else:
                merged_kwargs[f.name] = getattr(self.default_options, f.name)

        # Merge extra dicts (per-call wins on conflicts)
        merged_extra = {**self.default_options.extra, **per_call.extra}
        merged_kwargs["extra"] = merged_extra

        return ChatOptions(**merged_kwargs)

    async def chat(
        self,
        messages: list[ChatMessage],
        options: ChatOptions | None = None,
    ) -> AssistantMessage:
        """
        Generate a chat response from the LLM.

        Args:
            messages: List of ChatMessage (UserMessage, AssistantMessage, SystemMessage, etc.)
            options: Chat options for this request (overrides model defaults)

        Returns:
            AssistantMessage with content and optional tool_calls

        Raises:
            Provider-specific exceptions (httpx.HTTPError, openai.OpenAIError, etc.)

        Examples:
            >>> from casual_llm import ChatOptions
            >>> from pydantic import BaseModel
            >>>
            >>> class PersonInfo(BaseModel):
            ...     name: str
            ...     age: int
            >>>
            >>> # Pass Pydantic model for structured output
            >>> response = await model.chat(
            ...     messages=[UserMessage(content="Tell me about a person")],
            ...     options=ChatOptions(response_format=PersonInfo),
            ... )
        """
        merged = self._merge_options(options)
        resolved = self._resolve_system_message_handling(merged)
        if resolved is not None:
            merged = replace(merged, system_message_handling=resolved)
        result, usage = await self._client._chat(
            model=self.name,
            messages=messages,
            options=merged,
        )
        self._last_usage = usage
        return result

    async def stream(
        self,
        messages: list[ChatMessage],
        options: ChatOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a chat response from the LLM.

        This method yields response chunks in real-time as they are generated,
        enabling progressive display in chat interfaces.

        Args:
            messages: List of ChatMessage (UserMessage, AssistantMessage, SystemMessage, etc.)
            options: Chat options for this request (overrides model defaults)

        Yields:
            StreamChunk objects containing content fragments as tokens are generated.
            Each chunk has a ``content`` attribute with the text fragment.

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
        merged = self._merge_options(options)
        resolved = self._resolve_system_message_handling(merged)
        if resolved is not None:
            merged = replace(merged, system_message_handling=resolved)
        async for chunk in self._client._stream(
            model=self.name,
            messages=messages,
            options=merged,
        ):
            if chunk.usage is not None:
                self._last_usage = chunk.usage
            yield chunk

    def get_usage(self) -> Usage | None:
        """
        Get token usage statistics from the last chat() or stream() call.

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
        return f"Model(name={self.name!r}, default_options={self.default_options})"
