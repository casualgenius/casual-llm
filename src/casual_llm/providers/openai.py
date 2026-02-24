"""
OpenAI LLM client (compatible with OpenAI API and compatible services).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator
from openai import AsyncOpenAI
from pydantic import BaseModel

from casual_llm.config import ChatOptions
from casual_llm.messages import ChatMessage, AssistantMessage, StreamChunk
from casual_llm.usage import Usage
from casual_llm.tool_converters import tools_to_openai
from casual_llm.message_converters import (
    convert_messages_to_openai,
    convert_tool_calls_from_openai,
)

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    OpenAI-compatible LLM client.

    Works with OpenAI API and compatible services (OpenRouter, etc.).
    Manages the API connection - use with Model class for actual interactions.

    Examples:
        >>> from casual_llm import OpenAIClient, Model, UserMessage
        >>>
        >>> # Create client (configured once)
        >>> client = OpenAIClient(api_key="...")
        >>>
        >>> # Create models using the client
        >>> gpt4 = Model(client, name="gpt-4", temperature=0.7)
        >>> gpt4o = Model(client, name="gpt-4o")
        >>>
        >>> # Use models
        >>> response = await gpt4.chat([UserMessage(content="Hello")])
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        timeout: float = 60.0,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: API key (optional, can use OPENAI_API_KEY env var)
            base_url: Base URL for API (e.g., "https://openrouter.ai/api/v1")
            organization: OpenAI organization ID (optional)
            timeout: HTTP request timeout in seconds
        """
        client_kwargs: dict[str, Any] = {"timeout": timeout}

        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization

        self.client = AsyncOpenAI(**client_kwargs)

        logger.info("OpenAIClient initialized: base_url=%s", base_url or "default")

    def _build_request_kwargs(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build the request kwargs dict for the OpenAI API call."""
        chat_messages = convert_messages_to_openai(messages)
        logger.debug("Converted %d messages to OpenAI format", len(messages))

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
        }

        if stream:
            request_kwargs["stream"] = True

        if options.temperature is not None:
            request_kwargs["temperature"] = options.temperature

        if options.top_p is not None:
            request_kwargs["top_p"] = options.top_p

        if options.frequency_penalty is not None:
            request_kwargs["frequency_penalty"] = options.frequency_penalty

        if options.presence_penalty is not None:
            request_kwargs["presence_penalty"] = options.presence_penalty

        if options.seed is not None:
            request_kwargs["seed"] = options.seed

        if options.stop is not None:
            request_kwargs["stop"] = options.stop

        # Handle response_format
        if options.response_format == "json":
            request_kwargs["response_format"] = {"type": "json_object"}
        elif isinstance(options.response_format, type) and issubclass(
            options.response_format, BaseModel
        ):
            schema = options.response_format.model_json_schema()
            request_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": options.response_format.__name__,
                    "schema": schema,
                },
            }
            logger.debug(
                "Using JSON Schema from Pydantic model: %s",
                options.response_format.__name__,
            )

        if options.max_tokens is not None:
            request_kwargs["max_tokens"] = options.max_tokens

        # Add tools if provided
        if options.tools:
            converted_tools = tools_to_openai(options.tools)
            request_kwargs["tools"] = converted_tools
            logger.debug("Added %d tools to request", len(converted_tools))

        # Handle tool_choice
        if options.tool_choice is not None and options.tools:
            if options.tool_choice in ("auto", "none", "required"):
                request_kwargs["tool_choice"] = options.tool_choice
            else:
                # Treat as a specific tool name
                request_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": options.tool_choice},
                }

        # Merge extra kwargs (provider-specific pass-through)
        # Only add keys that don't conflict with core parameters
        for key, value in options.extra.items():
            if key not in request_kwargs:
                request_kwargs[key] = value
            else:
                logger.warning(
                    "Ignoring extra key %r that conflicts with a core request parameter", key
                )

        return request_kwargs

    async def _chat(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> tuple[AssistantMessage, Usage | None]:
        """
        Generate a chat response using OpenAI API.

        This is an internal method typically called by the Model class.

        Args:
            model: Model name (e.g., "gpt-4o-mini")
            messages: Conversation messages (ChatMessage format)
            options: Chat options controlling response format, sampling, tools, etc.

        Returns:
            Tuple of (AssistantMessage, Usage or None)

        Raises:
            openai.OpenAIError: If request fails
        """
        request_kwargs = self._build_request_kwargs(model, messages, options)

        logger.debug("Generating with model %s", model)
        response = await self.client.chat.completions.create(**request_kwargs)

        response_message = response.choices[0].message

        # Extract usage statistics
        usage: Usage | None = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
            logger.debug(
                "Usage: %d prompt tokens, %d completion tokens",
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

        # Parse tool calls if present
        tool_calls = None
        if hasattr(response_message, "tool_calls") and response_message.tool_calls:
            logger.debug("Assistant requested %d tool calls", len(response_message.tool_calls))
            tool_calls = convert_tool_calls_from_openai(response_message.tool_calls)

        # Return AssistantMessage and Usage
        content = response_message.content or ""
        logger.debug("Generated %d characters", len(content))
        return AssistantMessage(content=content, tool_calls=tool_calls), usage

    async def _stream(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a chat response from OpenAI API.

        This is an internal method typically called by the Model class.

        Args:
            model: Model name (e.g., "gpt-4o-mini")
            messages: Conversation messages (ChatMessage format)
            options: Chat options controlling response format, sampling, tools, etc.

        Yields:
            StreamChunk objects containing content fragments as tokens are generated.

        Raises:
            openai.OpenAIError: If request fails
        """
        request_kwargs = self._build_request_kwargs(model, messages, options, stream=True)
        # Request usage stats in the final streamed chunk
        request_kwargs["stream_options"] = {"include_usage": True}

        logger.debug("Starting stream with model %s", model)
        stream = await self.client.chat.completions.create(**request_kwargs)

        usage: Usage | None = None
        async for chunk in stream:
            # The final chunk carries usage data (no choices)
            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage = Usage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                )

            # Extract content from the delta if present
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                finish_reason = chunk.choices[0].finish_reason
                yield StreamChunk(content=content, finish_reason=finish_reason)

        # Yield a final empty chunk with usage if we captured it
        if usage is not None:
            yield StreamChunk(content="", finish_reason="stop", usage=usage)

        logger.debug("Stream completed")
