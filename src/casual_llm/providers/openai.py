"""
OpenAI LLM client (compatible with OpenAI API and compatible services).
"""

from __future__ import annotations

import logging
from typing import Literal, Any, AsyncIterator
from openai import AsyncOpenAI
from pydantic import BaseModel

from casual_llm.messages import ChatMessage, AssistantMessage, StreamChunk
from casual_llm.tools import Tool
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
        extra_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: API key (optional, can use OPENAI_API_KEY env var)
            base_url: Base URL for API (e.g., "https://openrouter.ai/api/v1")
            organization: OpenAI organization ID (optional)
            timeout: HTTP request timeout in seconds
            extra_kwargs: Additional kwargs to pass to client.chat.completions.create()
        """
        client_kwargs: dict[str, Any] = {"timeout": timeout}

        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization

        self.client = AsyncOpenAI(**client_kwargs)
        self.extra_kwargs = extra_kwargs or {}

        logger.info("OpenAIClient initialized: base_url=%s", base_url or "default")

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
        Generate a chat response using OpenAI API.

        This is an internal method typically called by the Model class.

        Args:
            model: Model name (e.g., "gpt-4o-mini")
            messages: Conversation messages (ChatMessage format)
            response_format: "json" for JSON output, "text" for plain text, or a Pydantic
                BaseModel class for JSON Schema-based structured output.
            max_tokens: Maximum tokens to generate (optional)
            tools: List of tools available for the LLM to call (optional)
            temperature: Temperature for this request (optional)

        Returns:
            Tuple of (AssistantMessage, Usage or None)

        Raises:
            openai.OpenAIError: If request fails
        """
        # Convert messages to OpenAI format using converter
        chat_messages = convert_messages_to_openai(messages)
        logger.debug("Converted %d messages to OpenAI format", len(messages))

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
        }

        # Only add temperature if specified
        if temperature is not None:
            request_kwargs["temperature"] = temperature

        # Handle response_format: "json", "text", or Pydantic model class
        if response_format == "json":
            request_kwargs["response_format"] = {"type": "json_object"}
        elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
            # Extract JSON Schema from Pydantic model
            schema = response_format.model_json_schema()
            request_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": schema,
                },
            }
            logger.debug("Using JSON Schema from Pydantic model: %s", response_format.__name__)
        # "text" is the default - no response_format needed

        if max_tokens:
            request_kwargs["max_tokens"] = max_tokens

        # Add tools if provided
        if tools:
            converted_tools = tools_to_openai(tools)
            request_kwargs["tools"] = converted_tools
            logger.debug("Added %d tools to request", len(converted_tools))

        # Merge extra kwargs
        request_kwargs.update(self.extra_kwargs)

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
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a chat response from OpenAI API.

        This is an internal method typically called by the Model class.

        Args:
            model: Model name (e.g., "gpt-4o-mini")
            messages: Conversation messages (ChatMessage format)
            response_format: "json" for JSON output, "text" for plain text, or a Pydantic
                BaseModel class for JSON Schema-based structured output.
            max_tokens: Maximum tokens to generate (optional)
            tools: List of tools available for the LLM to call (optional)
            temperature: Temperature for this request (optional)

        Yields:
            StreamChunk objects containing content fragments as tokens are generated.

        Raises:
            openai.OpenAIError: If request fails
        """
        # Convert messages to OpenAI format using converter
        chat_messages = convert_messages_to_openai(messages)
        logger.debug("Converted %d messages to OpenAI format for streaming", len(messages))

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "stream": True,
        }

        # Only add temperature if specified
        if temperature is not None:
            request_kwargs["temperature"] = temperature

        # Handle response_format: "json", "text", or Pydantic model class
        if response_format == "json":
            request_kwargs["response_format"] = {"type": "json_object"}
        elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
            # Extract JSON Schema from Pydantic model
            schema = response_format.model_json_schema()
            request_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": schema,
                },
            }
            logger.debug("Using JSON Schema from Pydantic model: %s", response_format.__name__)
        # "text" is the default - no response_format needed

        if max_tokens:
            request_kwargs["max_tokens"] = max_tokens

        # Add tools if provided
        if tools:
            converted_tools = tools_to_openai(tools)
            request_kwargs["tools"] = converted_tools
            logger.debug("Added %d tools to streaming request", len(converted_tools))

        # Merge extra kwargs
        request_kwargs.update(self.extra_kwargs)

        logger.debug("Starting stream with model %s", model)
        stream = await self.client.chat.completions.create(**request_kwargs)

        async for chunk in stream:
            # Extract content from the delta if present
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                finish_reason = chunk.choices[0].finish_reason
                yield StreamChunk(content=content, finish_reason=finish_reason)

        logger.debug("Stream completed")
