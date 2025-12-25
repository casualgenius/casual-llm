"""
OpenAI LLM provider (compatible with OpenAI API and compatible services).
"""

from __future__ import annotations

import logging
from typing import Literal, Any
from openai import AsyncOpenAI
from pydantic import BaseModel

from casual_llm.config import RetryConfig
from casual_llm.messages import ChatMessage, AssistantMessage
from casual_llm.tools import Tool
from casual_llm.usage import Usage
from casual_llm.tool_converters import tools_to_openai
from casual_llm.message_converters import (
    convert_messages_to_openai,
    convert_tool_calls_from_openai,
)

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """
    OpenAI-compatible LLM provider.

    Works with OpenAI API and compatible services (OpenRouter, etc.).
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        temperature: float | None = None,
        timeout: float = 60.0,
        extra_kwargs: dict[str, Any] | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: Model name (e.g., "gpt-4o-mini")
            api_key: API key (optional, can use OPENAI_API_KEY env var)
            base_url: Base URL for API (e.g., "https://openrouter.ai/api/v1")
            organization: OpenAI organization ID (optional)
            temperature: Temperature for generation (0.0-1.0, optional - uses OpenAI default if not set)
            timeout: HTTP request timeout in seconds
            extra_kwargs: Additional kwargs to pass to client.chat.completions.create()
            retry_config: Configuration for retry behavior with exponential backoff (optional)
        """
        client_kwargs: dict[str, Any] = {"timeout": timeout}

        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization

        self.client = AsyncOpenAI(**client_kwargs)
        self.model = model
        self.temperature = temperature
        self.extra_kwargs = extra_kwargs or {}
        self.retry_config = retry_config

        # Usage tracking
        self._last_usage: Usage | None = None

        logger.info(
            f"OpenAIProvider initialized: model={model}, " f"base_url={base_url or 'default'}"
        )

    def get_usage(self) -> Usage | None:
        """
        Get token usage statistics from the last chat() call.

        Returns:
            Usage object with token counts, or None if no calls have been made
        """
        return self._last_usage

    async def chat(
        self,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> AssistantMessage:
        """
        Generate a chat response using OpenAI API.

        Args:
            messages: Conversation messages (ChatMessage format)
            response_format: "json" for JSON output, "text" for plain text, or a Pydantic
                BaseModel class for JSON Schema-based structured output. When a Pydantic
                model is provided, the LLM will be instructed to return JSON matching the
                schema.
            max_tokens: Maximum tokens to generate (optional)
            tools: List of tools available for the LLM to call (optional)
            temperature: Temperature for this request (optional, overrides instance temperature)

        Returns:
            AssistantMessage with content and optional tool_calls

        Raises:
            openai.OpenAIError: If request fails

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
        # Convert messages to OpenAI format using converter
        chat_messages = convert_messages_to_openai(messages)
        logger.debug(f"Converted {len(messages)} messages to OpenAI format")

        # Use provided temperature or fall back to instance temperature
        temp = temperature if temperature is not None else self.temperature

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
        }

        # Only add temperature if specified
        if temp is not None:
            request_kwargs["temperature"] = temp

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
            logger.debug(f"Using JSON Schema from Pydantic model: {response_format.__name__}")
        # "text" is the default - no response_format needed

        if max_tokens:
            request_kwargs["max_tokens"] = max_tokens

        # Add tools if provided
        if tools:
            converted_tools = tools_to_openai(tools)
            request_kwargs["tools"] = converted_tools
            logger.debug(f"Added {len(converted_tools)} tools to request")

        # Merge extra kwargs
        request_kwargs.update(self.extra_kwargs)

        logger.debug(f"Generating with model {self.model}")
        response = await self.client.chat.completions.create(**request_kwargs)

        response_message = response.choices[0].message

        # Extract usage statistics
        if response.usage:
            self._last_usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
            logger.debug(
                f"Usage: {response.usage.prompt_tokens} prompt tokens, "
                f"{response.usage.completion_tokens} completion tokens"
            )

        # Parse tool calls if present
        tool_calls = None
        if hasattr(response_message, "tool_calls") and response_message.tool_calls:
            logger.debug(f"Assistant requested {len(response_message.tool_calls)} tool calls")
            tool_calls = convert_tool_calls_from_openai(response_message.tool_calls)

        # Always return AssistantMessage
        content = response_message.content or ""
        logger.debug(f"Generated {len(content)} characters")
        return AssistantMessage(content=content, tool_calls=tool_calls)
