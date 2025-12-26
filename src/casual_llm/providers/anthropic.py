"""
Anthropic LLM provider for Claude models.
"""

from __future__ import annotations

import logging
from typing import Literal, Any, AsyncIterator
from anthropic import AsyncAnthropic
from pydantic import BaseModel

from casual_llm.messages import ChatMessage, AssistantMessage, StreamChunk
from casual_llm.tools import Tool
from casual_llm.usage import Usage
from casual_llm.tool_converters.anthropic import tools_to_anthropic
from casual_llm.message_converters.anthropic import (
    convert_messages_to_anthropic,
    extract_system_message,
    convert_tool_calls_from_anthropic,
)

logger = logging.getLogger(__name__)

# Default max_tokens for Anthropic API (required parameter)
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider:
    """
    Anthropic LLM provider for Claude models.

    Supports Claude 3 (opus, sonnet, haiku), Claude 3.5, and Claude 4 models.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        timeout: float = 60.0,
        extra_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize Anthropic provider.

        Args:
            model: Model name (e.g., "claude-3-haiku-20240307", "claude-3-5-sonnet-latest")
            api_key: API key (optional, can use ANTHROPIC_API_KEY env var)
            base_url: Base URL for API (optional, for custom endpoints)
            temperature: Temperature for generation (0.0-1.0, optional - uses Anthropic
                default if not set)
            timeout: HTTP request timeout in seconds
            extra_kwargs: Additional kwargs to pass to client.messages.create()
        """
        client_kwargs: dict[str, Any] = {"timeout": timeout}

        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = AsyncAnthropic(**client_kwargs)
        self.model = model
        self.temperature = temperature
        self.extra_kwargs = extra_kwargs or {}

        # Usage tracking
        self._last_usage: Usage | None = None

        logger.info(
            f"AnthropicProvider initialized: model={model}, "
            f"base_url={base_url or 'default'}"
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
        Generate a chat response using Anthropic API.

        Args:
            messages: Conversation messages (ChatMessage format)
            response_format: "json" for JSON output, "text" for plain text, or a Pydantic
                BaseModel class for JSON Schema-based structured output. When a Pydantic
                model is provided, the LLM will be instructed to return JSON matching the
                schema.
            max_tokens: Maximum tokens to generate (default: 4096)
            tools: List of tools available for the LLM to call (optional)
            temperature: Temperature for this request (optional, overrides instance temperature)

        Returns:
            AssistantMessage with content and optional tool_calls

        Raises:
            anthropic.APIError: If request fails

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
        # Extract system message (Anthropic uses separate system parameter)
        system_content = extract_system_message(messages)

        # Convert messages to Anthropic format (excludes system messages)
        anthropic_messages = convert_messages_to_anthropic(messages)
        logger.debug(f"Converted {len(messages)} messages to Anthropic format")

        # Use provided temperature or fall back to instance temperature
        temp = temperature if temperature is not None else self.temperature

        # Build request kwargs - max_tokens is required by Anthropic
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
        }

        # Add system message if present
        if system_content:
            request_kwargs["system"] = system_content

        # Only add temperature if specified
        if temp is not None:
            request_kwargs["temperature"] = temp

        # Handle response_format: "json", "text", or Pydantic model class
        # Anthropic doesn't have native JSON mode like OpenAI, but we can use
        # system prompts to guide the model
        if response_format == "json":
            # Add JSON instruction to system prompt
            json_instruction = "You must respond with valid JSON only. No other text."
            if system_content:
                request_kwargs["system"] = f"{system_content}\n\n{json_instruction}"
            else:
                request_kwargs["system"] = json_instruction
            logger.debug("Added JSON response format instruction to system prompt")
        elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
            # Extract JSON Schema from Pydantic model and add to system prompt
            schema = response_format.model_json_schema()
            schema_instruction = (
                f"You must respond with valid JSON matching this schema:\n"
                f"{schema}\n\n"
                f"Respond with JSON only. No other text."
            )
            if system_content:
                request_kwargs["system"] = f"{system_content}\n\n{schema_instruction}"
            else:
                request_kwargs["system"] = schema_instruction
            logger.debug(f"Using JSON Schema from Pydantic model: {response_format.__name__}")

        # Add tools if provided
        if tools:
            converted_tools = tools_to_anthropic(tools)
            request_kwargs["tools"] = converted_tools
            logger.debug(f"Added {len(converted_tools)} tools to request")

        # Merge extra kwargs
        request_kwargs.update(self.extra_kwargs)

        logger.debug(f"Generating with model {self.model}")
        response = await self.client.messages.create(**request_kwargs)

        # Extract usage statistics (Anthropic uses input_tokens/output_tokens)
        if response.usage:
            self._last_usage = Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )
            logger.debug(
                f"Usage: {response.usage.input_tokens} input tokens, "
                f"{response.usage.output_tokens} output tokens"
            )

        # Parse response content blocks
        text_content = ""
        tool_use_blocks = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_use_blocks.append(block)

        # Convert tool calls if present
        tool_calls = None
        if tool_use_blocks:
            logger.debug(f"Assistant requested {len(tool_use_blocks)} tool calls")
            tool_calls = convert_tool_calls_from_anthropic(tool_use_blocks)

        logger.debug(f"Generated {len(text_content)} characters")
        return AssistantMessage(content=text_content, tool_calls=tool_calls)

    async def stream(
        self,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a chat response from Anthropic API.

        This method yields response chunks in real-time as they are generated,
        enabling progressive display in chat interfaces.

        Args:
            messages: Conversation messages (ChatMessage format)
            response_format: "json" for JSON output, "text" for plain text, or a Pydantic
                BaseModel class for JSON Schema-based structured output. When a Pydantic
                model is provided, the LLM will be instructed to return JSON matching the
                schema.
            max_tokens: Maximum tokens to generate (default: 4096)
            tools: List of tools available for the LLM to call (optional, may not work
                with all streaming scenarios)
            temperature: Temperature for this request (optional, overrides instance temperature)

        Yields:
            StreamChunk objects containing content fragments as tokens are generated.

        Raises:
            anthropic.APIError: If request fails

        Examples:
            >>> async for chunk in provider.stream([UserMessage(content="Hello")]):
            ...     print(chunk.content, end="", flush=True)
        """
        # Extract system message (Anthropic uses separate system parameter)
        system_content = extract_system_message(messages)

        # Convert messages to Anthropic format (excludes system messages)
        anthropic_messages = convert_messages_to_anthropic(messages)
        logger.debug(f"Converted {len(messages)} messages to Anthropic format for streaming")

        # Use provided temperature or fall back to instance temperature
        temp = temperature if temperature is not None else self.temperature

        # Build request kwargs - max_tokens is required by Anthropic
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
        }

        # Add system message if present
        if system_content:
            request_kwargs["system"] = system_content

        # Only add temperature if specified
        if temp is not None:
            request_kwargs["temperature"] = temp

        # Handle response_format: "json", "text", or Pydantic model class
        if response_format == "json":
            json_instruction = "You must respond with valid JSON only. No other text."
            if system_content:
                request_kwargs["system"] = f"{system_content}\n\n{json_instruction}"
            else:
                request_kwargs["system"] = json_instruction
            logger.debug("Added JSON response format instruction to system prompt")
        elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
            schema = response_format.model_json_schema()
            schema_instruction = (
                f"You must respond with valid JSON matching this schema:\n"
                f"{schema}\n\n"
                f"Respond with JSON only. No other text."
            )
            if system_content:
                request_kwargs["system"] = f"{system_content}\n\n{schema_instruction}"
            else:
                request_kwargs["system"] = schema_instruction
            logger.debug(f"Using JSON Schema from Pydantic model: {response_format.__name__}")

        # Add tools if provided
        if tools:
            converted_tools = tools_to_anthropic(tools)
            request_kwargs["tools"] = converted_tools
            logger.debug(f"Added {len(converted_tools)} tools to streaming request")

        # Merge extra kwargs
        request_kwargs.update(self.extra_kwargs)

        logger.debug(f"Starting stream with model {self.model}")

        async with self.client.messages.stream(**request_kwargs) as stream:
            async for event in stream:
                # Handle content block delta events
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield StreamChunk(content=event.delta.text, finish_reason=None)
                # Handle message stop event
                elif event.type == "message_stop":
                    # Anthropic uses stop_reason in the final message
                    pass

        logger.debug("Stream completed")
