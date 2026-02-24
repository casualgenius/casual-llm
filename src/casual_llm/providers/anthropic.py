"""
Anthropic LLM client for Claude models.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator
from anthropic import AsyncAnthropic
from pydantic import BaseModel

from casual_llm.config import ChatOptions
from casual_llm.messages import ChatMessage, AssistantMessage, StreamChunk
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


class AnthropicClient:
    """
    Anthropic LLM client for Claude models.

    Supports Claude 3 (opus, sonnet, haiku), Claude 3.5, and Claude 4 models.
    Manages the API connection - use with Model class for actual interactions.

    Examples:
        >>> from casual_llm import AnthropicClient, Model, UserMessage
        >>>
        >>> # Create client (configured once)
        >>> client = AnthropicClient(api_key="...")
        >>>
        >>> # Create models using the client
        >>> claude_sonnet = Model(client, name="claude-3-5-sonnet-latest", temperature=0.7)
        >>> claude_haiku = Model(client, name="claude-3-haiku-20240307")
        >>>
        >>> # Use models
        >>> response = await claude_sonnet.chat([UserMessage(content="Hello")])
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: API key (optional, can use ANTHROPIC_API_KEY env var)
            base_url: Base URL for API (optional, for custom endpoints)
            timeout: HTTP request timeout in seconds
        """
        client_kwargs: dict[str, Any] = {"timeout": timeout}

        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = AsyncAnthropic(**client_kwargs)

        logger.info("AnthropicClient initialized: base_url=%s", base_url or "default")

    def _build_request_kwargs(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> dict[str, Any]:
        """Build the request kwargs dict for the Anthropic API call."""
        # Extract system message (Anthropic uses separate system parameter)
        system_content = extract_system_message(messages)

        # Convert messages to Anthropic format (excludes system messages)
        anthropic_messages = convert_messages_to_anthropic(messages)
        logger.debug("Converted %d messages to Anthropic format", len(messages))

        # Build request kwargs - max_tokens is required by Anthropic
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": options.max_tokens or DEFAULT_MAX_TOKENS,
        }

        # Add system message if present
        if system_content:
            request_kwargs["system"] = system_content

        if options.temperature is not None:
            request_kwargs["temperature"] = options.temperature

        if options.top_p is not None:
            request_kwargs["top_p"] = options.top_p

        if options.top_k is not None:
            request_kwargs["top_k"] = options.top_k

        if options.stop is not None:
            request_kwargs["stop_sequences"] = options.stop

        # Handle response_format via system prompt instructions
        if options.response_format == "json":
            json_instruction = "You must respond with valid JSON only. No other text."
            if system_content:
                request_kwargs["system"] = f"{system_content}\n\n{json_instruction}"
            else:
                request_kwargs["system"] = json_instruction
            logger.debug("Added JSON response format instruction to system prompt")
        elif isinstance(options.response_format, type) and issubclass(
            options.response_format, BaseModel
        ):
            schema = options.response_format.model_json_schema()
            schema_instruction = (
                f"You must respond with valid JSON matching this schema:\n"
                f"{schema}\n\n"
                f"Respond with JSON only. No other text."
            )
            if system_content:
                request_kwargs["system"] = f"{system_content}\n\n{schema_instruction}"
            else:
                request_kwargs["system"] = schema_instruction
            logger.debug(
                "Using JSON Schema from Pydantic model: %s",
                options.response_format.__name__,
            )

        # Add tools if provided
        if options.tools:
            converted_tools = tools_to_anthropic(options.tools)
            request_kwargs["tools"] = converted_tools
            logger.debug("Added %d tools to request", len(converted_tools))

        # Handle tool_choice
        if options.tool_choice is not None and options.tools:
            if options.tool_choice == "auto":
                request_kwargs["tool_choice"] = {"type": "auto"}
            elif options.tool_choice == "required":
                request_kwargs["tool_choice"] = {"type": "any"}
            elif options.tool_choice == "none":
                # Anthropic: remove tools to prevent use
                request_kwargs.pop("tools", None)
            else:
                # Treat as a specific tool name
                request_kwargs["tool_choice"] = {
                    "type": "tool",
                    "name": options.tool_choice,
                }

        # Merge extra kwargs (provider-specific pass-through)
        request_kwargs.update(options.extra)

        return request_kwargs

    async def _chat(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> tuple[AssistantMessage, Usage | None]:
        """
        Generate a chat response using Anthropic API.

        This is an internal method typically called by the Model class.

        Args:
            model: Model name (e.g., "claude-3-haiku-20240307", "claude-3-5-sonnet-latest")
            messages: Conversation messages (ChatMessage format)
            options: Chat options controlling response format, sampling, tools, etc.

        Returns:
            Tuple of (AssistantMessage, Usage or None)

        Raises:
            anthropic.APIError: If request fails
        """
        request_kwargs = self._build_request_kwargs(model, messages, options)

        logger.debug("Generating with model %s", model)
        response = await self.client.messages.create(**request_kwargs)

        # Extract usage statistics (Anthropic uses input_tokens/output_tokens)
        usage: Usage | None = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )
            logger.debug(
                "Usage: %d input tokens, %d output tokens",
                response.usage.input_tokens,
                response.usage.output_tokens,
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
            logger.debug("Assistant requested %d tool calls", len(tool_use_blocks))
            tool_calls = convert_tool_calls_from_anthropic(tool_use_blocks)

        logger.debug("Generated %d characters", len(text_content))
        return AssistantMessage(content=text_content, tool_calls=tool_calls), usage

    async def _stream(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a chat response from Anthropic API.

        This is an internal method typically called by the Model class.

        Args:
            model: Model name (e.g., "claude-3-haiku-20240307", "claude-3-5-sonnet-latest")
            messages: Conversation messages (ChatMessage format)
            options: Chat options controlling response format, sampling, tools, etc.

        Yields:
            StreamChunk objects containing content fragments as tokens are generated.

        Raises:
            anthropic.APIError: If request fails
        """
        request_kwargs = self._build_request_kwargs(model, messages, options)

        logger.debug("Starting stream with model %s", model)

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
