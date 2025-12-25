"""
Anthropic LLM provider (Claude models with vision support).
"""

from __future__ import annotations

import logging
from typing import Literal, Any, AsyncIterator, TYPE_CHECKING
from pydantic import BaseModel

from casual_llm.messages import ChatMessage, AssistantMessage, StreamChunk
from casual_llm.tools import Tool
from casual_llm.usage import Usage
from casual_llm.message_converters import (
    convert_messages_to_anthropic,
    convert_tool_calls_from_anthropic,
)

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from anthropic.types import ToolUseBlock

logger = logging.getLogger(__name__)


def _tool_to_anthropic(tool: Tool) -> dict[str, Any]:
    """
    Convert a casual-llm Tool to Anthropic tool format.

    Args:
        tool: Tool to convert

    Returns:
        Dictionary in Anthropic's tool format
    """
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": {
            "type": "object",
            "properties": {
                name: param.model_dump(exclude_none=True)
                for name, param in tool.parameters.items()
            },
            "required": tool.required,
        },
    }


def _tools_to_anthropic(tools: list[Tool]) -> list[dict[str, Any]]:
    """
    Convert multiple casual-llm Tools to Anthropic format.

    Args:
        tools: List of tools to convert

    Returns:
        List of tool dictionaries in Anthropic format
    """
    logger.debug(f"Converting {len(tools)} tools to Anthropic format")
    return [_tool_to_anthropic(tool) for tool in tools]


class AnthropicProvider:
    """
    Anthropic LLM provider for Claude models.

    Supports Claude 3 family models including vision capabilities.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        timeout: float = 60.0,
        max_tokens: int = 4096,
        extra_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize Anthropic provider.

        Args:
            model: Model name (e.g., "claude-3-5-sonnet-20241022", "claude-3-opus-20240229")
            api_key: API key (optional, can use ANTHROPIC_API_KEY env var)
            base_url: Base URL for API (optional, for custom endpoints)
            temperature: Temperature for generation (0.0-1.0, optional - uses Anthropic
                default if not set)
            timeout: HTTP request timeout in seconds
            max_tokens: Default max tokens for responses (Anthropic requires this)
            extra_kwargs: Additional kwargs to pass to client.messages.create()
        """
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package. "
                "Install it with: pip install casual-llm[anthropic]"
            )

        client_kwargs: dict[str, Any] = {"timeout": timeout}

        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client: AsyncAnthropic = AsyncAnthropic(**client_kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
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
                BaseModel class for JSON Schema-based structured output. Note: Anthropic
                handles JSON through system prompts, not native response_format.
            max_tokens: Maximum tokens to generate (optional, uses instance default)
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
        # Convert messages to Anthropic format (returns messages and system prompt separately)
        # This is async because URL images need to be fetched and converted to base64
        anthropic_messages, system_prompt = await convert_messages_to_anthropic(messages)
        logger.debug(f"Converted {len(messages)} messages to Anthropic format")

        # Use provided temperature or fall back to instance temperature
        temp = temperature if temperature is not None else self.temperature

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or self.max_tokens,
        }

        # Add system prompt if present
        if system_prompt:
            # Handle JSON response format by augmenting system prompt
            if response_format == "json":
                system_prompt = (
                    f"{system_prompt}\n\n"
                    "IMPORTANT: You must respond with valid JSON only. "
                    "Do not include any text before or after the JSON."
                )
            elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
                schema = response_format.model_json_schema()
                system_prompt = (
                    f"{system_prompt}\n\n"
                    f"IMPORTANT: You must respond with valid JSON matching this schema:\n"
                    f"{schema}\n"
                    "Do not include any text before or after the JSON."
                )
            request_kwargs["system"] = system_prompt
        elif response_format == "json":
            request_kwargs["system"] = (
                "IMPORTANT: You must respond with valid JSON only. "
                "Do not include any text before or after the JSON."
            )
        elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
            schema = response_format.model_json_schema()
            request_kwargs["system"] = (
                f"IMPORTANT: You must respond with valid JSON matching this schema:\n"
                f"{schema}\n"
                "Do not include any text before or after the JSON."
            )
            logger.debug(f"Using JSON Schema from Pydantic model: {response_format.__name__}")

        # Only add temperature if specified
        if temp is not None:
            request_kwargs["temperature"] = temp

        # Add tools if provided
        if tools:
            converted_tools = _tools_to_anthropic(tools)
            request_kwargs["tools"] = converted_tools
            logger.debug(f"Added {len(converted_tools)} tools to request")

        # Merge extra kwargs
        request_kwargs.update(self.extra_kwargs)

        logger.debug(f"Generating with model {self.model}")
        response = await self.client.messages.create(**request_kwargs)

        # Extract usage statistics
        if response.usage:
            self._last_usage = Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )
            logger.debug(
                f"Usage: {response.usage.input_tokens} prompt tokens, "
                f"{response.usage.output_tokens} completion tokens"
            )

        # Process content blocks
        content_parts: list[str] = []
        tool_use_blocks: list[ToolUseBlock] = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_use_blocks.append(block)

        # Parse tool calls if present
        tool_calls = None
        if tool_use_blocks:
            logger.debug(f"Assistant requested {len(tool_use_blocks)} tool calls")
            tool_calls = convert_tool_calls_from_anthropic(tool_use_blocks)

        # Combine text content
        content = "".join(content_parts)
        logger.debug(f"Generated {len(content)} characters")
        return AssistantMessage(content=content, tool_calls=tool_calls)

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
                BaseModel class for JSON Schema-based structured output.
            max_tokens: Maximum tokens to generate (optional, uses instance default)
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
        # Convert messages to Anthropic format
        # This is async because URL images need to be fetched and converted to base64
        anthropic_messages, system_prompt = await convert_messages_to_anthropic(messages)
        logger.debug(f"Converted {len(messages)} messages to Anthropic format for streaming")

        # Use provided temperature or fall back to instance temperature
        temp = temperature if temperature is not None else self.temperature

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or self.max_tokens,
        }

        # Add system prompt if present
        if system_prompt:
            if response_format == "json":
                system_prompt = (
                    f"{system_prompt}\n\n"
                    "IMPORTANT: You must respond with valid JSON only. "
                    "Do not include any text before or after the JSON."
                )
            elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
                schema = response_format.model_json_schema()
                system_prompt = (
                    f"{system_prompt}\n\n"
                    f"IMPORTANT: You must respond with valid JSON matching this schema:\n"
                    f"{schema}\n"
                    "Do not include any text before or after the JSON."
                )
            request_kwargs["system"] = system_prompt
        elif response_format == "json":
            request_kwargs["system"] = (
                "IMPORTANT: You must respond with valid JSON only. "
                "Do not include any text before or after the JSON."
            )
        elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
            schema = response_format.model_json_schema()
            request_kwargs["system"] = (
                f"IMPORTANT: You must respond with valid JSON matching this schema:\n"
                f"{schema}\n"
                "Do not include any text before or after the JSON."
            )
            logger.debug(f"Using JSON Schema from Pydantic model: {response_format.__name__}")

        # Only add temperature if specified
        if temp is not None:
            request_kwargs["temperature"] = temp

        # Add tools if provided
        if tools:
            converted_tools = _tools_to_anthropic(tools)
            request_kwargs["tools"] = converted_tools
            logger.debug(f"Added {len(converted_tools)} tools to streaming request")

        # Merge extra kwargs
        request_kwargs.update(self.extra_kwargs)

        logger.debug(f"Starting stream with model {self.model}")

        async with self.client.messages.stream(**request_kwargs) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(content=text, finish_reason=None)

        logger.debug("Stream completed")


__all__ = ["AnthropicProvider"]
