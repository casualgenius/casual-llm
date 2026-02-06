"""
Ollama LLM client using the official ollama library.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Literal
from ollama import AsyncClient
from pydantic import BaseModel

from casual_llm.messages import ChatMessage, AssistantMessage, StreamChunk
from casual_llm.tools import Tool
from casual_llm.usage import Usage
from casual_llm.tool_converters import tools_to_ollama
from casual_llm.message_converters import (
    convert_messages_to_ollama,
    convert_tool_calls_from_ollama,
)

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Ollama LLM client.

    Uses the official ollama Python library for communication.
    Manages the API connection - use with Model class for actual interactions.

    Examples:
        >>> from casual_llm import OllamaClient, Model, UserMessage
        >>>
        >>> # Create client (configured once)
        >>> client = OllamaClient(host="http://localhost:11434")
        >>>
        >>> # Create models using the client
        >>> llama = Model(client, name="llama3.1", temperature=0.7)
        >>> qwen = Model(client, name="qwen2.5:7b-instruct")
        >>>
        >>> # Use models
        >>> response = await llama.chat([UserMessage(content="Hello")])
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
    ):
        """
        Initialize Ollama client.

        Args:
            host: Ollama server URL (e.g., "http://localhost:11434")
            timeout: HTTP request timeout in seconds
        """
        self.host = host.rstrip("/")  # Remove trailing slashes
        self.timeout = timeout

        # Create async client
        self.client = AsyncClient(host=self.host, timeout=timeout)

        logger.info("OllamaClient initialized: host=%s", host)

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
        Generate a chat response using Ollama.

        This is an internal method typically called by the Model class.

        Args:
            model: Model name (e.g., "qwen2.5:7b-instruct")
            messages: Conversation messages (ChatMessage format)
            response_format: "json" for JSON output, "text" for plain text, or a Pydantic
                BaseModel class for JSON Schema-based structured output.
            max_tokens: Maximum tokens to generate (optional)
            tools: List of tools available for the LLM to call (optional)
            temperature: Temperature for this request (optional)

        Returns:
            Tuple of (AssistantMessage, Usage or None)

        Raises:
            ResponseError: If the request could not be fulfilled
            RequestError: If the request was invalid
        """
        # Convert messages to Ollama format using converter (async for image support)
        chat_messages = await convert_messages_to_ollama(messages)
        logger.debug("Converted %d messages to Ollama format", len(messages))

        # Build options
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens:
            options["num_predict"] = max_tokens

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "stream": False,
            "options": options,
        }

        # Handle response_format: "json", "text", or Pydantic model class
        if response_format == "json":
            request_kwargs["format"] = "json"
        elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
            # Extract JSON Schema from Pydantic model and pass directly to format
            schema = response_format.model_json_schema()
            request_kwargs["format"] = schema
            logger.debug("Using JSON Schema from Pydantic model: %s", response_format.__name__)
        # "text" is the default - no format parameter needed

        # Add tools if provided
        if tools:
            converted_tools = tools_to_ollama(tools)
            request_kwargs["tools"] = converted_tools
            logger.debug("Added %d tools to request", len(converted_tools))

        logger.debug("Generating with model %s", model)
        response = await self.client.chat(**request_kwargs)

        # Extract message from response
        response_message = response.message

        # Extract usage statistics
        prompt_tokens = getattr(response, "prompt_eval_count", 0)
        completion_tokens = getattr(response, "eval_count", 0)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        logger.debug("Usage: %d prompt tokens, %d completion tokens", prompt_tokens, completion_tokens)

        # Parse tool calls if present
        tool_calls = None
        if response_message.tool_calls:
            logger.debug("Assistant requested %d tool calls", len(response_message.tool_calls))
            # Convert ollama tool calls to our format
            # The converter handles ID generation and argument conversion
            tool_calls = convert_tool_calls_from_ollama(response_message.tool_calls)

        # Return AssistantMessage and Usage
        content = response_message.content.strip() if response_message.content else ""
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
        Stream a chat response from Ollama.

        This is an internal method typically called by the Model class.

        Args:
            model: Model name (e.g., "qwen2.5:7b-instruct")
            messages: Conversation messages (ChatMessage format)
            response_format: "json" for JSON output, "text" for plain text, or a Pydantic
                BaseModel class for JSON Schema-based structured output.
            max_tokens: Maximum tokens to generate (optional)
            tools: List of tools available for the LLM to call (optional)
            temperature: Temperature for this request (optional)

        Yields:
            StreamChunk objects containing content fragments as tokens are generated.

        Raises:
            ResponseError: If the request could not be fulfilled
            RequestError: If the request was invalid
        """
        # Convert messages to Ollama format using converter (async for image support)
        chat_messages = await convert_messages_to_ollama(messages)
        logger.debug("Converted %d messages to Ollama format for streaming", len(messages))

        # Build options
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens:
            options["num_predict"] = max_tokens

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "stream": True,
            "options": options,
        }

        # Handle response_format: "json", "text", or Pydantic model class
        if response_format == "json":
            request_kwargs["format"] = "json"
        elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
            # Extract JSON Schema from Pydantic model and pass directly to format
            schema = response_format.model_json_schema()
            request_kwargs["format"] = schema
            logger.debug("Using JSON Schema from Pydantic model: %s", response_format.__name__)
        # "text" is the default - no format parameter needed

        # Add tools if provided
        if tools:
            converted_tools = tools_to_ollama(tools)
            request_kwargs["tools"] = converted_tools
            logger.debug("Added %d tools to streaming request", len(converted_tools))

        logger.debug("Starting stream with model %s", model)
        stream = await self.client.chat(**request_kwargs)

        async for chunk in stream:
            # Extract content from the message if present
            if chunk.message and chunk.message.content:
                content = chunk.message.content
                # Ollama uses 'done' field to indicate completion
                finish_reason = "stop" if getattr(chunk, "done", False) else None
                yield StreamChunk(content=content, finish_reason=finish_reason)

        logger.debug("Stream completed")
