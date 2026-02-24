"""
Ollama LLM client using the official ollama library.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

try:
    from ollama import AsyncClient
except ImportError:
    AsyncClient = None  # type: ignore[assignment, misc]

from pydantic import BaseModel

from casual_llm.config import ChatOptions
from casual_llm.messages import ChatMessage, AssistantMessage, StreamChunk
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
        if AsyncClient is None:
            raise ImportError(
                "OllamaClient requires the 'ollama' package. "
                "Install it with: pip install 'casual-llm[ollama]'"
            )

        self.host = host.rstrip("/")  # Remove trailing slashes
        self.timeout = timeout

        # Create async client
        self.client = AsyncClient(host=self.host, timeout=timeout)

        logger.info("OllamaClient initialized: host=%s", host)

    async def _build_request_kwargs(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build the request kwargs dict for the Ollama API call."""
        # Convert messages to Ollama format using converter (async for image support)
        chat_messages = await convert_messages_to_ollama(messages)
        logger.debug("Converted %d messages to Ollama format", len(messages))

        # Build Ollama options dict
        ollama_options: dict[str, Any] = {}
        if options.temperature is not None:
            ollama_options["temperature"] = options.temperature
        if options.max_tokens is not None:
            ollama_options["num_predict"] = options.max_tokens
        if options.top_p is not None:
            ollama_options["top_p"] = options.top_p
        if options.top_k is not None:
            ollama_options["top_k"] = options.top_k
        if options.frequency_penalty is not None:
            ollama_options["frequency_penalty"] = options.frequency_penalty
        if options.presence_penalty is not None:
            ollama_options["presence_penalty"] = options.presence_penalty
        if options.seed is not None:
            ollama_options["seed"] = options.seed
        if options.stop is not None:
            ollama_options["stop"] = options.stop

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "stream": stream,
            "options": ollama_options,
        }

        # Handle response_format
        if options.response_format == "json":
            request_kwargs["format"] = "json"
        elif isinstance(options.response_format, type) and issubclass(
            options.response_format, BaseModel
        ):
            schema = options.response_format.model_json_schema()
            request_kwargs["format"] = schema
            logger.debug(
                "Using JSON Schema from Pydantic model: %s",
                options.response_format.__name__,
            )

        # Add tools if provided
        if options.tools:
            converted_tools = tools_to_ollama(options.tools)
            request_kwargs["tools"] = converted_tools
            logger.debug("Added %d tools to request", len(converted_tools))

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
        Generate a chat response using Ollama.

        This is an internal method typically called by the Model class.

        Args:
            model: Model name (e.g., "qwen2.5:7b-instruct")
            messages: Conversation messages (ChatMessage format)
            options: Chat options controlling response format, sampling, tools, etc.

        Returns:
            Tuple of (AssistantMessage, Usage or None)

        Raises:
            ResponseError: If the request could not be fulfilled
            RequestError: If the request was invalid
        """
        request_kwargs = await self._build_request_kwargs(model, messages, options)

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
        logger.debug(
            "Usage: %d prompt tokens, %d completion tokens", prompt_tokens, completion_tokens
        )

        # Parse tool calls if present
        tool_calls = None
        if response_message.tool_calls:
            logger.debug("Assistant requested %d tool calls", len(response_message.tool_calls))
            tool_calls = convert_tool_calls_from_ollama(response_message.tool_calls)

        # Return AssistantMessage and Usage
        content = response_message.content.strip() if response_message.content else ""
        logger.debug("Generated %d characters", len(content))
        return AssistantMessage(content=content, tool_calls=tool_calls), usage

    async def _stream(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a chat response from Ollama.

        This is an internal method typically called by the Model class.

        Args:
            model: Model name (e.g., "qwen2.5:7b-instruct")
            messages: Conversation messages (ChatMessage format)
            options: Chat options controlling response format, sampling, tools, etc.

        Yields:
            StreamChunk objects containing content fragments as tokens are generated.

        Raises:
            ResponseError: If the request could not be fulfilled
            RequestError: If the request was invalid
        """
        request_kwargs = await self._build_request_kwargs(model, messages, options, stream=True)

        logger.debug("Starting stream with model %s", model)
        stream = await self.client.chat(**request_kwargs)

        usage: Usage | None = None
        usage_emitted = False
        async for chunk in stream:
            is_done = getattr(chunk, "done", False)

            # Ollama's final chunk (done=True) contains usage counters
            if is_done:
                prompt_tokens = getattr(chunk, "prompt_eval_count", 0) or 0
                completion_tokens = getattr(chunk, "eval_count", 0) or 0
                if prompt_tokens or completion_tokens:
                    usage = Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )

            # Extract content from the message if present
            if chunk.message and chunk.message.content:
                content = chunk.message.content
                finish_reason = "stop" if is_done else None
                chunk_usage = usage if is_done else None
                if chunk_usage:
                    usage_emitted = True
                yield StreamChunk(content=content, finish_reason=finish_reason, usage=chunk_usage)

        # If usage wasn't attached to a content chunk, emit it separately
        if usage is not None and not usage_emitted:
            yield StreamChunk(content="", finish_reason="stop", usage=usage)

        logger.debug("Stream completed")
