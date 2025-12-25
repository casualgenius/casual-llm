"""
Ollama LLM provider using the official ollama library.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

import httpx
from ollama import AsyncClient, RequestError, ResponseError
from pydantic import BaseModel

from casual_llm.config import RetryConfig
from casual_llm.messages import ChatMessage, AssistantMessage
from casual_llm.tools import Tool
from casual_llm.usage import Usage
from casual_llm.tool_converters import tools_to_ollama
from casual_llm.message_converters import (
    convert_messages_to_ollama,
    convert_tool_calls_from_ollama,
)

logger = logging.getLogger(__name__)

# Exceptions that should trigger retry with exponential backoff
# httpx.ConnectError: Connection failed (server down, network issues)
# httpx.TimeoutException: Request timed out
# ResponseError: Server-side errors (we'll check for 5xx status codes)
RETRYABLE_EXCEPTIONS = (httpx.ConnectError, httpx.TimeoutException, ResponseError)


class OllamaProvider:
    """
    Ollama LLM provider.

    Uses the official ollama Python library for communication.
    Supports both JSON and text response formats.
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float | None = None,
        timeout: float = 60.0,
        retry_config: RetryConfig | None = None,
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Model name (e.g., "qwen2.5:7b-instruct")
            host: Ollama server URL (e.g., "http://localhost:11434")
            temperature: Temperature for generation (0.0-1.0, optional - uses Ollama default if not set)
            timeout: HTTP request timeout in seconds
            retry_config: Configuration for retry behavior with exponential backoff (optional)
        """
        self.model = model
        self.host = host.rstrip("/")  # Remove trailing slashes
        self.temperature = temperature
        self.timeout = timeout
        self.retry_config = retry_config

        # Create async client
        self.client = AsyncClient(host=self.host, timeout=timeout)

        # Usage tracking
        self._last_usage: Usage | None = None

        logger.info(f"OllamaProvider initialized: model={model}, host={host}")

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
        Generate a chat response using Ollama.

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
            ResponseError: If the request could not be fulfilled
            RequestError: If the request was invalid

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
        # Convert messages to Ollama format using converter
        chat_messages = convert_messages_to_ollama(messages)
        logger.debug(f"Converted {len(messages)} messages to Ollama format")

        # Use provided temperature or fall back to instance temperature
        temp = temperature if temperature is not None else self.temperature

        # Build options
        options: dict[str, Any] = {}
        if temp is not None:
            options["temperature"] = temp
        if max_tokens:
            options["num_predict"] = max_tokens

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self.model,
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
            logger.debug(f"Using JSON Schema from Pydantic model: {response_format.__name__}")
        # "text" is the default - no format parameter needed

        # Add tools if provided
        if tools:
            converted_tools = tools_to_ollama(tools)
            request_kwargs["tools"] = converted_tools
            logger.debug(f"Added {len(converted_tools)} tools to request")

        logger.debug(f"Generating with model {self.model}")

        # Determine number of attempts based on retry config
        max_attempts = (
            self.retry_config.max_attempts
            if self.retry_config and self.retry_config.max_attempts > 0
            else 1
        )

        last_exception: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = await self.client.chat(**request_kwargs)
                break  # Success, exit retry loop
            except RETRYABLE_EXCEPTIONS as e:
                # For ResponseError, only retry on server errors (5xx)
                # ResponseError doesn't have a status_code attribute directly,
                # but we can check the error message or treat all ResponseErrors as retryable
                # since they typically indicate server-side issues
                last_exception = e

                # If this was the last attempt, raise the exception
                if attempt >= max_attempts:
                    logger.warning(
                        f"All {max_attempts} attempts failed for model {self.model}"
                    )
                    raise

                # Calculate exponential backoff delay: backoff_factor^(attempt-1)
                # For attempt 1: delay = 1s, attempt 2: delay = 2s, attempt 3: delay = 4s (with factor=2.0)
                backoff_factor = (
                    self.retry_config.backoff_factor if self.retry_config else 2.0
                )
                delay = backoff_factor ** (attempt - 1)

                logger.warning(
                    f"Retry attempt {attempt}/{max_attempts} for model {self.model} "
                    f"after {delay:.1f}s delay due to {type(e).__name__}: {e}"
                )

                await asyncio.sleep(delay)
        else:
            # This else clause is reached if the loop completes without break
            # This should only happen if max_attempts is 0, which we guard against
            if last_exception:
                raise last_exception

        # Extract message from response
        response_message = response.message

        # Extract usage statistics
        prompt_tokens = getattr(response, "prompt_eval_count", 0)
        completion_tokens = getattr(response, "eval_count", 0)
        self._last_usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        logger.debug(f"Usage: {prompt_tokens} prompt tokens, {completion_tokens} completion tokens")

        # Parse tool calls if present
        tool_calls = None
        if response_message.tool_calls:
            logger.debug(f"Assistant requested {len(response_message.tool_calls)} tool calls")
            # Convert ollama tool calls to our format
            # The converter handles ID generation and argument conversion
            tool_calls = convert_tool_calls_from_ollama(response_message.tool_calls)

        # Always return AssistantMessage
        content = response_message.content.strip() if response_message.content else ""
        logger.debug(f"Generated {len(content)} characters")
        return AssistantMessage(content=content, tool_calls=tool_calls)
