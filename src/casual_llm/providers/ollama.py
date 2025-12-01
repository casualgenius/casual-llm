"""
Ollama LLM provider using the official ollama library.
"""

from __future__ import annotations

import logging
import asyncio
import uuid
from typing import Any, Literal
from ollama import AsyncClient
from ollama import ResponseError, RequestError

from casual_llm.messages import ChatMessage, AssistantMessage
from casual_llm.tools import Tool
from casual_llm.tool_converters import tools_to_ollama
from casual_llm.message_converters import (
    convert_messages_to_ollama,
    convert_tool_calls_from_ollama,
)

logger = logging.getLogger(__name__)


class OllamaProvider:
    """
    Ollama LLM provider with configurable retry logic and metrics.

    Uses the official ollama Python library for communication.
    Supports both JSON and text response formats.
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float | None = None,
        timeout: float = 60.0,
        max_retries: int = 0,
        enable_metrics: bool = False,
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Model name (e.g., "qwen2.5:7b-instruct")
            host: Ollama server URL (e.g., "http://localhost:11434")
            temperature: Temperature for generation (0.0-1.0, optional - uses Ollama default if not set)
            timeout: HTTP request timeout in seconds
            max_retries: Number of retries for transient failures (default: 0)
            enable_metrics: Track success/failure metrics (default: False)
        """
        self.model = model
        self.host = host.rstrip("/")  # Remove trailing slashes
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_metrics = enable_metrics

        # Create async client
        self.client = AsyncClient(host=self.host, timeout=timeout)

        # Metrics tracking
        self.success_count = 0
        self.failure_count = 0

        logger.info(
            f"OllamaProvider initialized: model={model}, "
            f"host={host}, max_retries={max_retries}"
        )

    def get_metrics(self) -> dict[str, int | float]:
        """
        Get performance metrics.

        Returns:
            Dictionary with success/failure counts and success rate
        """
        if not self.enable_metrics:
            return {}

        total = self.success_count + self.failure_count
        success_rate = (self.success_count / total * 100) if total > 0 else 0.0

        return {
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_calls": total,
            "success_rate_percent": round(success_rate, 2),
        }

    async def chat(
        self,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> AssistantMessage:
        """
        Generate a chat response using Ollama.

        Args:
            messages: Conversation messages (ChatMessage format)
            response_format: "json" for structured output, "text" for plain text
            max_tokens: Maximum tokens to generate (optional)
            tools: List of tools available for the LLM to call (optional)
            temperature: Temperature for this request (optional, overrides instance temperature)

        Returns:
            AssistantMessage with content and optional tool_calls

        Raises:
            ResponseError: If the request could not be fulfilled
            RequestError: If the request was invalid
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

        # Add format for JSON responses
        if response_format == "json":
            request_kwargs["format"] = "json"

        # Add tools if provided
        converted_tools = None
        if tools:
            converted_tools = tools_to_ollama(tools)
            request_kwargs["tools"] = converted_tools
            logger.debug(f"Added {len(converted_tools)} tools to request")

        # Execute with retry logic
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Calling Ollama (attempt {attempt + 1}/{self.max_retries + 1})")
                response = await self.client.chat(**request_kwargs)

                # Success - update metrics
                if self.enable_metrics:
                    self.success_count += 1

                if attempt > 0:
                    logger.info(f"Request succeeded on attempt {attempt + 1}")

                # Extract message from response
                response_message = response.message

                # Parse tool calls if present
                tool_calls = None
                if response_message.tool_calls:
                    logger.debug(f"Assistant requested {len(response_message.tool_calls)} tool calls")
                    # Convert ollama tool calls to our format
                    tool_calls_dicts = []
                    for tc in response_message.tool_calls:
                        # Generate a unique ID if Ollama doesn't provide one
                        tool_call_id = getattr(tc, "id", None)
                        if not tool_call_id:
                            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
                            logger.debug(f"Generated tool call ID: {tool_call_id}")

                        tool_calls_dicts.append({
                            "id": tool_call_id,
                            "type": getattr(tc, "type", "function"),
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        })
                    tool_calls = convert_tool_calls_from_ollama(tool_calls_dicts)

                # Always return AssistantMessage
                content = response_message.content.strip() if response_message.content else ""
                logger.debug(f"Generated {len(content)} characters")
                return AssistantMessage(
                    content=content,
                    tool_calls=tool_calls
                )

            except (ConnectionError, TimeoutError) as e:
                # Transient errors - retry
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {self.max_retries + 1} attempts: {e}")

            except (ResponseError, RequestError) as e:
                # Non-retriable errors from ollama library
                last_exception = e
                logger.error(f"Ollama error: {e}")
                break  # Don't retry

        # All attempts failed
        if self.enable_metrics:
            self.failure_count += 1

        if last_exception:
            raise last_exception
        raise RuntimeError("Unknown error occurred")
