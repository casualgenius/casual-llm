"""
Ollama LLM provider with retry logic and metrics tracking.
"""

import logging
import asyncio
from typing import List, Literal, Optional
import httpx

from casual_llm.messages import ChatMessage
from casual_llm.utils import extract_json_from_markdown

logger = logging.getLogger(__name__)


class OllamaProvider:
    """
    Ollama LLM provider with configurable retry logic and metrics.

    Supports both JSON and text response formats.
    """

    def __init__(
        self,
        model: str,
        endpoint: str,
        temperature: float = 0.2,
        timeout: float = 60.0,
        max_retries: int = 0,
        enable_metrics: bool = False,
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Model name (e.g., "qwen2.5:7b-instruct")
            endpoint: Ollama base URL (e.g., "http://localhost:11434")
            temperature: Temperature for generation (0.0-1.0)
            timeout: HTTP request timeout in seconds
            max_retries: Number of retries for transient failures (default: 0)
            enable_metrics: Track success/failure metrics (default: False)
        """
        self.model = model
        # Ensure endpoint doesn't have trailing slash and construct full API URL
        base_endpoint = endpoint.rstrip('/')
        self.endpoint = f"{base_endpoint}/api/chat"
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_metrics = enable_metrics

        # Metrics tracking
        self.success_count = 0
        self.failure_count = 0

        logger.info(
            f"OllamaProvider initialized: model={model}, "
            f"endpoint={endpoint}, max_retries={max_retries}"
        )

    def get_metrics(self) -> dict:
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
        messages: List[ChatMessage],
        response_format: Literal["json", "text"] = "text",
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a chat response using Ollama.

        Args:
            messages: Conversation messages (ChatMessage format)
            response_format: "json" for structured output, "text" for plain text
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            The LLM's response as a string

        Raises:
            httpx.HTTPError: If all retry attempts fail
        """
        # Convert ChatMessages to Ollama format (similar to OpenAI)
        chat_messages = [msg.model_dump(exclude_none=True) for msg in messages]

        # Build request payload for /api/chat endpoint
        payload = {
            "model": self.model,
            "messages": chat_messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        if response_format == "json":
            payload["format"] = "json"

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        # Execute with retry logic
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.endpoint, json=payload)
                    response.raise_for_status()

                    result = response.json()["message"]["content"].strip()

                    # Success - update metrics
                    if self.enable_metrics:
                        self.success_count += 1

                    if attempt > 0:
                        logger.info(f"Request succeeded on attempt {attempt + 1}")

                    return result

            except (httpx.ConnectError, httpx.TimeoutException) as e:
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

            except httpx.HTTPStatusError as e:
                # Non-retriable HTTP errors
                last_exception = e
                logger.error(f"HTTP error {e.response.status_code}: {e}")
                break  # Don't retry

        # All attempts failed
        if self.enable_metrics:
            self.failure_count += 1

        raise last_exception or httpx.HTTPError("Unknown error")

    async def chat_json(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        Generate and parse JSON response.

        Convenience method that calls chat() with response_format="json"
        and automatically parses the result.

        Args:
            messages: Conversation messages (ChatMessage format)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            Parsed JSON object

        Raises:
            httpx.HTTPError: If request fails
            json.JSONDecodeError: If response is not valid JSON
        """
        response = await self.chat(messages, response_format="json", max_tokens=max_tokens)
        return extract_json_from_markdown(response)
