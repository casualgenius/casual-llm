"""
OpenAI LLM provider (compatible with OpenAI API and compatible services).
"""

import logging
from typing import List, Literal, Optional, Dict, Any
from openai import AsyncOpenAI

from casual_llm.messages import ChatMessage
from casual_llm.utils import extract_json_from_markdown

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """
    OpenAI-compatible LLM provider.

    Works with OpenAI API and compatible services (OpenRouter, etc.).
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        temperature: float = 0.2,
        timeout: float = 60.0,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: Model name (e.g., "gpt-4o-mini")
            api_key: API key (optional, can use OPENAI_API_KEY env var)
            base_url: Base URL for API (e.g., "https://openrouter.ai/api/v1")
            organization: OpenAI organization ID (optional)
            temperature: Temperature for generation (0.0-1.0)
            timeout: HTTP request timeout in seconds
            extra_kwargs: Additional kwargs to pass to client.chat.completions.create()
        """
        client_kwargs: Dict[str, Any] = {"timeout": timeout}

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

        logger.info(
            f"OpenAIProvider initialized: model={model}, " f"base_url={base_url or 'default'}"
        )

    async def chat(
        self,
        messages: List[ChatMessage],
        response_format: Literal["json", "text"] = "text",
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a chat response using OpenAI API.

        Args:
            messages: Conversation messages (ChatMessage format)
            response_format: "json" for structured output, "text" for plain text
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            The LLM's response as a string

        Raises:
            openai.OpenAIError: If request fails
        """
        # ChatMessages already match OpenAI format - just convert to dict
        chat_messages = [msg.model_dump(exclude_none=True) for msg in messages]

        # Build request kwargs
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": self.temperature,
        }

        if response_format == "json":
            request_kwargs["response_format"] = {"type": "json_object"}

        if max_tokens:
            request_kwargs["max_tokens"] = max_tokens

        # Merge extra kwargs
        request_kwargs.update(self.extra_kwargs)

        logger.debug(f"Generating with model {self.model}")
        response = await self.client.chat.completions.create(**request_kwargs)

        content = response.choices[0].message.content or ""
        logger.debug(f"Generated {len(content)} characters")

        return content

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
            openai.OpenAIError: If request fails
            json.JSONDecodeError: If response is not valid JSON
        """
        response = await self.chat(messages, response_format="json", max_tokens=max_tokens)
        return extract_json_from_markdown(response)
