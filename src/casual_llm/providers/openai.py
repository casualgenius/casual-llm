"""
OpenAI LLM provider (compatible with OpenAI API and compatible services).
"""

import logging
from typing import List, Literal, Optional, Dict, Any
from openai import AsyncOpenAI

from casual_llm.messages import ChatMessage, AssistantMessage
from casual_llm.tools import Tool
from casual_llm.tool_converters import tools_to_openai
from casual_llm.message_converters import (
    convert_messages_to_openai,
    convert_tool_calls_from_openai,
)
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
        tools: Optional[List[Tool]] = None,
    ) -> str | AssistantMessage:
        """
        Generate a chat response using OpenAI API.

        Args:
            messages: Conversation messages (ChatMessage format)
            response_format: "json" for structured output, "text" for plain text
            max_tokens: Maximum tokens to generate (optional)
            tools: List of tools available for the LLM to call (optional)

        Returns:
            If tools are provided: AssistantMessage (may contain tool_calls)
            If no tools: String response content

        Raises:
            openai.OpenAIError: If request fails
        """
        # Convert messages to OpenAI format using converter
        chat_messages = convert_messages_to_openai(messages)
        logger.debug(f"Converted {len(messages)} messages to OpenAI format")

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

        # If tools were provided, return AssistantMessage with potential tool calls
        if tools:
            tool_calls = None
            if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                logger.debug(f"Assistant requested {len(response_message.tool_calls)} tool calls")
                tool_calls = convert_tool_calls_from_openai(response_message.tool_calls)

            return AssistantMessage(
                content=response_message.content,
                tool_calls=tool_calls
            )

        # No tools - return simple string response
        content = response_message.content or ""
        logger.debug(f"Generated {len(content)} characters")
        return content

    async def chat_json(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
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
        # Response should be a string when no tools are provided
        if isinstance(response, str):
            result: dict[str, Any] = extract_json_from_markdown(response)
            return result
        raise ValueError("chat_json cannot be used with tools")
