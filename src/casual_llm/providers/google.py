"""
Google LLM provider (Gemini models with vision support).
"""

from __future__ import annotations

import json
import logging
from typing import Literal, Any, AsyncIterator, TYPE_CHECKING
from pydantic import BaseModel

from casual_llm.messages import (
    ChatMessage,
    AssistantMessage,
    StreamChunk,
    AssistantToolCall,
    AssistantToolCallFunction,
)
from casual_llm.tools import Tool
from casual_llm.usage import Usage
from casual_llm.message_converters import convert_messages_to_google

if TYPE_CHECKING:
    from google.generativeai import GenerativeModel

logger = logging.getLogger(__name__)


def _tool_to_google(tool: Tool) -> dict[str, Any]:
    """
    Convert a casual-llm Tool to Google function declaration format.

    Args:
        tool: Tool to convert

    Returns:
        Dictionary in Google's function declaration format
    """
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": {
                name: param.model_dump(exclude_none=True) for name, param in tool.parameters.items()
            },
            "required": tool.required,
        },
    }


def _tools_to_google(tools: list[Tool]) -> list[dict[str, Any]]:
    """
    Convert multiple casual-llm Tools to Google function declarations.

    Args:
        tools: List of tools to convert

    Returns:
        List of function declarations in Google format
    """
    logger.debug(f"Converting {len(tools)} tools to Google format")
    return [_tool_to_google(tool) for tool in tools]


class GoogleProvider:
    """
    Google LLM provider for Gemini models.

    Supports Gemini 1.5 and 2.0 family models including vision capabilities.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float | None = None,
        timeout: float = 60.0,
        max_tokens: int = 4096,
        extra_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize Google provider.

        Args:
            model: Model name (e.g., "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash")
            api_key: API key (optional, can use GOOGLE_API_KEY env var)
            temperature: Temperature for generation (0.0-1.0, optional - uses Google
                default if not set)
            timeout: HTTP request timeout in seconds
            max_tokens: Default max tokens for responses
            extra_kwargs: Additional kwargs to pass to model.generate_content()
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Google provider requires the 'google-generativeai' package. "
                "Install it with: pip install casual-llm[google]"
            )

        # Configure the SDK with API key if provided
        if api_key:
            genai.configure(api_key=api_key)

        self._genai = genai
        self.model_name = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.extra_kwargs = extra_kwargs or {}

        # Usage tracking
        self._last_usage: Usage | None = None

        logger.info(f"GoogleProvider initialized: model={model}")

    def _create_model(
        self,
        system_instruction: str | None = None,
        tools: list[Tool] | None = None,
    ) -> "GenerativeModel":
        """
        Create a GenerativeModel instance with the given configuration.

        Args:
            system_instruction: Optional system instruction for the model
            tools: Optional list of tools for function calling

        Returns:
            Configured GenerativeModel instance
        """
        model_kwargs: dict[str, Any] = {}

        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        if tools:
            google_tools = _tools_to_google(tools)
            model_kwargs["tools"] = [{"function_declarations": google_tools}]
            logger.debug(f"Added {len(google_tools)} tools to model")

        return self._genai.GenerativeModel(self.model_name, **model_kwargs)

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
        Generate a chat response using Google Gemini API.

        Args:
            messages: Conversation messages (ChatMessage format)
            response_format: "json" for JSON output, "text" for plain text, or a Pydantic
                BaseModel class for JSON Schema-based structured output. Note: Google
                handles JSON through system instruction augmentation.
            max_tokens: Maximum tokens to generate (optional, uses instance default)
            tools: List of tools available for the LLM to call (optional)
            temperature: Temperature for this request (optional, overrides instance temperature)

        Returns:
            AssistantMessage with content and optional tool_calls

        Raises:
            google.generativeai.types.GoogleAPIError: If request fails

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
        # Convert messages to Google format (returns messages and system instruction separately)
        # This is async because URL images need to be fetched and converted to base64
        google_messages, system_instruction = await convert_messages_to_google(messages)
        logger.debug(f"Converted {len(messages)} messages to Google format")

        # Handle JSON response format by augmenting system instruction
        if response_format == "json":
            json_instruction = (
                "IMPORTANT: You must respond with valid JSON only. "
                "Do not include any text before or after the JSON."
            )
            if system_instruction:
                system_instruction = f"{system_instruction}\n\n{json_instruction}"
            else:
                system_instruction = json_instruction
        elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
            schema = response_format.model_json_schema()
            json_instruction = (
                f"IMPORTANT: You must respond with valid JSON matching this schema:\n"
                f"{json.dumps(schema)}\n"
                "Do not include any text before or after the JSON."
            )
            if system_instruction:
                system_instruction = f"{system_instruction}\n\n{json_instruction}"
            else:
                system_instruction = json_instruction
            logger.debug(f"Using JSON Schema from Pydantic model: {response_format.__name__}")

        # Create model with system instruction and tools
        model = self._create_model(
            system_instruction=system_instruction,
            tools=tools,
        )

        # Use provided temperature or fall back to instance temperature
        temp = temperature if temperature is not None else self.temperature

        # Build generation config
        generation_config: dict[str, Any] = {
            "max_output_tokens": max_tokens or self.max_tokens,
        }

        if temp is not None:
            generation_config["temperature"] = temp

        # Merge extra kwargs
        generation_config.update(self.extra_kwargs)

        logger.debug(f"Generating with model {self.model_name}")
        response = await model.generate_content_async(
            google_messages,
            generation_config=generation_config,
            request_options={"timeout": self.timeout},
        )

        # Extract usage statistics if available
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self._last_usage = Usage(
                prompt_tokens=response.usage_metadata.prompt_token_count,
                completion_tokens=response.usage_metadata.candidates_token_count,
            )
            logger.debug(
                f"Usage: {response.usage_metadata.prompt_token_count} prompt tokens, "
                f"{response.usage_metadata.candidates_token_count} completion tokens"
            )

        # Process response parts
        content_parts: list[str] = []
        tool_calls: list[AssistantToolCall] = []

        # Get the first candidate's content
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    content_parts.append(part.text)
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    # Generate a unique ID for the tool call
                    tool_call_id = f"call_{hash(fc.name) % 100000:05d}"
                    tool_calls.append(
                        AssistantToolCall(
                            id=tool_call_id,
                            function=AssistantToolCallFunction(
                                name=fc.name,
                                arguments=json.dumps(dict(fc.args)),
                            ),
                        )
                    )

        if tool_calls:
            logger.debug(f"Assistant requested {len(tool_calls)} tool calls")

        # Combine text content
        content = "".join(content_parts)
        logger.debug(f"Generated {len(content)} characters")

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a chat response from Google Gemini API.

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
            google.generativeai.types.GoogleAPIError: If request fails

        Examples:
            >>> async for chunk in provider.stream([UserMessage(content="Hello")]):
            ...     print(chunk.content, end="", flush=True)
        """
        # Convert messages to Google format
        # This is async because URL images need to be fetched and converted to base64
        google_messages, system_instruction = await convert_messages_to_google(messages)
        logger.debug(f"Converted {len(messages)} messages to Google format for streaming")

        # Handle JSON response format by augmenting system instruction
        if response_format == "json":
            json_instruction = (
                "IMPORTANT: You must respond with valid JSON only. "
                "Do not include any text before or after the JSON."
            )
            if system_instruction:
                system_instruction = f"{system_instruction}\n\n{json_instruction}"
            else:
                system_instruction = json_instruction
        elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
            schema = response_format.model_json_schema()
            json_instruction = (
                f"IMPORTANT: You must respond with valid JSON matching this schema:\n"
                f"{json.dumps(schema)}\n"
                "Do not include any text before or after the JSON."
            )
            if system_instruction:
                system_instruction = f"{system_instruction}\n\n{json_instruction}"
            else:
                system_instruction = json_instruction
            logger.debug(f"Using JSON Schema from Pydantic model: {response_format.__name__}")

        # Create model with system instruction and tools
        model = self._create_model(
            system_instruction=system_instruction,
            tools=tools,
        )

        # Use provided temperature or fall back to instance temperature
        temp = temperature if temperature is not None else self.temperature

        # Build generation config
        generation_config: dict[str, Any] = {
            "max_output_tokens": max_tokens or self.max_tokens,
        }

        if temp is not None:
            generation_config["temperature"] = temp

        # Merge extra kwargs
        generation_config.update(self.extra_kwargs)

        logger.debug(f"Starting stream with model {self.model_name}")
        response = await model.generate_content_async(
            google_messages,
            generation_config=generation_config,
            request_options={"timeout": self.timeout},
            stream=True,
        )

        async for chunk in response:
            # Extract text from chunk parts
            if chunk.candidates and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        yield StreamChunk(content=part.text, finish_reason=None)

        logger.debug("Stream completed")


__all__ = ["GoogleProvider"]
