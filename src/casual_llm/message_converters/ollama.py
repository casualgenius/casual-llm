"""
Ollama message converters.

Converts casual-llm ChatMessage format to Ollama API format and vice versa.
"""

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from casual_llm.messages import (
    ChatMessage,
    AssistantToolCall,
    AssistantToolCallFunction,
    TextContent,
    ImageContent,
)
from casual_llm.utils.image import (
    strip_base64_prefix,
    fetch_image_as_base64,
)

if TYPE_CHECKING:
    from ollama._types import Message

logger = logging.getLogger(__name__)


async def _convert_image_to_ollama(image: ImageContent) -> str:
    """
    Convert ImageContent to Ollama base64 format.

    Ollama expects images as raw base64 strings (no data URI prefix).

    For URL sources, this function fetches the image and converts to base64.

    Raises:
        ImageFetchError: If a URL image cannot be fetched.
    """
    if isinstance(image.source, str):
        # Check if it's a data URI or a URL
        if image.source.startswith("data:"):
            # Data URI - extract base64 data
            return strip_base64_prefix(image.source)
        else:
            # Regular URL - fetch and convert to base64
            logger.debug(f"Fetching image from URL for Ollama: {image.source}")
            base64_data, _ = await fetch_image_as_base64(image.source)
            return base64_data
    else:
        # Base64 dict source - use directly
        base64_data = image.source.get("data", "")
        # Strip any data URI prefix that might be present
        return strip_base64_prefix(base64_data)


async def _convert_user_content_to_ollama(
    content: str | list[TextContent | ImageContent] | None,
) -> tuple[str, list[str]]:
    """
    Convert UserMessage content to Ollama format.

    Handles both simple string content (backward compatible) and
    multimodal content arrays (text + images).

    Ollama uses a format where text goes in "content" and images
    go in a separate "images" array as raw base64 strings.

    Returns:
        A tuple of (text_content, images_list) where:
            - text_content: Combined text from all TextContent items
            - images_list: List of base64-encoded image strings

    Raises:
        ImageFetchError: If a URL image cannot be fetched.
    """
    if content is None:
        return "", []

    if isinstance(content, str):
        # Simple string content
        return content, []

    # Multimodal content array
    text_parts: list[str] = []
    images: list[str] = []

    for item in content:
        if isinstance(item, TextContent):
            text_parts.append(item.text)
        elif isinstance(item, ImageContent):
            images.append(await _convert_image_to_ollama(item))

    # Join text parts with newlines
    text_content = "\n".join(text_parts) if text_parts else ""

    return text_content, images


async def convert_messages_to_ollama(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """
    Convert casual-llm ChatMessage list to Ollama format.

    Unlike OpenAI which expects tool call arguments as JSON strings,
    Ollama expects them as dictionaries. This function handles that conversion.

    Supports multimodal messages with images. Ollama expects images as raw
    base64 strings in a separate "images" array.

    Args:
        messages: List of ChatMessage objects

    Returns:
        List of dictionaries in Ollama message format

    Raises:
        ImageFetchError: If a URL image cannot be fetched.

    Examples:
        >>> import asyncio
        >>> from casual_llm import UserMessage
        >>> messages = [UserMessage(content="Hello")]
        >>> ollama_msgs = asyncio.run(convert_messages_to_ollama(messages))
        >>> ollama_msgs[0]["role"]
        'user'
    """
    if not messages:
        return []

    logger.debug(f"Converting {len(messages)} messages to Ollama format")

    ollama_messages: list[dict[str, Any]] = []

    for msg in messages:
        match msg.role:
            case "assistant":
                # Handle assistant messages with optional tool calls
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content,
                }

                # Add tool calls if present
                # Ollama expects arguments as dict, not JSON string
                if msg.tool_calls:
                    tool_calls = []
                    for tool_call in msg.tool_calls:
                        # Parse arguments from JSON string to dict for Ollama
                        arguments_dict = (
                            json.loads(tool_call.function.arguments)
                            if tool_call.function.arguments
                            else {}
                        )

                        tool_calls.append(
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": arguments_dict,  # dict for Ollama
                                },
                            }
                        )
                    message["tool_calls"] = tool_calls

                ollama_messages.append(message)

            case "system":
                ollama_messages.append({"role": "system", "content": msg.content})

            case "tool":
                ollama_messages.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id,
                        "name": msg.name,
                    }
                )

            case "user":
                text_content, images = await _convert_user_content_to_ollama(msg.content)
                user_message: dict[str, Any] = {"role": "user", "content": text_content}
                if images:
                    user_message["images"] = images
                ollama_messages.append(user_message)

            case _:
                logger.warning(f"Unknown message role: {msg.role}")

    return ollama_messages


def convert_tool_calls_from_ollama(
    response_tool_calls: list["Message.ToolCall"],
) -> list[AssistantToolCall]:
    """
    Convert Ollama tool calls to casual-llm format.

    Handles Ollama's ToolCall objects which have function arguments as a Mapping
    instead of a JSON string. Also generates unique IDs if not provided.

    Args:
        response_tool_calls: List of ollama._types.Message.ToolCall objects

    Returns:
        List of AssistantToolCall objects

    Examples:
        >>> # from ollama response.message.tool_calls
        >>> # tool_calls = convert_tool_calls_from_ollama(response.message.tool_calls)
        >>> # assert len(tool_calls) > 0
        pass
    """
    tool_calls = []

    for tool in response_tool_calls:
        # Get tool call ID, generate one if missing
        tool_call_id = getattr(tool, "id", None)
        if not tool_call_id:
            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
            logger.debug(f"Generated tool call ID: {tool_call_id}")

        logger.debug(f"Converting tool call: {tool.function.name}")

        # Convert arguments from Mapping[str, Any] to JSON string
        # Ollama returns arguments as a dict, but we need a JSON string
        arguments_dict = tool.function.arguments
        arguments_json = json.dumps(arguments_dict) if arguments_dict else "{}"

        tool_call = AssistantToolCall(
            id=tool_call_id,
            type=getattr(tool, "type", "function"),
            function=AssistantToolCallFunction(name=tool.function.name, arguments=arguments_json),
        )
        tool_calls.append(tool_call)

    logger.debug(f"Converted {len(tool_calls)} tool calls")
    return tool_calls


__all__ = [
    "convert_messages_to_ollama",
    "convert_tool_calls_from_ollama",
]
