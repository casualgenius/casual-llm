"""
OpenAI message converters.

Converts casual-llm ChatMessage format to OpenAI API format and vice versa.
"""

import logging
from typing import TYPE_CHECKING, Any

from casual_llm.messages import (
    ChatMessage,
    AssistantToolCall,
    AssistantToolCallFunction,
    TextContent,
    ImageContent,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageToolCall

logger = logging.getLogger(__name__)


def _convert_image_to_openai(image: ImageContent) -> dict[str, Any]:
    """
    Convert ImageContent to OpenAI image_url format.

    OpenAI expects images in the format:
    {"type": "image_url", "image_url": {"url": "..."}}

    For base64 images, the URL should be a data URI:
    data:image/jpeg;base64,...
    """
    if isinstance(image.source, str):
        # URL source - use directly
        image_url = image.source
    else:
        # Base64 dict source - construct data URI
        base64_data = image.source.get("data", "")
        image_url = f"data:{image.media_type};base64,{base64_data}"

    return {
        "type": "image_url",
        "image_url": {"url": image_url},
    }


def _convert_user_content_to_openai(
    content: str | list[TextContent | ImageContent] | None,
) -> str | list[dict[str, Any]] | None:
    """
    Convert UserMessage content to OpenAI format.

    Handles both simple string content (backward compatible) and
    multimodal content arrays (text + images).
    """
    if content is None or isinstance(content, str):
        # Simple string content or None - pass through
        return content

    # Multimodal content array
    openai_content: list[dict[str, Any]] = []

    for item in content:
        if isinstance(item, TextContent):
            openai_content.append({"type": "text", "text": item.text})
        elif isinstance(item, ImageContent):
            openai_content.append(_convert_image_to_openai(item))

    return openai_content


def convert_messages_to_openai(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """
    Convert casual-llm ChatMessage list to OpenAI format.

    Handles all message types including tool calls and tool results.

    Args:
        messages: List of ChatMessage objects

    Returns:
        List of dictionaries in OpenAI ChatCompletionMessageParam format

    Examples:
        >>> from casual_llm import UserMessage, AssistantMessage
        >>> messages = [UserMessage(content="Hello")]
        >>> openai_msgs = convert_messages_to_openai(messages)
        >>> openai_msgs[0]["role"]
        'user'
    """
    if not messages:
        return []

    logger.debug(f"Converting {len(messages)} messages to OpenAI format")

    openai_messages: list[dict[str, Any]] = []

    for msg in messages:
        match msg.role:
            case "assistant":
                # Handle assistant messages with optional tool calls
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content,
                }

                # Add tool calls if present
                if msg.tool_calls:
                    tool_calls = []
                    for tool_call in msg.tool_calls:
                        tool_calls.append(
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        )
                    message["tool_calls"] = tool_calls

                openai_messages.append(message)

            case "system":
                openai_messages.append({"role": "system", "content": msg.content})

            case "tool":
                openai_messages.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id,
                        "name": msg.name,
                    }
                )

            case "user":
                openai_messages.append({
                    "role": "user",
                    "content": _convert_user_content_to_openai(msg.content),
                })

            case _:
                logger.warning(f"Unknown message role: {msg.role}")

    return openai_messages


def convert_tool_calls_from_openai(
    response_tool_calls: list["ChatCompletionMessageToolCall"],
) -> list[AssistantToolCall]:
    """
    Convert OpenAI ChatCompletionMessageToolCall to casual-llm format.

    Args:
        response_tool_calls: List of ChatCompletionMessageToolCall from OpenAI response

    Returns:
        List of AssistantToolCall objects

    Examples:
        >>> # Assuming response has tool_calls
        >>> # tool_calls = convert_tool_calls_from_openai(response.choices[0].message.tool_calls)
        >>> # assert len(tool_calls) > 0
        pass
    """
    tool_calls = []

    for tool in response_tool_calls:
        logger.debug(f"Converting tool call: {tool.function.name}")

        tool_call = AssistantToolCall(
            id=tool.id,
            type="function",
            function=AssistantToolCallFunction(
                name=tool.function.name, arguments=tool.function.arguments
            ),
        )
        tool_calls.append(tool_call)

    logger.debug(f"Converted {len(tool_calls)} tool calls")
    return tool_calls


__all__ = [
    "convert_messages_to_openai",
    "convert_tool_calls_from_openai",
]
