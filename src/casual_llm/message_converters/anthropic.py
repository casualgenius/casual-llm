"""
Anthropic message converters.

Converts casual-llm ChatMessage format to Anthropic API format and vice versa.
"""

import json
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
    from anthropic.types import ToolUseBlock

logger = logging.getLogger(__name__)


def _convert_image_to_anthropic(image: ImageContent) -> dict[str, Any]:
    """
    Convert ImageContent to Anthropic image block format.

    Anthropic supports both URL and base64 images directly.

    Args:
        image: ImageContent with either URL or base64 source

    Returns:
        Dictionary in Anthropic image block format

    Examples:
        >>> from casual_llm import ImageContent
        >>> img = ImageContent(source="https://example.com/image.jpg")
        >>> block = _convert_image_to_anthropic(img)
        >>> block["type"]
        'image'
    """
    if isinstance(image.source, str):
        # URL image - Anthropic supports URLs directly
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": image.source,
            },
        }
    else:
        # Base64 image
        base64_data = image.source.get("data", "")
        media_type = image.media_type or "image/jpeg"

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_data,
            },
        }


def _convert_user_content_to_anthropic(
    content: str | list[TextContent | ImageContent] | None,
) -> list[dict[str, Any]]:
    """
    Convert UserMessage content to Anthropic format.

    Handles both simple string content (backward compatible) and
    multimodal content arrays (text + images).

    Anthropic always uses content blocks (array format), even for simple text.

    Args:
        content: Message content (string, multimodal array, or None)

    Returns:
        List of content blocks in Anthropic format

    Examples:
        >>> content_blocks = _convert_user_content_to_anthropic("Hello")
        >>> content_blocks[0]["type"]
        'text'
    """
    if content is None:
        return [{"type": "text", "text": ""}]

    if isinstance(content, str):
        # Simple string content
        return [{"type": "text", "text": content}]

    # Multimodal content array
    content_blocks: list[dict[str, Any]] = []

    for item in content:
        if isinstance(item, TextContent):
            content_blocks.append({"type": "text", "text": item.text})
        elif isinstance(item, ImageContent):
            content_blocks.append(_convert_image_to_anthropic(item))

    return content_blocks


def extract_system_message(messages: list[ChatMessage]) -> str | None:
    """
    Extract system message content from the messages list.

    Anthropic requires system messages to be passed as a separate parameter,
    not as part of the messages array. This function extracts the first
    system message content for use with the Anthropic API.

    Args:
        messages: List of ChatMessage objects

    Returns:
        System message content string, or None if no system message present

    Examples:
        >>> from casual_llm import SystemMessage, UserMessage
        >>> messages = [SystemMessage(content="You are helpful"), UserMessage(content="Hello")]
        >>> extract_system_message(messages)
        'You are helpful'
    """
    for msg in messages:
        if msg.role == "system":
            logger.debug("Extracted system message")
            return msg.content
    return None


def convert_messages_to_anthropic(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """
    Convert casual-llm ChatMessage list to Anthropic format.

    Handles all message types including tool calls and tool results.
    Note: System messages are excluded - use extract_system_message() to get
    the system message content for the separate `system` parameter.

    Anthropic format differences:
    - System messages are NOT included (passed separately)
    - Tool results go in user messages with "tool_result" content type
    - Content is always an array of content blocks

    Args:
        messages: List of ChatMessage objects

    Returns:
        List of dictionaries in Anthropic MessageParam format

    Examples:
        >>> from casual_llm import UserMessage, AssistantMessage
        >>> messages = [UserMessage(content="Hello")]
        >>> anthropic_msgs = convert_messages_to_anthropic(messages)
        >>> anthropic_msgs[0]["role"]
        'user'
    """
    if not messages:
        return []

    logger.debug("Converting %d messages to Anthropic format", len(messages))

    anthropic_messages: list[dict[str, Any]] = []

    for msg in messages:
        match msg.role:
            case "assistant":
                # Handle assistant messages with optional tool calls
                content_blocks: list[dict[str, Any]] = []

                # Add text content if present
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})

                # Add tool use blocks if present
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        # Parse arguments JSON string back to dict for Anthropic
                        try:
                            input_data = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            input_data = {}
                            logger.warning(
                                "Failed to parse tool call arguments: %s",
                                tool_call.function.arguments,
                            )

                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "input": input_data,
                            }
                        )

                anthropic_messages.append(
                    {
                        "role": "assistant",
                        "content": content_blocks,
                    }
                )

            case "system":
                # System messages are excluded - they are passed separately
                # via the `system` parameter in the API call
                logger.debug("Skipping system message (handled separately)")
                continue

            case "tool":
                # Tool results go in user messages with tool_result content type
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )

            case "user":
                # User messages with text and/or image content
                content_blocks = _convert_user_content_to_anthropic(msg.content)
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": content_blocks,
                    }
                )

            case _:
                logger.warning("Unknown message role: %s", msg.role)

    return anthropic_messages


def convert_tool_calls_from_anthropic(
    response_tool_calls: list["ToolUseBlock"],
) -> list[AssistantToolCall]:
    """
    Convert Anthropic ToolUseBlock to casual-llm format.

    Anthropic returns tool call arguments as a dict in the `input` field,
    which must be serialized to JSON string for AssistantToolCallFunction.

    Args:
        response_tool_calls: List of ToolUseBlock from Anthropic response

    Returns:
        List of AssistantToolCall objects

    Examples:
        >>> # Assuming response has tool_use blocks
        >>> # tool_calls = convert_tool_calls_from_anthropic(tool_use_blocks)
        >>> # assert len(tool_calls) > 0
        pass
    """
    tool_calls = []

    for tool in response_tool_calls:
        logger.debug("Converting tool call: %s", tool.name)

        # Serialize input dict to JSON string for casual-llm format
        arguments = json.dumps(tool.input) if tool.input else "{}"

        tool_call = AssistantToolCall(
            id=tool.id,
            type="function",
            function=AssistantToolCallFunction(name=tool.name, arguments=arguments),
        )
        tool_calls.append(tool_call)

    logger.debug("Converted %d tool calls", len(tool_calls))
    return tool_calls


__all__ = [
    "convert_messages_to_anthropic",
    "extract_system_message",
    "convert_tool_calls_from_anthropic",
    "_convert_image_to_anthropic",
    "_convert_user_content_to_anthropic",
]
