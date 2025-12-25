"""
Anthropic message converters.

Converts casual-llm ChatMessage format to Anthropic API format and vice versa.
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
from casual_llm.utils.image import (
    strip_base64_prefix,
    fetch_image_as_base64,
)

if TYPE_CHECKING:
    from anthropic.types import ToolUseBlock

logger = logging.getLogger(__name__)


async def _convert_image_to_anthropic(image: ImageContent) -> dict[str, Any]:
    """
    Convert ImageContent to Anthropic image format.

    Anthropic expects images in the format:
    {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}

    For URL sources, this function fetches the image and converts to base64,
    since Anthropic requires base64-encoded data rather than URL references.

    Note: Unlike OpenAI, Anthropic does NOT use data URI format. The base64 data
    must be raw (no 'data:image/...;base64,' prefix).

    Raises:
        ImageFetchError: If a URL image cannot be fetched.
    """
    if isinstance(image.source, str):
        # Check if it's a data URI or a URL
        if image.source.startswith("data:"):
            # Data URI - extract base64 data
            base64_data = strip_base64_prefix(image.source)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.media_type,
                    "data": base64_data,
                },
            }
        else:
            # Regular URL - fetch and convert to base64
            # Anthropic requires base64 data, not URL references
            logger.debug(f"Fetching image from URL for Anthropic: {image.source}")
            base64_data, fetched_media_type = await fetch_image_as_base64(image.source)
            # Use fetched media type if not explicitly specified
            media_type = (
                image.media_type
                if image.media_type != "image/jpeg"  # not the default
                else fetched_media_type
            )
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
    else:
        # Base64 dict source - use directly (already in correct format)
        base64_data = image.source.get("data", "")
        # Strip any data URI prefix that might be present
        base64_data = strip_base64_prefix(base64_data)
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.media_type,
                "data": base64_data,
            },
        }


async def _convert_user_content_to_anthropic(
    content: str | list[TextContent | ImageContent] | None,
) -> str | list[dict[str, Any]]:
    """
    Convert UserMessage content to Anthropic format.

    Handles both simple string content (backward compatible) and
    multimodal content arrays (text + images).

    For images with URL sources, this will fetch the image and convert
    it to base64 format since Anthropic requires base64-encoded images.

    Raises:
        ImageFetchError: If a URL image cannot be fetched.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        # Simple string content - pass through
        return content

    # Multimodal content array
    anthropic_content: list[dict[str, Any]] = []

    for item in content:
        if isinstance(item, TextContent):
            anthropic_content.append({"type": "text", "text": item.text})
        elif isinstance(item, ImageContent):
            anthropic_content.append(await _convert_image_to_anthropic(item))

    return anthropic_content


async def convert_messages_to_anthropic(
    messages: list[ChatMessage],
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Convert casual-llm ChatMessage list to Anthropic format.

    Handles all message types including tool calls and tool results.
    System messages are extracted and returned separately since Anthropic
    takes system prompts as a separate parameter.

    For images with URL sources, this function will fetch the images and
    convert them to base64 format since Anthropic requires base64-encoded images.

    Args:
        messages: List of ChatMessage objects

    Returns:
        A tuple of (messages, system_prompt) where:
            - messages: List of dictionaries in Anthropic message format
            - system_prompt: The system message content (if any), or None

    Raises:
        ImageFetchError: If a URL image cannot be fetched.

    Examples:
        >>> import asyncio
        >>> from casual_llm import UserMessage, SystemMessage
        >>> messages = [SystemMessage(content="Be helpful"), UserMessage(content="Hello")]
        >>> anthropic_msgs, system = asyncio.run(convert_messages_to_anthropic(messages))
        >>> anthropic_msgs[0]["role"]
        'user'
        >>> system
        'Be helpful'
    """
    if not messages:
        return [], None

    logger.debug(f"Converting {len(messages)} messages to Anthropic format")

    anthropic_messages: list[dict[str, Any]] = []
    system_prompt: str | None = None

    for msg in messages:
        match msg.role:
            case "assistant":
                # Handle assistant messages with optional tool calls
                content: list[dict[str, Any]] = []

                # Add text content if present
                if msg.content:
                    content.append({"type": "text", "text": msg.content})

                # Add tool_use blocks if present
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        # Parse arguments from JSON string
                        import json

                        try:
                            input_data = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            input_data = {}

                        content.append(
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
                        "content": content if content else msg.content,
                    }
                )

            case "system":
                # Anthropic takes system prompt as separate parameter
                # Use the last system message if multiple are present
                system_prompt = msg.content

            case "tool":
                # Anthropic uses tool_result content blocks within user messages
                # Tool results should be grouped with their corresponding user turn
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
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": await _convert_user_content_to_anthropic(msg.content),
                    }
                )

            case _:
                logger.warning(f"Unknown message role: {msg.role}")

    return anthropic_messages, system_prompt


def convert_tool_calls_from_anthropic(
    content_blocks: list["ToolUseBlock"],
) -> list[AssistantToolCall]:
    """
    Convert Anthropic ToolUseBlock to casual-llm format.

    Args:
        content_blocks: List of ToolUseBlock from Anthropic response content

    Returns:
        List of AssistantToolCall objects

    Examples:
        >>> # Assuming response has tool_use blocks
        >>> # tool_calls = convert_tool_calls_from_anthropic(tool_use_blocks)
        >>> # assert len(tool_calls) > 0
        pass
    """
    import json

    tool_calls = []

    for block in content_blocks:
        logger.debug(f"Converting tool call: {block.name}")

        # Anthropic provides input as dict, we need to serialize to JSON string
        arguments = json.dumps(block.input) if block.input else "{}"

        tool_call = AssistantToolCall(
            id=block.id,
            type="function",
            function=AssistantToolCallFunction(
                name=block.name,
                arguments=arguments,
            ),
        )
        tool_calls.append(tool_call)

    logger.debug(f"Converted {len(tool_calls)} tool calls")
    return tool_calls


__all__ = [
    "convert_messages_to_anthropic",
    "convert_tool_calls_from_anthropic",
]
