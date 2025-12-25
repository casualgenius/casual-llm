"""
Google message converters.

Converts casual-llm ChatMessage format to Google Generative AI (Gemini) API format.
"""

import logging
from typing import Any

from casual_llm.messages import (
    ChatMessage,
    TextContent,
    ImageContent,
)
from casual_llm.utils.image import (
    strip_base64_prefix,
    fetch_image_as_base64,
)

logger = logging.getLogger(__name__)


async def _convert_image_to_google(image: ImageContent) -> dict[str, Any]:
    """
    Convert ImageContent to Google inline_data format.

    Google expects images in the parts format:
    {"inline_data": {"mime_type": "image/jpeg", "data": "..."}}

    For URL sources, this function fetches the image and converts to base64,
    since Google's inline_data format requires base64-encoded data.

    Note: Google uses raw base64 data (no 'data:image/...;base64,' prefix).

    Raises:
        ImageFetchError: If a URL image cannot be fetched.
    """
    if isinstance(image.source, str):
        # Check if it's a data URI or a URL
        if image.source.startswith("data:"):
            # Data URI - extract base64 data
            base64_data = strip_base64_prefix(image.source)
            return {
                "inline_data": {
                    "mime_type": image.media_type,
                    "data": base64_data,
                },
            }
        else:
            # Regular URL - fetch and convert to base64
            logger.debug(f"Fetching image from URL for Google: {image.source}")
            base64_data, fetched_media_type = await fetch_image_as_base64(image.source)
            # Use fetched media type if not explicitly specified
            media_type = (
                image.media_type
                if image.media_type != "image/jpeg"  # not the default
                else fetched_media_type
            )
            return {
                "inline_data": {
                    "mime_type": media_type,
                    "data": base64_data,
                },
            }
    else:
        # Base64 dict source - use directly
        base64_data = image.source.get("data", "")
        # Strip any data URI prefix that might be present
        base64_data = strip_base64_prefix(base64_data)
        return {
            "inline_data": {
                "mime_type": image.media_type,
                "data": base64_data,
            },
        }


async def _convert_user_content_to_google_parts(
    content: str | list[TextContent | ImageContent] | None,
) -> list[dict[str, Any]]:
    """
    Convert UserMessage content to Google parts format.

    Handles both simple string content (backward compatible) and
    multimodal content arrays (text + images).

    Google uses a parts-based format where each part is either:
    - {"text": "..."} for text content
    - {"inline_data": {"mime_type": "...", "data": "..."}} for images

    Raises:
        ImageFetchError: If a URL image cannot be fetched.
    """
    if content is None:
        return []

    if isinstance(content, str):
        # Simple string content - wrap in parts format
        return [{"text": content}]

    # Multimodal content array
    google_parts: list[dict[str, Any]] = []

    for item in content:
        if isinstance(item, TextContent):
            google_parts.append({"text": item.text})
        elif isinstance(item, ImageContent):
            google_parts.append(await _convert_image_to_google(item))

    return google_parts


async def convert_messages_to_google(
    messages: list[ChatMessage],
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Convert casual-llm ChatMessage list to Google Generative AI format.

    Handles all message types. System messages are extracted and returned
    separately since Gemini takes system instructions as a separate parameter.

    Google uses a parts-based format where each message has a "role" and "parts":
    {"role": "user", "parts": [{"text": "..."}, {"inline_data": {...}}]}

    Args:
        messages: List of ChatMessage objects

    Returns:
        A tuple of (messages, system_instruction) where:
            - messages: List of dictionaries in Google GenerativeAI format
            - system_instruction: The system message content (if any), or None

    Raises:
        ImageFetchError: If a URL image cannot be fetched.

    Examples:
        >>> import asyncio
        >>> from casual_llm import UserMessage, SystemMessage
        >>> messages = [SystemMessage(content="Be helpful"), UserMessage(content="Hello")]
        >>> google_msgs, system = asyncio.run(convert_messages_to_google(messages))
        >>> google_msgs[0]["role"]
        'user'
        >>> system
        'Be helpful'
    """
    if not messages:
        return [], None

    logger.debug(f"Converting {len(messages)} messages to Google format")

    google_messages: list[dict[str, Any]] = []
    system_instruction: str | None = None

    for msg in messages:
        match msg.role:
            case "assistant":
                # Handle assistant messages (model role in Google)
                parts: list[dict[str, Any]] = []

                # Add text content if present
                if msg.content:
                    parts.append({"text": msg.content})

                # Add function call parts if tool_calls present
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        import json

                        try:
                            args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            args = {}

                        parts.append(
                            {
                                "function_call": {
                                    "name": tool_call.function.name,
                                    "args": args,
                                }
                            }
                        )

                google_messages.append(
                    {
                        "role": "model",
                        "parts": parts if parts else [{"text": ""}],
                    }
                )

            case "system":
                # Google takes system instruction as separate parameter
                # Use the last system message if multiple are present
                system_instruction = msg.content

            case "tool":
                # Google uses function_response in parts for tool results
                google_messages.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "function_response": {
                                    "name": msg.name,
                                    "response": {"content": msg.content},
                                }
                            }
                        ],
                    }
                )

            case "user":
                google_messages.append(
                    {
                        "role": "user",
                        "parts": await _convert_user_content_to_google_parts(msg.content),
                    }
                )

            case _:
                logger.warning(f"Unknown message role: {msg.role}")

    return google_messages, system_instruction


__all__ = [
    "convert_messages_to_google",
]
