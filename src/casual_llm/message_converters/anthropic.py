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
)

if TYPE_CHECKING:
    from anthropic.types import ToolUseBlock

logger = logging.getLogger(__name__)


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

    logger.debug(f"Converting {len(messages)} messages to Anthropic format")

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
                                f"Failed to parse tool call arguments: {tool_call.function.arguments}"
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
                # User messages with simple text content
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": msg.content or ""}],
                    }
                )

            case _:
                logger.warning(f"Unknown message role: {msg.role}")

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
        logger.debug(f"Converting tool call: {tool.name}")

        # Serialize input dict to JSON string for casual-llm format
        arguments = json.dumps(tool.input) if tool.input else "{}"

        tool_call = AssistantToolCall(
            id=tool.id,
            type="function",
            function=AssistantToolCallFunction(
                name=tool.name, arguments=arguments
            ),
        )
        tool_calls.append(tool_call)

    logger.debug(f"Converted {len(tool_calls)} tool calls")
    return tool_calls


__all__ = [
    "convert_messages_to_anthropic",
    "extract_system_message",
    "convert_tool_calls_from_anthropic",
]
