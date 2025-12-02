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
)

if TYPE_CHECKING:
    from ollama._types import Message

logger = logging.getLogger(__name__)


def convert_messages_to_ollama(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """
    Convert casual-llm ChatMessage list to Ollama format.

    Unlike OpenAI which expects tool call arguments as JSON strings,
    Ollama expects them as dictionaries. This function handles that conversion.

    Args:
        messages: List of ChatMessage objects

    Returns:
        List of dictionaries in Ollama message format

    Examples:
        >>> from casual_llm import UserMessage
        >>> messages = [UserMessage(content="Hello")]
        >>> ollama_msgs = convert_messages_to_ollama(messages)
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
                ollama_messages.append({"role": "user", "content": msg.content})

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
