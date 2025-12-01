"""
Message converters for different LLM provider formats.

Converts casual-llm ChatMessage format to provider-specific formats (OpenAI, Ollama).
"""

import logging
from typing import Any

from casual_llm.messages import (
    ChatMessage,
    AssistantToolCall,
    AssistantToolCallFunction,
)

logger = logging.getLogger(__name__)


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
                openai_messages.append({"role": "user", "content": msg.content})

            case _:
                logger.warning(f"Unknown message role: {msg.role}")

    return openai_messages


def convert_messages_to_ollama(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """
    Convert casual-llm ChatMessage list to Ollama format.

    Ollama uses the same format as OpenAI, so this is currently an alias.

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
    logger.debug(f"Converting {len(messages)} messages to Ollama format")
    # Ollama uses the same message format as OpenAI
    return convert_messages_to_openai(messages)


def convert_tool_calls_from_openai(
    response_tool_calls: list[Any],
) -> list[AssistantToolCall]:
    """
    Convert OpenAI ChatCompletionMessageToolCall to casual-llm format.

    Args:
        response_tool_calls: List of tool calls from OpenAI response

    Returns:
        List of AssistantToolCall objects

    Examples:
        >>> # Assuming response has tool_calls
        >>> # tool_calls = convert_tool_calls_from_openai(response.message.tool_calls)
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


def convert_tool_calls_from_ollama(
    response_tool_calls: list[dict[str, Any]],
) -> list[AssistantToolCall]:
    """
    Convert Ollama tool calls to casual-llm format.

    Ollama returns tool calls in the same format as OpenAI.

    Args:
        response_tool_calls: List of tool call dictionaries from Ollama response

    Returns:
        List of AssistantToolCall objects

    Examples:
        >>> # tool_calls = convert_tool_calls_from_ollama(response["message"]["tool_calls"])
        >>> # assert len(tool_calls) > 0
        pass
    """
    tool_calls = []

    for tool in response_tool_calls:
        logger.debug(f"Converting tool call: {tool['function']['name']}")

        tool_call = AssistantToolCall(
            id=tool["id"],
            type=tool.get("type", "function"),
            function=AssistantToolCallFunction(
                name=tool["function"]["name"], arguments=tool["function"]["arguments"]
            ),
        )
        tool_calls.append(tool_call)

    logger.debug(f"Converted {len(tool_calls)} tool calls")
    return tool_calls


__all__ = [
    "convert_messages_to_openai",
    "convert_messages_to_ollama",
    "convert_tool_calls_from_openai",
    "convert_tool_calls_from_ollama",
]
