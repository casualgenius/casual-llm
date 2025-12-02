"""
OpenAI tool converters.

Converts casual-llm Tool format to OpenAI ChatCompletionToolParam format.
"""

import logging
from typing import TYPE_CHECKING, Any

from casual_llm.tools import Tool

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionToolParam

logger = logging.getLogger(__name__)


def tool_to_openai(tool: Tool) -> "ChatCompletionToolParam":
    """
    Convert a casual-llm Tool to OpenAI ChatCompletionToolParam format.

    Args:
        tool: Tool to convert

    Returns:
        ChatCompletionToolParam dictionary in OpenAI's expected format

    Examples:
        >>> tool = Tool(name="weather", description="Get weather", parameters={}, required=[])
        >>> openai_tool = tool_to_openai(tool)
        >>> openai_tool["type"]
        'function'
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": {
                    name: param.model_dump(exclude_none=True)
                    for name, param in tool.parameters.items()
                },
                "required": tool.required,
            },
        },
    }


def tools_to_openai(tools: list[Tool]) -> list["ChatCompletionToolParam"]:
    """
    Convert multiple casual-llm Tools to OpenAI format.

    Args:
        tools: List of tools to convert

    Returns:
        List of ChatCompletionToolParam dictionaries

    Examples:
        >>> tools = [Tool(name="t1", description="d1", parameters={}, required=[])]
        >>> openai_tools = tools_to_openai(tools)
        >>> len(openai_tools)
        1
    """
    logger.debug(f"Converting {len(tools)} tools to OpenAI format")
    return [tool_to_openai(tool) for tool in tools]


__all__ = [
    "tool_to_openai",
    "tools_to_openai",
]
