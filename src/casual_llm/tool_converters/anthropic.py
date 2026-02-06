"""
Anthropic tool converters.

Converts casual-llm Tool format to Anthropic ToolParam format.
"""

import logging
from typing import TYPE_CHECKING

from casual_llm.tools import Tool

if TYPE_CHECKING:
    from anthropic.types import ToolParam

logger = logging.getLogger(__name__)


def tool_to_anthropic(tool: Tool) -> "ToolParam":
    """
    Convert a casual-llm Tool to Anthropic ToolParam format.

    Args:
        tool: Tool to convert

    Returns:
        ToolParam dictionary in Anthropic's expected format

    Examples:
        >>> tool = Tool(name="weather", description="Get weather", parameters={}, required=[])
        >>> anthropic_tool = tool_to_anthropic(tool)
        >>> anthropic_tool["name"]
        'weather'
    """
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": {
            "type": "object",
            "properties": {
                name: param.model_dump(exclude_none=True) for name, param in tool.parameters.items()
            },
            "required": tool.required,
        },
    }


def tools_to_anthropic(tools: list[Tool]) -> list["ToolParam"]:
    """
    Convert multiple casual-llm Tools to Anthropic format.

    Args:
        tools: List of tools to convert

    Returns:
        List of ToolParam dictionaries

    Examples:
        >>> tools = [Tool(name="t1", description="d1", parameters={}, required=[])]
        >>> anthropic_tools = tools_to_anthropic(tools)
        >>> len(anthropic_tools)
        1
    """
    logger.debug("Converting %d tools to Anthropic format", len(tools))
    return [tool_to_anthropic(tool) for tool in tools]


__all__ = [
    "tool_to_anthropic",
    "tools_to_anthropic",
]
