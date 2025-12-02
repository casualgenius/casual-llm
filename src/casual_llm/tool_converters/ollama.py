"""
Ollama tool converters.

Converts casual-llm Tool format to Ollama tool format.
"""

import logging
from typing import TYPE_CHECKING, Any

from casual_llm.tools import Tool

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def tool_to_ollama(tool: Tool) -> dict[str, Any]:
    """
    Convert a casual-llm Tool to Ollama tool format.

    Args:
        tool: Tool to convert

    Returns:
        Dictionary conforming to ollama._types.Tool structure.
        The Ollama SDK accepts dicts as Mapping[str, Any] for tools.

    Examples:
        >>> tool = Tool(name="weather", description="Get weather", parameters={}, required=[])
        >>> ollama_tool = tool_to_ollama(tool)
        >>> ollama_tool["type"]
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


def tools_to_ollama(tools: list[Tool]) -> list[dict[str, Any]]:
    """
    Convert multiple casual-llm Tools to Ollama format.

    Args:
        tools: List of tools to convert

    Returns:
        List of dictionaries conforming to ollama._types.Tool structure.
        The Ollama SDK accepts dicts as Mapping[str, Any] for tools.

    Examples:
        >>> tools = [Tool(name="t1", description="d1", parameters={}, required=[])]
        >>> ollama_tools = tools_to_ollama(tools)
        >>> len(ollama_tools)
        1
    """
    logger.debug(f"Converting {len(tools)} tools to Ollama format")
    return [tool_to_ollama(tool) for tool in tools]


__all__ = [
    "tool_to_ollama",
    "tools_to_ollama",
]
