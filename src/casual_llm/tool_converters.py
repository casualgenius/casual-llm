"""
Converters for tool definitions between different formats.

Supports conversion from casual-llm Tool format to:
- Ollama tool format
- OpenAI ChatCompletionToolParam format
- MCP Tool format (for interop with MCP libraries)
"""

import logging
from typing import Any

from casual_llm.tools import Tool

logger = logging.getLogger(__name__)


# Ollama format converters
def tool_to_ollama(tool: Tool) -> dict[str, Any]:
    """
    Convert a casual-llm Tool to Ollama tool format.

    Args:
        tool: Tool to convert

    Returns:
        Dictionary in Ollama's expected format

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
        List of Ollama-formatted tool dictionaries

    Examples:
        >>> tools = [Tool(name="t1", description="d1", parameters={}, required=[])]
        >>> ollama_tools = tools_to_ollama(tools)
        >>> len(ollama_tools)
        1
    """
    logger.debug(f"Converting {len(tools)} tools to Ollama format")
    return [tool_to_ollama(tool) for tool in tools]


# OpenAI format converters
def tool_to_openai(tool: Tool) -> dict[str, Any]:
    """
    Convert a casual-llm Tool to OpenAI ChatCompletionToolParam format.

    Args:
        tool: Tool to convert

    Returns:
        Dictionary in OpenAI's expected format

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


def tools_to_openai(tools: list[Tool]) -> list[dict[str, Any]]:
    """
    Convert multiple casual-llm Tools to OpenAI format.

    Args:
        tools: List of tools to convert

    Returns:
        List of OpenAI ChatCompletionToolParam dictionaries

    Examples:
        >>> tools = [Tool(name="t1", description="d1", parameters={}, required=[])]
        >>> openai_tools = tools_to_openai(tools)
        >>> len(openai_tools)
        1
    """
    logger.debug(f"Converting {len(tools)} tools to OpenAI format")
    return [tool_to_openai(tool) for tool in tools]


__all__ = [
    "tool_to_ollama",
    "tools_to_ollama",
    "tool_to_openai",
    "tools_to_openai",
]
