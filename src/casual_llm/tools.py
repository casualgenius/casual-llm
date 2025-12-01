"""
Tool definitions for LLM function calling.

Provides unified tool models compatible with Ollama, OpenAI, and MCP.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """
    A single tool parameter definition following JSON Schema.

    Supports common JSON Schema fields for describing function parameters.
    """

    type: str = Field(..., description="JSON Schema type (string, number, object, array, etc.)")
    description: str | None = Field(None, description="Parameter description")
    enum: list[Any] | None = Field(None, description="Allowed values for enum types")
    items: dict[str, Any] | None = Field(None, description="Array item schema")
    properties: dict[str, ToolParameter] | None = Field(
        None, description="Object properties for nested objects"
    )
    required: list[str] | None = Field(None, description="Required properties for nested objects")
    default: Any | None = Field(None, description="Default value")

    model_config = {"extra": "allow"}  # Allow additional JSON Schema fields


class Tool(BaseModel):
    """
    Unified tool definition compatible with Ollama, OpenAI, and MCP.

    Represents a function that an LLM can call, with a JSON Schema
    describing its parameters.

    Examples:
        >>> tool = Tool(
        ...     name="get_weather",
        ...     description="Get current weather for a location",
        ...     parameters={
        ...         "location": ToolParameter(
        ...             type="string",
        ...             description="City name"
        ...         ),
        ...         "units": ToolParameter(
        ...             type="string",
        ...             enum=["celsius", "fahrenheit"],
        ...             default="celsius"
        ...         )
        ...     },
        ...     required=["location"]
        ... )
    """

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="What the tool does")
    parameters: dict[str, ToolParameter] = Field(
        default_factory=dict, description="Tool parameters as JSON Schema properties"
    )
    required: list[str] = Field(
        default_factory=list, description="List of required parameter names"
    )

    @property
    def input_schema(self) -> dict[str, Any]:
        """
        Get the full JSON Schema for tool inputs.

        Returns a schema compatible with MCP's inputSchema format.

        Returns:
            Dictionary with 'type', 'properties', and 'required' keys
        """
        return {
            "type": "object",
            "properties": {
                name: param.model_dump(exclude_none=True) for name, param in self.parameters.items()
            },
            "required": self.required,
        }

    @classmethod
    def from_input_schema(cls, name: str, description: str, input_schema: dict[str, Any]) -> "Tool":
        """
        Create a Tool from an MCP-style inputSchema.

        Args:
            name: Tool name
            description: Tool description
            input_schema: JSON Schema with 'properties' and optional 'required'

        Returns:
            Tool instance

        Examples:
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "city": {"type": "string", "description": "City name"}
            ...     },
            ...     "required": ["city"]
            ... }
            >>> tool = Tool.from_input_schema("weather", "Get weather", schema)
        """
        properties = input_schema.get("properties", {})
        parameters = {
            param_name: ToolParameter(**param_def) for param_name, param_def in properties.items()
        }

        return cls(
            name=name,
            description=description,
            parameters=parameters,
            required=input_schema.get("required", []),
        )


# Re-export at module level
__all__ = ["Tool", "ToolParameter"]
