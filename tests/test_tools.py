"""Tests for tool models and converters."""

import pytest
from casual_llm.tools import Tool, ToolParameter
from casual_llm.tool_converters import (
    tool_to_ollama,
    tools_to_ollama,
    tool_to_openai,
    tools_to_openai,
)


class TestToolParameter:
    """Tests for ToolParameter model."""

    def test_simple_parameter(self):
        """Test creating a simple string parameter."""
        param = ToolParameter(
            type="string",
            description="A test parameter"
        )
        assert param.type == "string"
        assert param.description == "A test parameter"
        assert param.enum is None

    def test_enum_parameter(self):
        """Test creating a parameter with enum values."""
        param = ToolParameter(
            type="string",
            description="Temperature units",
            enum=["celsius", "fahrenheit"]
        )
        assert param.enum == ["celsius", "fahrenheit"]

    def test_nested_object_parameter(self):
        """Test creating a parameter with nested properties."""
        param = ToolParameter(
            type="object",
            description="A location object",
            properties={
                "city": ToolParameter(type="string"),
                "country": ToolParameter(type="string")
            },
            required=["city"]
        )
        assert param.type == "object"
        assert "city" in param.properties
        assert param.required == ["city"]

    def test_array_parameter(self):
        """Test creating an array parameter."""
        param = ToolParameter(
            type="array",
            description="List of items",
            items={"type": "string"}
        )
        assert param.type == "array"
        assert param.items == {"type": "string"}

    def test_parameter_with_default(self):
        """Test parameter with default value."""
        param = ToolParameter(
            type="string",
            description="Format",
            default="json"
        )
        assert param.default == "json"

    def test_parameter_serialization(self):
        """Test parameter can be serialized without None values."""
        param = ToolParameter(
            type="string",
            description="Test",
            enum=["a", "b"]
        )
        dumped = param.model_dump(exclude_none=True)
        assert "default" not in dumped
        assert "items" not in dumped
        assert dumped["type"] == "string"
        assert dumped["enum"] == ["a", "b"]

    def test_extra_fields_allowed(self):
        """Test that extra JSON Schema fields are allowed."""
        param = ToolParameter(
            type="string",
            description="Test",
            minLength=1,
            maxLength=100,
            pattern="^[a-z]+$"
        )
        dumped = param.model_dump(exclude_none=True)
        assert dumped["minLength"] == 1
        assert dumped["maxLength"] == 100
        assert dumped["pattern"] == "^[a-z]+$"


class TestTool:
    """Tests for Tool model."""

    def test_simple_tool(self):
        """Test creating a simple tool with no parameters."""
        tool = Tool(
            name="get_time",
            description="Get current time"
        )
        assert tool.name == "get_time"
        assert tool.description == "Get current time"
        assert tool.parameters == {}
        assert tool.required == []

    def test_tool_name_flexibility(self):
        """Test that tool names support various naming conventions."""
        # Tool names are API strings and should support various conventions
        valid_names = [
            "get_weather",  # snake_case (Python style)
            "get-weather",  # kebab-case (common in APIs)
            "weather.get",  # dotted notation (namespaced)
            "GetWeather",   # PascalCase
            "getWeather",   # camelCase
            "tool123",      # with numbers
            "_private",     # underscore prefix
        ]
        for name in valid_names:
            tool = Tool(name=name, description="Test")
            assert tool.name == name

    def test_tool_with_parameters(self):
        """Test creating a tool with parameters."""
        tool = Tool(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "location": ToolParameter(
                    type="string",
                    description="City name"
                ),
                "units": ToolParameter(
                    type="string",
                    description="Temperature units",
                    enum=["celsius", "fahrenheit"]
                )
            },
            required=["location"]
        )
        assert len(tool.parameters) == 2
        assert "location" in tool.parameters
        assert "units" in tool.parameters
        assert tool.required == ["location"]

    def test_input_schema_property(self):
        """Test that input_schema property generates correct schema."""
        tool = Tool(
            name="search",
            description="Search for items",
            parameters={
                "query": ToolParameter(
                    type="string",
                    description="Search query"
                ),
                "limit": ToolParameter(
                    type="number",
                    description="Max results",
                    default=10
                )
            },
            required=["query"]
        )

        schema = tool.input_schema
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"
        assert schema["properties"]["limit"]["default"] == 10
        assert schema["required"] == ["query"]

    def test_from_input_schema(self):
        """Test creating a Tool from an input schema."""
        input_schema = {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                },
                "country": {
                    "type": "string",
                    "description": "Country code"
                }
            },
            "required": ["city"]
        }

        tool = Tool.from_input_schema(
            name="get_weather",
            description="Get weather data",
            input_schema=input_schema
        )

        assert tool.name == "get_weather"
        assert tool.description == "Get weather data"
        assert "city" in tool.parameters
        assert "country" in tool.parameters
        assert tool.parameters["city"].type == "string"
        assert tool.required == ["city"]

    def test_from_input_schema_minimal(self):
        """Test from_input_schema with minimal schema."""
        tool = Tool.from_input_schema(
            name="ping",
            description="Ping test",
            input_schema={}
        )
        assert tool.name == "ping"
        assert tool.parameters == {}
        assert tool.required == []

    def test_roundtrip_input_schema(self):
        """Test that input_schema can roundtrip through from_input_schema."""
        original_tool = Tool(
            name="test",
            description="Test tool",
            parameters={
                "param1": ToolParameter(type="string", description="First param"),
                "param2": ToolParameter(type="number", default=5)
            },
            required=["param1"]
        )

        schema = original_tool.input_schema
        reconstructed = Tool.from_input_schema(
            name="test",
            description="Test tool",
            input_schema=schema
        )

        assert reconstructed.name == original_tool.name
        assert reconstructed.description == original_tool.description
        assert reconstructed.required == original_tool.required
        assert len(reconstructed.parameters) == len(original_tool.parameters)


class TestOllamaConverters:
    """Tests for Ollama format converters."""

    def test_tool_to_ollama_simple(self):
        """Test converting a simple tool to Ollama format."""
        tool = Tool(
            name="get_time",
            description="Get current time"
        )

        ollama_tool = tool_to_ollama(tool)

        assert ollama_tool["type"] == "function"
        assert ollama_tool["function"]["name"] == "get_time"
        assert ollama_tool["function"]["description"] == "Get current time"
        assert ollama_tool["function"]["parameters"]["type"] == "object"
        assert ollama_tool["function"]["parameters"]["properties"] == {}
        assert ollama_tool["function"]["parameters"]["required"] == []

    def test_tool_to_ollama_with_params(self):
        """Test converting a tool with parameters to Ollama format."""
        tool = Tool(
            name="search",
            description="Search items",
            parameters={
                "query": ToolParameter(type="string", description="Search query"),
                "limit": ToolParameter(type="number", default=10)
            },
            required=["query"]
        )

        ollama_tool = tool_to_ollama(tool)

        params = ollama_tool["function"]["parameters"]
        assert "query" in params["properties"]
        assert "limit" in params["properties"]
        assert params["properties"]["query"]["type"] == "string"
        assert params["properties"]["limit"]["default"] == 10
        assert params["required"] == ["query"]

    def test_tools_to_ollama(self):
        """Test converting multiple tools to Ollama format."""
        tools = [
            Tool(name="tool1", description="First tool"),
            Tool(name="tool2", description="Second tool")
        ]

        ollama_tools = tools_to_ollama(tools)

        assert len(ollama_tools) == 2
        assert ollama_tools[0]["function"]["name"] == "tool1"
        assert ollama_tools[1]["function"]["name"] == "tool2"

    def test_tools_to_ollama_empty(self):
        """Test converting empty tool list."""
        assert tools_to_ollama([]) == []


class TestOpenAIConverters:
    """Tests for OpenAI format converters."""

    def test_tool_to_openai_simple(self):
        """Test converting a simple tool to OpenAI format."""
        tool = Tool(
            name="get_time",
            description="Get current time"
        )

        openai_tool = tool_to_openai(tool)

        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == "get_time"
        assert openai_tool["function"]["description"] == "Get current time"
        assert openai_tool["function"]["parameters"]["type"] == "object"
        assert openai_tool["function"]["parameters"]["properties"] == {}
        assert openai_tool["function"]["parameters"]["required"] == []

    def test_tool_to_openai_with_params(self):
        """Test converting a tool with parameters to OpenAI format."""
        tool = Tool(
            name="calculator",
            description="Perform calculation",
            parameters={
                "operation": ToolParameter(
                    type="string",
                    enum=["add", "subtract", "multiply", "divide"]
                ),
                "a": ToolParameter(type="number"),
                "b": ToolParameter(type="number")
            },
            required=["operation", "a", "b"]
        )

        openai_tool = tool_to_openai(tool)

        params = openai_tool["function"]["parameters"]
        assert len(params["properties"]) == 3
        assert params["properties"]["operation"]["enum"] == ["add", "subtract", "multiply", "divide"]
        assert params["required"] == ["operation", "a", "b"]

    def test_tools_to_openai(self):
        """Test converting multiple tools to OpenAI format."""
        tools = [
            Tool(name="tool1", description="First tool"),
            Tool(name="tool2", description="Second tool")
        ]

        openai_tools = tools_to_openai(tools)

        assert len(openai_tools) == 2
        assert openai_tools[0]["function"]["name"] == "tool1"
        assert openai_tools[1]["function"]["name"] == "tool2"

    def test_tools_to_openai_empty(self):
        """Test converting empty tool list."""
        assert tools_to_openai([]) == []

    def test_openai_ollama_equivalence(self):
        """Test that OpenAI and Ollama formats are equivalent."""
        tool = Tool(
            name="test",
            description="Test tool",
            parameters={
                "param": ToolParameter(type="string")
            },
            required=["param"]
        )

        openai_tool = tool_to_openai(tool)
        ollama_tool = tool_to_ollama(tool)

        # The formats should be identical
        assert openai_tool == ollama_tool
