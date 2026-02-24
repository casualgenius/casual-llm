"""
Example demonstrating tool calling with Ollama or OpenAI clients.

Shows how to:
1. Define tools using the Tool model
2. Make LLM requests with tools
3. Handle tool calls from the LLM
4. Execute tools and return results
5. Continue the conversation with tool results
"""

import asyncio
import json
from casual_llm import (
    ChatOptions,
    Tool,
    ToolParameter,
    UserMessage,
    ToolResultMessage,
    SystemMessage,
    OllamaClient,
    Model,
)

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"

# Create client and model
client = OllamaClient(host=OLLAMA_HOST)
model = Model(client, name=OLLAMA_MODEL)


# Define example tools
def get_current_weather(location: str, units: str = "celsius") -> dict:
    """
    Mock function to get weather.
    In a real application, this would call a weather API.
    """
    print(f"  [TOOL CALL] get_current_weather(location='{location}', units='{units}')")

    # Mock weather data
    temps = {
        "celsius": {"san francisco": 18, "tokyo": 22, "paris": 15},
        "fahrenheit": {"san francisco": 64, "tokyo": 72, "paris": 59},
    }

    temp = temps.get(units.lower(), temps["celsius"]).get(location.lower(), 20)

    return {"location": location, "temperature": temp, "units": units, "condition": "partly cloudy"}


def calculate(operation: str, a: float, b: float) -> dict:
    """Mock calculator function."""
    print(f"  [TOOL CALL] calculate(operation='{operation}', a={a}, b={b})")

    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "error: division by zero",
    }

    result = operations.get(operation, "error: unknown operation")

    return {"operation": operation, "a": a, "b": b, "result": result}


# Tool definitions for the LLM
weather_tool = Tool(
    name="get_current_weather",
    description="Get the current weather for a location",
    parameters={
        "location": ToolParameter(type="string", description="The city name, e.g. 'San Francisco'"),
        "units": ToolParameter(
            type="string",
            description="Temperature units",
            enum=["celsius", "fahrenheit"],
        ),
    },
    required=["location"],
)

calculator_tool = Tool(
    name="calculate",
    description="Perform basic arithmetic operations",
    parameters={
        "operation": ToolParameter(
            type="string",
            description="The operation to perform",
            enum=["add", "subtract", "multiply", "divide"],
        ),
        "a": ToolParameter(type="number", description="First number"),
        "b": ToolParameter(type="number", description="Second number"),
    },
    required=["operation", "a", "b"],
)


# Map tool names to functions
TOOL_FUNCTIONS = {
    "get_current_weather": get_current_weather,
    "calculate": calculate,
}


async def run_tool_calling_example():
    """
    Run a tool calling conversation example.

    This demonstrates the full tool calling workflow:
    1. User asks a question
    2. LLM decides to call tools
    3. We execute the tools
    4. Return results to LLM
    5. LLM provides final answer
    """
    print("=" * 70)
    print("  Tool Calling Example")
    print("=" * 70)
    print()

    # Initialize model (using Ollama - replace with OpenAIClient as needed)
    print("Initializing Ollama model...")
    print()

    # Define tools available to the LLM
    tools = [weather_tool, calculator_tool]

    # Start conversation
    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant with access to tools. "
                "When you need information, use the available tools."
            )
        ),
        UserMessage(
            content=("What's the weather like in San Francisco and Tokyo? " "Also, what's 25 + 17?")
        ),
    ]

    print("User: " + messages[1].content)
    print()

    # Initial request with tools
    print("Sending request to LLM with tools...")
    response = await model.chat(messages, ChatOptions(tools=tools))
    print()

    # Check if LLM wants to call tools
    if response.tool_calls:
        print(f"LLM requested {len(response.tool_calls)} tool call(s):")
        print()

        # Add assistant's response (with tool calls) to conversation
        messages.append(response)

        # Execute each tool call
        for tool_call in response.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"Tool: {function_name}")
            print(f"Arguments: {json.dumps(function_args, indent=2)}")

            # Execute the tool
            if function_name in TOOL_FUNCTIONS:
                tool_function = TOOL_FUNCTIONS[function_name]
                tool_result = tool_function(**function_args)

                # Add tool result to conversation
                tool_result_message = ToolResultMessage(
                    name=function_name, tool_call_id=tool_call.id, content=json.dumps(tool_result)
                )
                messages.append(tool_result_message)

                print(f"Result: {json.dumps(tool_result, indent=2)}")
                print()
            else:
                print(f"ERROR: Unknown tool '{function_name}'")
                print()

        # Send tool results back to LLM for final response
        print("Sending tool results back to LLM...")
        final_response = await model.chat(messages, ChatOptions(tools=tools))
        print()

        print("Assistant (final response):")
        print(final_response.content)

    else:
        # LLM responded directly without tools
        print("Assistant (no tools used):")
        print(response.content)

    print()
    print("=" * 70)
    print("  Example Complete")
    print("=" * 70)


async def run_simple_no_tools_example():
    """
    Example showing regular chat without tools.

    This demonstrates that the chat() method works without tools,
    always returning an AssistantMessage.
    """
    print()
    print("=" * 70)
    print("  Simple Example (No Tools)")
    print("=" * 70)
    print()

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="Tell me a fun fact about Python programming."),
    ]

    print("User: " + messages[1].content)
    print()

    # Call without tools - returns AssistantMessage
    response = await model.chat(messages)

    print("Assistant:")
    print(response.content)
    print()
    print("=" * 70)


async def main():
    """Run both examples."""
    # Example 1: No tools (simple, backward compatible)
    await run_simple_no_tools_example()

    print("\n\n")

    # Example 2: With tools (new functionality)
    try:
        await run_tool_calling_example()
    except Exception as e:
        print(f"\nNote: Tool calling example failed: {e}")
        print("This is expected if:")
        print("  1. Ollama is not running")
        print("  2. The model doesn't support tool calling")
        print("  3. Network issues")
        print("\nFor OpenAI, replace OllamaClient with OpenAIClient:")
        print("  from casual_llm import OpenAIClient, Model")
        print("  client = OpenAIClient(api_key='...')")
        print("  model = Model(client, name='gpt-4o-mini')")


if __name__ == "__main__":
    asyncio.run(main())
