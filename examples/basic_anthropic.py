"""
Basic Anthropic example using Claude models.

This example demonstrates basic usage of the AnthropicClient + Model with
text responses and JSON output.

Requirements:
- ANTHROPIC_API_KEY environment variable
- pip install casual-llm[anthropic]
"""

import asyncio
import os
from casual_llm import (
    AnthropicClient,
    ChatOptions,
    Model,
    UserMessage,
    SystemMessage,
)


# Configuration from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", None)
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


async def basic_text_response():
    """Example: Basic text response from Claude."""
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY=your-api-key")
        return

    print("=" * 50)
    print("Basic Text Response Example")
    print("=" * 50)

    # Create Anthropic client (manages API connection)
    client = AnthropicClient(api_key=ANTHROPIC_API_KEY)

    # Create model (configure model name and parameters)
    model = Model(client, name=ANTHROPIC_MODEL, default_options=ChatOptions(temperature=0.7))

    # Create messages with system and user messages
    messages = [
        SystemMessage(content="You are a helpful assistant who gives concise answers."),
        UserMessage(content="What is the capital of France?"),
    ]

    print(f"Sending message to {ANTHROPIC_MODEL}...")
    response = await model.chat(messages, ChatOptions(response_format="text"))

    print(f"\nResponse:\n{response.content}")

    # Show usage statistics
    usage = model.get_usage()
    if usage:
        print("\nUsage:")
        print(f"  Prompt tokens: {usage.prompt_tokens}")
        print(f"  Completion tokens: {usage.completion_tokens}")
        print(f"  Total tokens: {usage.total_tokens}")

    return client  # Return client for reuse


async def json_response_example(client):
    """Example: Request JSON-formatted response from Claude."""
    print("\n" + "=" * 50)
    print("JSON Response Example")
    print("=" * 50)

    # Reuse the same client for a different model configuration
    model = Model(client, name=ANTHROPIC_MODEL, default_options=ChatOptions(temperature=0.7))

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content="List three programming languages and their primary use cases. "
            "Format as JSON with 'name' and 'use_case' fields."
        ),
    ]

    print(f"Requesting JSON response from {ANTHROPIC_MODEL}...")
    response = await model.chat(messages, ChatOptions(response_format="json"))

    print(f"\nJSON Response:\n{response.content}")

    # Show usage
    usage = model.get_usage()
    if usage:
        print(f"\nUsage: {usage.total_tokens} tokens")


async def pydantic_model_example(client):
    """Example: Use Pydantic model for structured output."""
    print("\n" + "=" * 50)
    print("Pydantic Model Example")
    print("=" * 50)

    from pydantic import BaseModel

    class Person(BaseModel):
        """A person with basic information."""

        name: str
        age: int
        occupation: str
        city: str

    # Reuse the same client
    model = Model(client, name=ANTHROPIC_MODEL, default_options=ChatOptions(temperature=0.7))

    messages = [
        UserMessage(
            content="Create a fictional person who is a software engineer in San Francisco."
        ),
    ]

    print(f"Requesting structured output from {ANTHROPIC_MODEL}...")
    print("Using Pydantic model: Person(name, age, occupation, city)")
    response = await model.chat(messages, ChatOptions(response_format=Person))

    print(f"\nStructured Response:\n{response.content}")

    # Show usage
    usage = model.get_usage()
    if usage:
        print(f"\nUsage: {usage.total_tokens} tokens")


async def conversation_example(client):
    """Example: Multi-turn conversation with Claude."""
    print("\n" + "=" * 50)
    print("Multi-turn Conversation Example")
    print("=" * 50)

    from casual_llm import AssistantMessage

    # Reuse the same client
    model = Model(client, name=ANTHROPIC_MODEL, default_options=ChatOptions(temperature=0.7))

    # First message
    messages = [
        SystemMessage(content="You are a helpful coding assistant."),
        UserMessage(content="What is a Python list comprehension?"),
    ]

    print("User: What is a Python list comprehension?")
    response1 = await model.chat(messages, ChatOptions(response_format="text"))
    print(f"\nClaude: {response1.content}")

    # Second message - continue the conversation
    messages.append(AssistantMessage(content=response1.content))
    messages.append(UserMessage(content="Can you show me a simple example?"))

    print("\nUser: Can you show me a simple example?")
    response2 = await model.chat(messages, ChatOptions(response_format="text"))
    print(f"\nClaude: {response2.content}")

    # Show total usage
    usage = model.get_usage()
    if usage:
        print(f"\nTotal usage: {usage.total_tokens} tokens")


async def multi_model_example(client):
    """Example: Multiple models using the same client."""
    print("\n" + "=" * 50)
    print("Multi-Model Example")
    print("=" * 50)

    # You can create multiple models from the same client
    # This is efficient as the client connection is reused
    print("You can create multiple Model instances from a single AnthropicClient!")
    print()
    print("# One client, multiple models:")
    print("client = AnthropicClient(api_key=api_key)")
    print(
        "sonnet = Model(client, name='claude-3-5-sonnet-latest', default_options=ChatOptions(temperature=0.7))"
    )
    print(
        "haiku = Model(client, name='claude-3-haiku-20240307', default_options=ChatOptions(temperature=0.5))"
    )
    print()
    print("# Each model tracks its own usage:")
    print("response1 = await sonnet.chat(messages)")
    print("response2 = await haiku.chat(messages)")
    print("sonnet.get_usage()  # Usage for sonnet model")
    print("haiku.get_usage()   # Usage for haiku model")


async def main():
    """Run all Anthropic examples."""
    print("casual-llm Anthropic (Claude) Examples")
    print("This demonstrates basic usage of the AnthropicClient + Model.\n")

    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY=your-api-key")
        return

    # Create a single client to reuse across examples
    client = AnthropicClient(api_key=ANTHROPIC_API_KEY)

    # Run basic text example
    await basic_text_response()

    # Run JSON example (reusing client)
    await json_response_example(client)

    # Run Pydantic model example (reusing client)
    await pydantic_model_example(client)

    # Run conversation example (reusing client)
    await conversation_example(client)

    # Show multi-model pattern
    await multi_model_example(client)

    print("\n" + "=" * 50)
    print("Anthropic examples complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
