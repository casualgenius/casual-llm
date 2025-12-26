"""
Basic Anthropic example using Claude models.

This example demonstrates basic usage of the AnthropicProvider with
text responses and JSON output.

Requirements:
- ANTHROPIC_API_KEY environment variable
- pip install casual-llm[anthropic]
"""

import asyncio
import os
from casual_llm import (
    create_provider,
    ModelConfig,
    Provider,
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

    # Create provider with model configuration
    config = ModelConfig(
        name=ANTHROPIC_MODEL,
        provider=Provider.ANTHROPIC,
        api_key=ANTHROPIC_API_KEY,
        temperature=0.7,
    )

    provider = create_provider(config)

    # Create messages with system and user messages
    messages = [
        SystemMessage(content="You are a helpful assistant who gives concise answers."),
        UserMessage(content="What is the capital of France?"),
    ]

    print(f"Sending message to {ANTHROPIC_MODEL}...")
    response = await provider.chat(messages, response_format="text")

    print(f"\nResponse:\n{response.content}")

    # Show usage statistics
    usage = provider.get_usage()
    if usage:
        print("\nUsage:")
        print(f"  Prompt tokens: {usage.prompt_tokens}")
        print(f"  Completion tokens: {usage.completion_tokens}")
        print(f"  Total tokens: {usage.total_tokens}")


async def json_response_example():
    """Example: Request JSON-formatted response from Claude."""
    if not ANTHROPIC_API_KEY:
        print("\nError: ANTHROPIC_API_KEY environment variable not set")
        return

    print("\n" + "=" * 50)
    print("JSON Response Example")
    print("=" * 50)

    config = ModelConfig(
        name=ANTHROPIC_MODEL,
        provider=Provider.ANTHROPIC,
        api_key=ANTHROPIC_API_KEY,
        temperature=0.7,
    )

    provider = create_provider(config)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content="List three programming languages and their primary use cases. "
            "Format as JSON with 'name' and 'use_case' fields."
        ),
    ]

    print(f"Requesting JSON response from {ANTHROPIC_MODEL}...")
    response = await provider.chat(messages, response_format="json")

    print(f"\nJSON Response:\n{response.content}")

    # Show usage
    usage = provider.get_usage()
    if usage:
        print(f"\nUsage: {usage.total_tokens} tokens")


async def pydantic_model_example():
    """Example: Use Pydantic model for structured output."""
    if not ANTHROPIC_API_KEY:
        print("\nError: ANTHROPIC_API_KEY environment variable not set")
        return

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

    config = ModelConfig(
        name=ANTHROPIC_MODEL,
        provider=Provider.ANTHROPIC,
        api_key=ANTHROPIC_API_KEY,
        temperature=0.7,
    )

    provider = create_provider(config)

    messages = [
        UserMessage(
            content="Create a fictional person who is a software engineer in San Francisco."
        ),
    ]

    print(f"Requesting structured output from {ANTHROPIC_MODEL}...")
    print("Using Pydantic model: Person(name, age, occupation, city)")
    response = await provider.chat(messages, response_format=Person)

    print(f"\nStructured Response:\n{response.content}")

    # Show usage
    usage = provider.get_usage()
    if usage:
        print(f"\nUsage: {usage.total_tokens} tokens")


async def conversation_example():
    """Example: Multi-turn conversation with Claude."""
    if not ANTHROPIC_API_KEY:
        print("\nError: ANTHROPIC_API_KEY environment variable not set")
        return

    print("\n" + "=" * 50)
    print("Multi-turn Conversation Example")
    print("=" * 50)

    from casual_llm import AssistantMessage

    config = ModelConfig(
        name=ANTHROPIC_MODEL,
        provider=Provider.ANTHROPIC,
        api_key=ANTHROPIC_API_KEY,
        temperature=0.7,
    )

    provider = create_provider(config)

    # First message
    messages = [
        SystemMessage(content="You are a helpful coding assistant."),
        UserMessage(content="What is a Python list comprehension?"),
    ]

    print("User: What is a Python list comprehension?")
    response1 = await provider.chat(messages, response_format="text")
    print(f"\nClaude: {response1.content}")

    # Second message - continue the conversation
    messages.append(AssistantMessage(content=response1.content))
    messages.append(UserMessage(content="Can you show me a simple example?"))

    print("\nUser: Can you show me a simple example?")
    response2 = await provider.chat(messages, response_format="text")
    print(f"\nClaude: {response2.content}")

    # Show total usage
    usage = provider.get_usage()
    if usage:
        print(f"\nTotal usage: {usage.total_tokens} tokens")


async def main():
    """Run all Anthropic examples."""
    print("casual-llm Anthropic (Claude) Examples")
    print("This demonstrates basic usage of the AnthropicProvider.\n")

    # Run basic text example
    await basic_text_response()

    # Run JSON example
    await json_response_example()

    # Run Pydantic model example
    await pydantic_model_example()

    # Run conversation example
    await conversation_example()

    print("\n" + "=" * 50)
    print("Anthropic examples complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
