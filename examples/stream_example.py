"""
Stream example using OpenAI and Ollama providers.

This example demonstrates how to use streaming responses from LLMs
using casual-llm's stream support.

Requirements:
- For OpenAI: OPENAI_API_KEY environment variable
- For Ollama: Ollama running locally or remote endpoint
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")


async def openai_stream_example():
    """Example: Stream responses from OpenAI."""
    if not OPENAI_API_KEY:
        print("Skipping OpenAI example (OPENAI_API_KEY not set)")
        return

    print("=" * 50)
    print("OpenAI Stream Example")
    print("=" * 50)

    config = ModelConfig(
        name=OPENAI_MODEL,
        provider=Provider.OPENAI,
        api_key=OPENAI_API_KEY,
        temperature=0.7,
    )

    provider = create_provider(config)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="Write a short poem about coding. Keep it under 50 words."),
    ]

    print(f"Streaming from {OPENAI_MODEL}...")
    print("\nResponse:")

    # Stream the response
    async for chunk in provider.stream(messages, response_format="text"):
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print("\n")

    # Show usage
    usage = provider.get_usage()
    if usage:
        print(f"Usage: {usage.total_tokens} tokens")


async def ollama_stream_example():
    """Example: Stream responses from Ollama."""
    print("\n" + "=" * 50)
    print("Ollama Stream Example")
    print("=" * 50)

    config = ModelConfig(
        name=OLLAMA_MODEL,
        provider=Provider.OLLAMA,
        base_url=OLLAMA_ENDPOINT,
        temperature=0.7,
    )

    provider = create_provider(config)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="Explain what async/await does in Python in 2-3 sentences."),
    ]

    print(f"Streaming from {OLLAMA_MODEL}...")
    print("\nResponse:")

    try:
        # Stream the response
        async for chunk in provider.stream(messages, response_format="text"):
            if chunk.content:
                print(chunk.content, end="", flush=True)

        print("\n")

        # Show usage
        usage = provider.get_usage()
        if usage:
            print(f"Usage: {usage.total_tokens} tokens")
    except Exception as e:
        print(f"\nError: {e}")
        print(f"Make sure Ollama is running and '{OLLAMA_MODEL}' is pulled:")
        print(f"  ollama pull {OLLAMA_MODEL}")


async def ollama_stream_conversation_example():
    """Example: Stream a multi-turn conversation with Ollama."""
    print("\n" + "=" * 50)
    print("Ollama Stream Conversation Example")
    print("=" * 50)

    config = ModelConfig(
        name=OLLAMA_MODEL,
        provider=Provider.OLLAMA,
        base_url=OLLAMA_ENDPOINT,
        temperature=0.7,
    )

    provider = create_provider(config)

    # First turn
    messages = [
        SystemMessage(content="You are a helpful coding assistant."),
        UserMessage(content="What is a Python decorator?"),
    ]

    print("User: What is a Python decorator?")
    print(f"\n{OLLAMA_MODEL}: ", end="", flush=True)

    try:
        response_text = ""
        async for chunk in provider.stream(messages, response_format="text"):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                response_text += chunk.content

        print("\n")

        # Second turn - follow-up question
        from casual_llm import AssistantMessage

        messages.append(AssistantMessage(content=response_text))
        messages.append(UserMessage(content="Can you show me a simple example?"))

        print("User: Can you show me a simple example?")
        print(f"\n{OLLAMA_MODEL}: ", end="", flush=True)

        async for chunk in provider.stream(messages, response_format="text"):
            if chunk.content:
                print(chunk.content, end="", flush=True)

        print("\n")

        # Show usage
        usage = provider.get_usage()
        if usage:
            print(f"Total usage: {usage.total_tokens} tokens")
    except Exception as e:
        print(f"\nError: {e}")


async def main():
    """Run all stream examples."""
    print("casual-llm Stream Examples")
    print("This demonstrates streaming responses from LLMs.\n")

    # Run OpenAI example
    await openai_stream_example()

    # Run Ollama example
    await ollama_stream_example()

    # Run conversation example
    await ollama_stream_conversation_example()

    print("\n" + "=" * 50)
    print("Stream examples complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
