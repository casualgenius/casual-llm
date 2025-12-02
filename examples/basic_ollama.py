"""
Basic example using Ollama provider.

Requirements:
- Ollama installed and running (https://ollama.ai)
- A model pulled (e.g., `ollama pull qwen2.5:7b-instruct`)
"""

import asyncio
import os
from casual_llm import create_provider, ModelConfig, Provider, UserMessage

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")


async def main():
    # Create Ollama provider configuration
    config = ModelConfig(
        name=OLLAMA_MODEL,
        provider=Provider.OLLAMA,
        base_url=OLLAMA_ENDPOINT,
        temperature=0.7,
    )

    # Create provider
    provider = create_provider(config)

    # Create a simple message
    messages = [UserMessage(content="What is the capital of France?")]

    # Generate response
    print("Generating response...")
    response = await provider.chat(messages, response_format="text")
    print(f"Response: {response.content}")

    # Check usage statistics
    usage = provider.get_usage()
    if usage:
        print("\nUsage:")
        print(f"  Prompt tokens: {usage.prompt_tokens}")
        print(f"  Completion tokens: {usage.completion_tokens}")
        print(f"  Total tokens: {usage.total_tokens}")

    # Example: JSON response
    print("\n--- JSON Example ---")
    json_messages = [
        UserMessage(content="List the 3 largest capital cities as JSON with their population.")
    ]

    json_response = await provider.chat(json_messages, response_format="json")
    print(f"JSON Response: {json_response.content}")

    # Check usage statistics for JSON response
    usage = provider.get_usage()
    if usage:
        print("\nUsage:")
        print(f"  Prompt tokens: {usage.prompt_tokens}")
        print(f"  Completion tokens: {usage.completion_tokens}")
        print(f"  Total tokens: {usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
