"""
Basic example using OpenAI provider.

Requirements:
- OpenAI API key set in OPENAI_API_KEY environment variable
- OR pass api_key to ModelConfig
- pip install casual-llm[openai]
"""

import asyncio
import os
from casual_llm import create_provider, ModelConfig, Provider, UserMessage, SystemMessage


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)


async def main():
    # Create OpenAI provider configuration
    config = ModelConfig(
        name=OPENAI_MODEL,
        provider=Provider.OPENAI,
        api_key=OPENAI_API_KEY,
        temperature=0.7,
        base_url=OPENAI_ENDPOINT,
    )

    # Create provider
    provider = create_provider(config)

    # Create conversation with system message
    messages = [
        SystemMessage(content="You are a helpful assistant that speaks in haiku."),
        UserMessage(content="Tell me about Python programming."),
    ]

    # Generate response
    print("Generating response...")
    response = await provider.chat(messages, response_format="text")
    print(f"Response:\n{response.content}")

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
        UserMessage(content="List 3 programming languages as JSON with their year of creation.")
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
