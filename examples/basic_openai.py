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


async def main():
    # Create OpenAI provider configuration
    config = ModelConfig(
        name="gpt-4o-mini",
        provider=Provider.OPENAI,
        api_key=os.getenv("OPENAI_API_KEY"),  # Or hardcode (not recommended)
        temperature=0.7,
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
    print(f"Response:\n{response}")

    # Example: JSON response
    print("\n--- JSON Example ---")
    json_messages = [
        UserMessage(content="List 3 programming languages as JSON with their year of creation.")
    ]

    json_response = await provider.chat(json_messages, response_format="json")
    print(f"JSON Response: {json_response}")


if __name__ == "__main__":
    asyncio.run(main())
