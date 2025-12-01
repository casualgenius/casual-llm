"""
Basic example using Ollama provider.

Requirements:
- Ollama installed and running (https://ollama.ai)
- A model pulled (e.g., `ollama pull qwen2.5:7b-instruct`)
"""

import asyncio
from casual_llm import create_provider, ModelConfig, Provider, UserMessage


async def main():
    # Create Ollama provider configuration
    config = ModelConfig(
        name="qwen2.5:7b-instruct",  # Change to your preferred model
        provider=Provider.OLLAMA,
        base_url="http://localhost:11434",
        temperature=0.7,
    )

    # Create provider with retry logic
    provider = create_provider(config, max_retries=2, enable_metrics=True)

    # Create a simple message
    messages = [UserMessage(content="What is the capital of France?")]

    # Generate response
    print("Generating response...")
    response = await provider.chat(messages, response_format="text")
    print(f"Response: {response.content}")

    # Check metrics (if enabled)
    if hasattr(provider, "get_metrics"):
        metrics = provider.get_metrics()
        print(f"\nMetrics: {metrics}")


if __name__ == "__main__":
    asyncio.run(main())
