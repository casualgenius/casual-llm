"""
Vision example using OpenAI and Ollama providers with image content.

This example demonstrates how to send images to vision-capable models
using casual-llm's multimodal message support.

Requirements:
- For OpenAI: OPENAI_API_KEY environment variable
- For Ollama: Ollama running with a vision model (e.g., `ollama pull llava`)
- pip install casual-llm[openai]
"""

import asyncio
import os
from casual_llm import (
    create_provider,
    ModelConfig,
    Provider,
    UserMessage,
    TextContent,
    ImageContent,
)


# Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-nano")
OLLAMA_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")

# Sample image URL for testing (a simple test image)
SAMPLE_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
)


async def openai_vision_example():
    """Example: Send an image URL to OpenAI's vision model."""
    if not OPENAI_API_KEY:
        print("Skipping OpenAI example (OPENAI_API_KEY not set)")
        return

    print("=" * 50)
    print("OpenAI Vision Example")
    print("=" * 50)

    config = ModelConfig(
        name=OPENAI_MODEL,
        provider=Provider.OPENAI,
        api_key=OPENAI_API_KEY,
        temperature=0.7,
    )

    provider = create_provider(config)

    # Create a multimodal message with text and image
    messages = [
        UserMessage(
            content=[
                TextContent(text="What do you see in this image? Describe it briefly."),
                ImageContent(source=SAMPLE_IMAGE_URL),
            ]
        )
    ]

    print(f"Sending image to {OPENAI_MODEL}...")
    print(f"Image URL: {SAMPLE_IMAGE_URL}")

    response = await provider.chat(messages, response_format="text")
    print(f"\nResponse:\n{response.content}")

    # Show usage
    usage = provider.get_usage()
    if usage:
        print(f"\nUsage: {usage.total_tokens} tokens")


async def ollama_vision_example():
    """Example: Send an image URL to Ollama's vision model (e.g., llava)."""
    print("\n" + "=" * 50)
    print("Ollama Vision Example")
    print("=" * 50)

    config = ModelConfig(
        name=OLLAMA_MODEL,
        provider=Provider.OLLAMA,
        base_url=OLLAMA_ENDPOINT,
        temperature=0.7,
    )

    provider = create_provider(config)

    # Create a multimodal message with text and image
    messages = [
        UserMessage(
            content=[
                TextContent(text="What is in this image?"),
                ImageContent(source=SAMPLE_IMAGE_URL),
            ]
        )
    ]

    print(f"Sending image to {OLLAMA_MODEL}...")
    print(f"Image URL: {SAMPLE_IMAGE_URL}")

    try:
        response = await provider.chat(messages, response_format="text")
        print(f"\nResponse:\n{response.content}")

        # Show usage
        usage = provider.get_usage()
        if usage:
            print(f"\nUsage: {usage.total_tokens} tokens")
    except Exception as e:
        print(f"\nError: {e}")
        print(f"Make sure Ollama is running and '{OLLAMA_MODEL}' is pulled:")
        print(f"  ollama pull {OLLAMA_MODEL}")


async def openai_base64_image_example():
    """Example: Send a base64-encoded image with OpenAI."""
    if not OPENAI_API_KEY:
        print("\nSkipping OpenAI base64 example (OPENAI_API_KEY not set)")
        return

    print("\n" + "=" * 50)
    print("OpenAI Base64 Image Example")
    print("=" * 50)

    config = ModelConfig(
        name=OPENAI_MODEL,
        provider=Provider.OPENAI,
        api_key=OPENAI_API_KEY,
        temperature=0.7,
    )

    provider = create_provider(config)

    # Load and encode the happy-dog.jpg image
    import base64
    from pathlib import Path

    image_path = Path(__file__).parent / "data" / "happy-dog.jpg"
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("ascii")

    # Create message with base64 image
    messages = [
        UserMessage(
            content=[
                TextContent(text="What do you see in this image? Describe the dog."),
                ImageContent(
                    source={"type": "base64", "data": image_data},
                    media_type="image/jpeg",
                ),
            ]
        )
    ]

    print("Sending base64-encoded image (happy-dog.jpg)...")
    response = await provider.chat(messages, response_format="text")
    print(f"\nResponse:\n{response.content}")

    # Show usage
    usage = provider.get_usage()
    if usage:
        print(f"\nUsage: {usage.total_tokens} tokens")


async def ollama_base64_image_example():
    """Example: Send a base64-encoded image with Ollama."""
    print("\n" + "=" * 50)
    print("Ollama Base64 Image Example")
    print("=" * 50)

    config = ModelConfig(
        name=OLLAMA_MODEL,
        provider=Provider.OLLAMA,
        base_url=OLLAMA_ENDPOINT,
        temperature=0.7,
    )

    provider = create_provider(config)

    # Load and encode the happy-dog.jpg image
    import base64
    from pathlib import Path

    image_path = Path(__file__).parent / "data" / "happy-dog.jpg"
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("ascii")

    # Create message with base64 image
    messages = [
        UserMessage(
            content=[
                TextContent(text="What do you see in this image? Describe the dog."),
                ImageContent(
                    source={"type": "base64", "data": image_data},
                    media_type="image/jpeg",
                ),
            ]
        )
    ]

    print("Sending base64-encoded image (happy-dog.jpg)...")

    try:
        response = await provider.chat(messages, response_format="text")
        print(f"\nResponse:\n{response.content}")

        # Show usage
        usage = provider.get_usage()
        if usage:
            print(f"\nUsage: {usage.total_tokens} tokens")
    except Exception as e:
        print(f"\nError: {e}")
        print(f"Make sure Ollama is running and '{OLLAMA_MODEL}' is pulled:")
        print(f"  ollama pull {OLLAMA_MODEL}")


async def main():
    """Run all vision examples."""
    print("casual-llm Vision Examples")
    print("This demonstrates sending images to vision-capable models.\n")

    # Run OpenAI example
    await openai_vision_example()

    # Run Ollama example
    await ollama_vision_example()

    # Run OpenAI base64 example
    await openai_base64_image_example()

    # Run Ollama base64 example
    await ollama_base64_image_example()

    print("\n" + "=" * 50)
    print("Vision examples complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
