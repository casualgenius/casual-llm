"""
Basic example using Ollama client.

Requirements:
- Ollama installed and running (https://ollama.ai)
- A model pulled (e.g., `ollama pull llama3.1`)
"""

import asyncio
import os
from casual_llm import ChatOptions, OllamaClient, Model, UserMessage

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")


async def main():
    # Create Ollama client (manages API connection)
    client = OllamaClient(host=OLLAMA_ENDPOINT)

    # Create model (configure model name and parameters)
    model = Model(client, name=OLLAMA_MODEL, default_options=ChatOptions(temperature=0.7))

    # Create a simple message
    messages = [UserMessage(content="What is the capital of France?")]

    # Generate response
    print("Generating response...")
    response = await model.chat(messages, ChatOptions(response_format="text"))
    print(f"Response: {response.content}")

    # Check usage statistics
    usage = model.get_usage()
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

    json_response = await model.chat(json_messages, ChatOptions(response_format="json"))
    print(f"JSON Response: {json_response.content}")

    # Check usage statistics for JSON response
    usage = model.get_usage()
    if usage:
        print("\nUsage:")
        print(f"  Prompt tokens: {usage.prompt_tokens}")
        print(f"  Completion tokens: {usage.completion_tokens}")
        print(f"  Total tokens: {usage.total_tokens}")

    # Example: Multiple models using the same client
    print("\n--- Multi-Model Example ---")
    # You can create multiple models using the same client connection
    # model2 = Model(client, name="codellama", default_options=ChatOptions(temperature=0.5))
    # This allows efficient reuse of the client connection
    print("You can create multiple Model instances from a single OllamaClient!")
    print("Example: Model(client, name='codellama', default_options=ChatOptions(temperature=0.5))")


if __name__ == "__main__":
    asyncio.run(main())
