"""
Basic example using OpenAI client.

Requirements:
- OpenAI API key set in OPENAI_API_KEY environment variable
- OR pass api_key to OpenAIClient
- pip install casual-llm[openai]
"""

import asyncio
import os
from casual_llm import ChatOptions, OpenAIClient, Model, UserMessage, SystemMessage


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)


async def main():
    # Create OpenAI client (manages API connection)
    client = OpenAIClient(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_ENDPOINT,
    )

    # Create model (configure model name and parameters)
    model = Model(client, name=OPENAI_MODEL, default_options=ChatOptions(temperature=0.7))

    # Create conversation with system message
    messages = [
        SystemMessage(content="You are a helpful assistant that speaks in haiku."),
        UserMessage(content="Tell me about Python programming."),
    ]

    # Generate response
    print("Generating response...")
    response = await model.chat(messages, ChatOptions(response_format="text"))
    print(f"Response:\n{response.content}")

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
        UserMessage(content="List 3 programming languages as JSON with their year of creation.")
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
    # gpt4 = Model(client, name="gpt-4", default_options=ChatOptions(temperature=0.7))
    # gpt35 = Model(client, name="gpt-3.5-turbo", default_options=ChatOptions(temperature=0.5))
    print("You can create multiple Model instances from a single OpenAIClient!")
    print("Example: Model(client, name='gpt-4', default_options=ChatOptions(temperature=0.7))")


if __name__ == "__main__":
    asyncio.run(main())
