"""
Multiple system messages example.

Demonstrates how different providers handle multiple system messages,
and how to use the system_message_handling option to merge them.

Requirements:
- OPENAI_API_KEY environment variable (or compatible endpoint)
- ANTHROPIC_API_KEY environment variable
- pip install casual-llm[openai,anthropic]
"""

import asyncio
import os

from casual_llm import (
    AnthropicClient,
    Model,
    OpenAIClient,
    SystemMessage,
    UserMessage,
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", None)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", None)
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# Multiple system messages — a common pattern when composing prompts
# from separate sources (base persona, memory, task-specific instructions).
MESSAGES = [
    SystemMessage(content="You are a helpful assistant who speaks concisely."),
    SystemMessage(content="Always format lists using bullet points."),
    SystemMessage(content="End every response with a fun fact."),
    UserMessage(content="What are the three states of matter?"),
]


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def print_response(response_content: str) -> None:
    print(f"\nResponse:\n{response_content}")


async def openai_passthrough():
    """OpenAI with passthrough (default) — all system messages sent as-is."""
    if not OPENAI_API_KEY:
        print("Skipping: OPENAI_API_KEY not set")
        return

    print_header(f"OpenAI passthrough ({OPENAI_MODEL})")
    print("Each system message is sent as a separate message in the API call.")

    client = OpenAIClient(api_key=OPENAI_API_KEY, base_url=OPENAI_ENDPOINT)
    model = Model(client, name=OPENAI_MODEL)

    response = await model.chat(MESSAGES)
    print_response(response.content)

    usage = model.get_usage()
    if usage:
        print(f"\nTokens: {usage.total_tokens}")


async def openai_merge():
    """OpenAI with merge — all system messages combined into one."""
    if not OPENAI_API_KEY:
        print("Skipping: OPENAI_API_KEY not set")
        return

    print_header(f"OpenAI merge ({OPENAI_MODEL})")
    print("All system messages are merged into a single message before sending.")

    client = OpenAIClient(api_key=OPENAI_API_KEY, base_url=OPENAI_ENDPOINT)
    model = Model(client, name=OPENAI_MODEL, system_message_handling="merge")

    response = await model.chat(MESSAGES)
    print_response(response.content)

    usage = model.get_usage()
    if usage:
        print(f"\nTokens: {usage.total_tokens}")


async def anthropic_content_blocks():
    """Anthropic — all system messages sent as separate content blocks."""
    if not ANTHROPIC_API_KEY:
        print("Skipping: ANTHROPIC_API_KEY not set")
        return

    print_header(f"Anthropic content blocks ({ANTHROPIC_MODEL})")
    print("Each system message becomes a separate content block in the system parameter.")
    print("Anthropic always uses this approach regardless of system_message_handling.")

    client = AnthropicClient(api_key=ANTHROPIC_API_KEY)
    model = Model(client, name=ANTHROPIC_MODEL)

    response = await model.chat(MESSAGES)
    print_response(response.content)

    usage = model.get_usage()
    if usage:
        print(f"\nTokens: {usage.total_tokens}")


async def main():
    print("casual-llm: Multiple System Messages Example")
    print("=" * 60)
    print()
    print("This example sends three system messages to each provider:")
    for msg in MESSAGES:
        if isinstance(msg, SystemMessage):
            print(f'  - "{msg.content}"')
    print()
    print(f"User: {MESSAGES[-1].content}")

    await openai_passthrough()
    await openai_merge()
    await anthropic_content_blocks()

    print(f"\n{'=' * 60}")
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
