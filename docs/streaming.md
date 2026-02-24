# Streaming Guide

Stream responses in real-time for better user experience.

> **Note:** All examples use `await` and `async for` which require an async context.
> Wrap in `async def main()` and run with `asyncio.run(main())`.

## Basic Streaming

```python
from casual_llm import OpenAIClient, Model, UserMessage

client = OpenAIClient(api_key="sk-...")
model = Model(client, name="gpt-4o")

messages = [UserMessage(content="Write a short poem about coding.")]

# Stream the response
async for chunk in model.stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)

print()  # New line after streaming
```

## Streaming with Options

Pass `ChatOptions` to control generation parameters:

```python
from casual_llm import ChatOptions

async for chunk in model.stream(messages, ChatOptions(temperature=0.9, max_tokens=500)):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

## StreamChunk Object

Each chunk contains:

```python
class StreamChunk:
    content: str              # Text content of this chunk
    finish_reason: str | None # Reason for completion, or None while streaming
```

### Finish Reasons by Provider

The `finish_reason` value depends on the provider:

| Provider | Possible Values |
|----------|----------------|
| OpenAI | `"stop"`, `"length"`, `"content_filter"`, `"tool_calls"`, `None` |
| Anthropic | `"end_turn"`, `"stop_sequence"`, `None` |
| Ollama | `"stop"`, `None` |

## Handling Finish Reason

```python
async for chunk in model.stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)

    if chunk.finish_reason:
        if chunk.finish_reason == "stop":
            print("\n[Complete]")
        elif chunk.finish_reason == "length":
            print("\n[Truncated - max tokens reached]")
        elif chunk.finish_reason == "content_filter":
            print("\n[Content filtered]")
```

## Complete Runnable Example

```python
import asyncio
from casual_llm import OpenAIClient, Model, UserMessage

async def main():
    client = OpenAIClient(api_key="sk-...")
    model = Model(client, name="gpt-4o")

    messages = [UserMessage(content="Write a haiku about Python.")]

    async for chunk in model.stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print()

if __name__ == "__main__":
    asyncio.run(main())
```

## Provider Support

All providers support streaming:

| Provider | Streaming Support |
|----------|------------------|
| OpenAI | Yes |
| Anthropic | Yes |
| Ollama | Yes |

## Next Steps

- [Quick Start Guide](quick-start.md) - Provider setup
- [Vision Guide](vision.md) - Send images to models
- [Examples](../examples/stream_example.py) - Complete streaming example
