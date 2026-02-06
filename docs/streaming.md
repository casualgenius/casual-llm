# Streaming Guide

Stream responses in real-time for better user experience.

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

## Checking Usage After Streaming

Usage statistics are available after the stream completes:

```python
async for chunk in model.stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)

print()

# Check usage after streaming
usage = model.get_usage()
if usage:
    print(f"\nTokens used: {usage.total_tokens}")
```

## StreamChunk Object

Each chunk contains:

```python
class StreamChunk:
    content: str | None       # Text content of this chunk
    finish_reason: str | None # "stop" when complete, None otherwise
```

## Handling Finish Reason

```python
async for chunk in model.stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)

    if chunk.finish_reason == "stop":
        print("\n[Stream complete]")
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
