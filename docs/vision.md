# Vision Guide

Send images to vision-capable models like GPT-4o, Claude 3.5 Sonnet, and llava.

## Sending Image URLs

The simplest way to include an image:

```python
from casual_llm import (
    OpenAIClient,
    Model,
    UserMessage,
    TextContent,
    ImageContent,
)

# Works with OpenAI, Anthropic, and Ollama
client = OpenAIClient(api_key="sk-...")
model = Model(client, name="gpt-4o")

# Send an image URL
messages = [
    UserMessage(
        content=[
            TextContent(text="What's in this image?"),
            ImageContent(source="https://example.com/image.jpg"),
        ]
    )
]

response = await model.chat(messages)
print(response.content)  # "I see a cat sitting on a windowsill..."
```

## Sending Base64 Images

For local images or when you need to encode the image yourself:

```python
import base64
from casual_llm import (
    OpenAIClient,
    Model,
    UserMessage,
    TextContent,
    ImageContent,
)

client = OpenAIClient(api_key="sk-...")
model = Model(client, name="gpt-4o")

# Read and encode local image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("ascii")

messages = [
    UserMessage(
        content=[
            TextContent(text="Describe this image"),
            ImageContent(
                source={"type": "base64", "data": image_data},
                media_type="image/jpeg",
            ),
        ]
    )
]

response = await model.chat(messages)
```

## Multiple Images

Send multiple images in a single message:

```python
messages = [
    UserMessage(
        content=[
            TextContent(text="Compare these two images"),
            ImageContent(source="https://example.com/image1.jpg"),
            ImageContent(source="https://example.com/image2.jpg"),
        ]
    )
]
```

## Provider Support

| Provider | URL Images | Base64 Images |
|----------|-----------|---------------|
| OpenAI (GPT-4o, GPT-4V) | Yes | Yes |
| Anthropic (Claude 3.x) | Yes | Yes |
| Ollama (llava, bakllava) | Yes* | Yes |

*Ollama fetches URL images and converts to base64 automatically.

## Supported Image Formats

- JPEG (`image/jpeg`)
- PNG (`image/png`)
- GIF (`image/gif`)
- WebP (`image/webp`)

## Next Steps

- [Quick Start Guide](quick-start.md) - Provider setup
- [Streaming Guide](streaming.md) - Real-time responses
- [Examples](../examples/vision_example.py) - Complete vision example
