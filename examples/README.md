# casual-llm Examples

This directory contains example scripts demonstrating the various features and capabilities of casual-llm.

## Quick Start Examples

These examples demonstrate basic usage with each provider.

### [`basic_ollama.py`](basic_ollama.py)

Basic usage with Ollama (local LLM server).

**What it demonstrates:**
- Creating an Ollama client and model
- Simple text responses
- JSON-formatted responses
- Pydantic model-based structured output
- Token usage tracking

**Requirements:**
- Ollama running locally (`ollama serve`)
- A model pulled (e.g., `ollama pull qwen2.5:7b-instruct`)

**Environment variables:**
- `OLLAMA_ENDPOINT` (optional, defaults to `http://localhost:11434`)
- `OLLAMA_MODEL` (optional, defaults to `qwen2.5:7b-instruct`)

**Run:**
```bash
uv run python examples/basic_ollama.py
```

---

### [`basic_openai.py`](basic_openai.py)

Basic usage with OpenAI API (GPT-4, GPT-3.5, etc.).

**What it demonstrates:**
- Creating an OpenAI client and model
- Simple text responses
- JSON-formatted responses
- Pydantic model-based structured output
- Token usage tracking
- Works with OpenAI-compatible APIs (OpenRouter, LM Studio, etc.)

**Requirements:**
- OpenAI API key

**Environment variables:**
- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (optional, defaults to `gpt-4o-mini`)

**Run:**
```bash
export OPENAI_API_KEY=sk-...
uv run python examples/basic_openai.py
```

---

### [`basic_anthropic.py`](basic_anthropic.py)

Basic usage with Anthropic API (Claude models).

**What it demonstrates:**
- Creating an Anthropic client and model
- Simple text responses
- JSON-formatted responses
- Pydantic model-based structured output
- Multi-turn conversations
- Token usage tracking

**Requirements:**
- Anthropic API key

**Environment variables:**
- `ANTHROPIC_API_KEY` (required)
- `ANTHROPIC_MODEL` (optional, defaults to `claude-haiku-4-5-20251001`)

**Run:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run python examples/basic_anthropic.py
```

---

## Advanced Examples

These examples demonstrate specific features and advanced capabilities.

### [`system_messages.py`](system_messages.py)

Multiple system message handling across providers.

**What it demonstrates:**
- Sending multiple system messages (composing prompts from separate sources)
- OpenAI passthrough mode (default) — each system message sent separately
- OpenAI merge mode — all system messages combined into one
- Anthropic content blocks — each system message as a separate content block

**Requirements:**
- OpenAI API key and/or Anthropic API key

**Environment variables:**
- `OPENAI_API_KEY` (optional)
- `OPENAI_MODEL` (optional, defaults to `gpt-4.1-nano`)
- `OPENAI_ENDPOINT` (optional)
- `ANTHROPIC_API_KEY` (optional)
- `ANTHROPIC_MODEL` (optional, defaults to `claude-haiku-4-5-20251001`)

**Run:**
```bash
# With both providers
OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... uv run python examples/system_messages.py

# OpenAI only
OPENAI_API_KEY=sk-... uv run python examples/system_messages.py

# Anthropic only
ANTHROPIC_API_KEY=sk-ant-... uv run python examples/system_messages.py
```

---

### [`message_formatting.py`](message_formatting.py)

Message model examples (no API calls required).

**What it demonstrates:**
- All message types (UserMessage, AssistantMessage, SystemMessage, ToolResultMessage)
- Tool call structures
- Message serialization and validation
- Message conversations

**Requirements:**
- None (no API calls, just model demonstrations)

**Run:**
```bash
uv run python examples/message_formatting.py
```

---

### [`tool_calling.py`](tool_calling.py)

Complete tool/function calling workflow.

**What it demonstrates:**
- Defining tools with parameters
- LLM requesting tool calls
- Processing tool calls
- Returning tool results
- Multi-turn tool calling conversation
- Works with all providers (OpenAI, Anthropic, Ollama)

**Requirements:**
- Ollama running locally (default) OR OpenAI/Anthropic API key

**Environment variables:**
- `OLLAMA_ENDPOINT` (optional)
- `OPENAI_API_KEY` (optional, to use OpenAI instead)
- `ANTHROPIC_API_KEY` (optional, to use Anthropic instead)

**Run:**
```bash
# With Ollama (default)
uv run python examples/tool_calling.py

# With OpenAI
OPENAI_API_KEY=sk-... uv run python examples/tool_calling.py

# With Anthropic
ANTHROPIC_API_KEY=sk-ant-... uv run python examples/tool_calling.py
```

---

### [`vision_example.py`](vision_example.py)

Vision/multimodal examples with image content.

**What it demonstrates:**
- Sending images from URLs (OpenAI, Anthropic, Ollama)
- Sending base64-encoded images (all providers)
- Using local image files
- Multimodal messages with text + images
- Vision support across all three providers

**Requirements:**
- At least one of: Ollama (with llava model), OpenAI API key, or Anthropic API key
- For Ollama: `ollama pull llava`

**Environment variables:**
- `OLLAMA_ENDPOINT` (optional)
- `OLLAMA_MODEL` (optional, defaults to `llava`)
- `OPENAI_API_KEY` (optional)
- `OPENAI_MODEL` (optional, defaults to `gpt-4o-mini`)
- `ANTHROPIC_API_KEY` (optional)
- `ANTHROPIC_MODEL` (optional, defaults to `claude-3-5-sonnet-20241022`)

**Run:**
```bash
# With Ollama
uv run python examples/vision_example.py

# With OpenAI
OPENAI_API_KEY=sk-... uv run python examples/vision_example.py

# With Anthropic
ANTHROPIC_API_KEY=sk-ant-... uv run python examples/vision_example.py
```

---

### [`stream_example.py`](stream_example.py)

Streaming response examples.

**What it demonstrates:**
- Real-time response streaming
- Streaming with OpenAI
- Streaming with Ollama
- Multi-turn conversations with streaming
- Displaying tokens as they arrive

**Requirements:**
- Ollama running locally (default) OR OpenAI/Anthropic API key

**Environment variables:**
- `OLLAMA_ENDPOINT` (optional)
- `OLLAMA_MODEL` (optional)
- `OPENAI_API_KEY` (optional, to use OpenAI instead)
- `OPENAI_MODEL` (optional)

**Run:**
```bash
# With Ollama (default)
uv run python examples/stream_example.py

# With OpenAI
OPENAI_API_KEY=sk-... uv run python examples/stream_example.py
```

---

## Provider Comparison

| Feature | Ollama | OpenAI | Anthropic |
|---------|--------|--------|-----------|
| **Text responses** | Yes | Yes | Yes |
| **JSON mode** | Yes | Yes | Yes (via system prompt) |
| **Pydantic models** | Yes | Yes | Yes (via JSON schema) |
| **Tool calling** | Yes | Yes | Yes |
| **Vision (URL)** | Yes* | Yes | Yes |
| **Vision (base64)** | Yes | Yes | Yes |
| **Streaming** | Yes | Yes | Yes |
| **Token usage** | Yes | Yes | Yes |
| **Cost** | Free (local) | Paid | Paid |

\* Ollama requires client-side image fetching (handled automatically by casual-llm)

---

## Common Patterns

### Setting Up a Client and Model

```python
from casual_llm import OllamaClient, OpenAIClient, AnthropicClient, Model, ChatOptions

# Ollama (local)
client = OllamaClient(host="http://localhost:11434")
model = Model(client, name="qwen2.5:7b-instruct", default_options=ChatOptions(temperature=0.7))

# OpenAI
client = OpenAIClient(api_key="sk-...")  # or set OPENAI_API_KEY env var
model = Model(client, name="gpt-4o-mini", default_options=ChatOptions(temperature=0.7))

# Anthropic
client = AnthropicClient(api_key="sk-ant-...")  # or set ANTHROPIC_API_KEY env var
model = Model(client, name="claude-sonnet-4-20250514", default_options=ChatOptions(temperature=0.7))
```

### Basic Chat

```python
from casual_llm import UserMessage

messages = [UserMessage(content="Hello!")]
response = await model.chat(messages)
print(response.content)
```

### Streaming

```python
messages = [UserMessage(content="Write a poem")]

async for chunk in model.stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### Vision

```python
from casual_llm import UserMessage, TextContent, ImageContent

messages = [
    UserMessage(
        content=[
            TextContent(text="What's in this image?"),
            ImageContent(source="https://example.com/image.jpg"),
        ]
    )
]

response = await model.chat(messages)
print(response.content)
```

### Structured Output

```python
from pydantic import BaseModel
from casual_llm import ChatOptions

class Person(BaseModel):
    name: str
    age: int
    occupation: str

messages = [UserMessage(content="Tell me about a software engineer")]
response = await model.chat(messages, ChatOptions(response_format=Person))
print(response.content)  # Valid JSON matching Person schema
```

---

## Tips

1. **Start with basic examples** - Get familiar with the client/model patterns first
2. **Use environment variables** - Keep API keys out of your code
3. **Check usage** - Use `model.get_usage()` to monitor token consumption
4. **Try streaming** - Provides better UX for long responses
5. **Use ChatOptions presets** - Define reusable option presets for common configurations
6. **Use Pydantic models** - Great for structured data extraction and validation

---

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve

# Pull a model if needed
ollama pull qwen2.5:7b-instruct
ollama pull llava  # For vision examples
```

### API Key Issues

```bash
# Set environment variables
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Verify they're set
echo $OPENAI_API_KEY
```

### Import Errors

```bash
# Install with all providers
uv add casual-llm[ollama,openai,anthropic]

# Or separately
uv add casual-llm[ollama]
uv add casual-llm[openai]
uv add casual-llm[anthropic]
```

---

## More Information

- **Main README**: [../README.md](../README.md)
- **API Reference**: [../docs/api-reference.md](../docs/api-reference.md)
- **GitHub**: https://github.com/casualgenius/casual-llm
- **Issues**: https://github.com/casualgenius/casual-llm/issues
