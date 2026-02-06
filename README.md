# casual-llm

![PyPI](https://img.shields.io/pypi/v/casual-llm)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

**Lightweight LLM provider abstraction with standardized message models.**

Part of the "casual" ecosystem of lightweight AI tools.

> **Upgrading from v0.4.x?** See the [Migration Guide](MIGRATION-0.5.0.md) for breaking changes.

## Features

- **Client/Model Separation** - Configure API connections once, create multiple models
- **Protocol-based** - Uses `typing.Protocol`, no inheritance required
- **Multi-provider** - Works with OpenAI, Anthropic (Claude), Ollama, or your custom provider
- **Lightweight** - Minimal dependencies (pydantic, ollama, httpx)
- **Async-first** - Built for modern async Python
- **Type-safe** - Full type hints with py.typed marker
- **OpenAI-compatible** - Standard message format used across the industry
- **Tool calling** - First-class support for function/tool calling
- **Per-model usage tracking** - Track token usage per model for cost monitoring
- **Vision support** - Send images to vision-capable models
- **Streaming** - Stream responses in real-time with `AsyncIterator`

## Installation

```bash
# Basic installation (includes Ollama support)
uv add casual-llm

# With OpenAI support
uv add casual-llm[openai]

# With Anthropic (Claude) support
uv add casual-llm[anthropic]

# With all providers
uv add casual-llm[openai,anthropic]

# Or using pip
pip install casual-llm[openai,anthropic]
```

## Quick Start

```python
from casual_llm import OpenAIClient, Model, UserMessage

# Create client (works with OpenAI, OpenRouter, LM Studio, etc.)
client = OpenAIClient(
    api_key="sk-...",  # or set OPENAI_API_KEY env var
    base_url="https://openrouter.ai/api/v1",  # optional, omit for OpenAI
)

# Create model
model = Model(client, "gpt-4o-mini")

# Generate response
response = await model.chat([UserMessage(content="Hello!")])
print(response.content)
```

**More examples:**
- [Quick Start Guide](docs/quick-start.md) - Ollama, Anthropic, and more
- [Vision Guide](docs/vision.md) - Send images to models
- [Streaming Guide](docs/streaming.md) - Real-time responses
- [Advanced Usage](docs/advanced.md) - Custom clients, configuration classes
- [API Reference](docs/api-reference.md) - Full API documentation
- [Examples Directory](examples/) - Complete working examples

## Message Models

casual-llm provides OpenAI-compatible message models that work with any provider:

```python
from casual_llm import (
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    TextContent,
    ImageContent,
)

# System message (sets behavior)
system_msg = SystemMessage(content="You are a helpful assistant.")

# User message (simple text)
user_msg = UserMessage(content="Hello!")

# User message (multimodal - text + image)
vision_msg = UserMessage(
    content=[
        TextContent(text="What's in this image?"),
        ImageContent(source="https://example.com/image.jpg"),
    ]
)

# Assistant message (response from LLM)
assistant_msg = AssistantMessage(content="I'll help you with that.")

# Tool result message (after executing a tool)
tool_msg = ToolResultMessage(
    name="get_weather",
    tool_call_id="call_123",
    content='{"temp": 20, "condition": "sunny"}'
)
```

## Why casual-llm?

| Feature | casual-llm | LangChain | litellm |
|---------|-----------|-----------|---------|
| **Dependencies** | 3 (pydantic, ollama, httpx) | 100+ | 50+ |
| **Protocol-based** | Yes | No | No |
| **Type-safe** | Full typing | Partial | Partial |
| **Message models** | Included | Separate | None |
| **Multi-model sharing** | Yes | No | Yes |
| **Vision support** | All providers | Yes | Yes |
| **Streaming** | All providers | Yes | Yes |
| **Providers** | OpenAI, Anthropic, Ollama | Many | Many |
| **Learning curve** | Minutes | Hours | Medium |

**Use casual-llm when you want:**
- Lightweight, focused library (not a framework)
- Protocol-based design (no inheritance)
- Standard message models shared across your codebase
- Efficient multi-model usage with shared connections
- Simple, predictable API

**Use LangChain when you need:**
- Full-featured framework with chains, agents, RAG
- Massive ecosystem of integrations
- Higher-level abstractions

## Part of the casual-* Ecosystem

- **[casual-mcp](https://github.com/casualgenius/casual-mcp)** - MCP server orchestration and tool calling
- **casual-llm** (this library) - LLM provider abstraction
- **[casual-memory](https://github.com/casualgenius/casual-memory)** - Memory intelligence with conflict detection

All casual-* libraries share the same philosophy: lightweight, protocol-based, easy to use.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- **GitHub**: https://github.com/casualgenius/casual-llm
- **PyPI**: https://pypi.org/project/casual-llm/
- **Issues**: https://github.com/casualgenius/casual-llm/issues
- **Migration Guide**: [MIGRATION-0.5.0.md](MIGRATION-0.5.0.md)
- **casual-mcp**: https://github.com/casualgenius/casual-mcp
