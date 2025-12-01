# casual-llm

![PyPI](https://img.shields.io/pypi/v/casual-llm)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

**Lightweight LLM provider abstraction with standardized message models.**

Part of the [casual-*](https://github.com/AlexStansfield/casual-mcp) ecosystem of lightweight AI tools.

## Features

- 🎯 **Protocol-based** - Uses `typing.Protocol`, no inheritance required
- 🔌 **Provider-agnostic** - Works with OpenAI, Ollama, or your custom provider
- 📦 **Lightweight** - Minimal dependencies (pydantic, ollama)
- 🔄 **Async-first** - Built for modern async Python
- 🛡️ **Type-safe** - Full type hints with py.typed marker
- 🔁 **Retry logic** - Built-in exponential backoff for transient failures
- 📊 **OpenAI-compatible** - Standard message format used across the industry

## Installation

```bash
# Basic installation (includes Ollama support)
uv add casual-llm

# With OpenAI support
uv add casual-llm[openai]

# Development dependencies
uv add casual-llm[dev]

# Or using pip
pip install casual-llm
```

## Quick Start

### Using Ollama

```python
from casual_llm import create_provider, ModelConfig, Provider, UserMessage

# Create Ollama provider
config = ModelConfig(
    name="qwen2.5:7b-instruct",
    provider=Provider.OLLAMA,
    base_url="http://localhost:11434",
    temperature=0.7
)

provider = create_provider(config, max_retries=2)

# Generate response
messages = [UserMessage(content="What is the capital of France?")]
response = await provider.chat(messages, response_format="text")
print(response.content)  # "The capital of France is Paris."
```

### Using OpenAI

```python
from casual_llm import create_provider, ModelConfig, Provider, UserMessage

# Create OpenAI provider
config = ModelConfig(
    name="gpt-4o-mini",
    provider=Provider.OPENAI,
    api_key="sk-...",  # or set OPENAI_API_KEY env var
    temperature=0.7
)

provider = create_provider(config)

# Generate JSON response
messages = [UserMessage(content="List 3 colors as JSON")]
response = await provider.chat(messages, response_format="json")
print(response.content)  # '{"colors": ["red", "blue", "green"]}'
```

### Using OpenAI-Compatible APIs (OpenRouter, LM Studio, etc.)

```python
config = ModelConfig(
    name="qwen/qwen-2.5-72b-instruct",
    provider=Provider.OPENAI,
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-...",
    temperature=0.7
)

provider = create_provider(config)
```

## Message Models

casual-llm provides OpenAI-compatible message models that work with any provider:

```python
from casual_llm import (
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    AssistantToolCall,
    ChatMessage,  # Type alias for any message type
)

# System message (sets behavior)
system_msg = SystemMessage(content="You are a helpful assistant.")

# User message
user_msg = UserMessage(content="Hello!")

# Assistant message (with optional tool calls)
assistant_msg = AssistantMessage(
    content="I'll help you with that.",
    tool_calls=[
        AssistantToolCall(
            id="call_123",
            function=AssistantToolCallFunction(
                name="get_weather",
                arguments='{"city": "Paris"}'
            )
        )
    ]
)

# Tool result message
tool_msg = ToolResultMessage(
    name="get_weather",
    tool_call_id="call_123",
    content='{"temp": 20, "condition": "sunny"}'
)

# All messages can be used in a conversation
messages: list[ChatMessage] = [system_msg, user_msg, assistant_msg, tool_msg]
```

## Custom Providers

Implement the `LLMProvider` protocol to add your own provider:

```python
from casual_llm import LLMProvider, ChatMessage, AssistantMessage, Tool

class MyCustomProvider:
    """Custom LLM provider implementation."""

    async def chat(
        self,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
    ) -> AssistantMessage:
        # Your implementation here
        ...

# Use it like any other provider
provider = MyCustomProvider(...)
response = await provider.chat(messages)
print(response.content)
```

## Advanced Usage

### Retry Logic and Metrics

```python
from casual_llm import create_provider, ModelConfig, Provider

config = ModelConfig(
    name="qwen2.5:7b-instruct",
    provider=Provider.OLLAMA,
)

# Enable retries and metrics
provider = create_provider(
    config,
    max_retries=3,  # Retry up to 3 times on transient failures
    enable_metrics=True  # Track success/failure counts
)

# After some calls, check metrics (Ollama only)
metrics = provider.get_metrics()
print(metrics)
# {'success_count': 42, 'failure_count': 3, 'success_rate_percent': 93.33}
```

## Why casual-llm?

| Feature | casual-llm | LangChain | litellm |
|---------|-----------|-----------|---------|
| **Dependencies** | 2 (pydantic, ollama) | 100+ | 50+ |
| **Protocol-based** | ✅ | ❌ | ❌ |
| **Type-safe** | ✅ Full typing | Partial | Partial |
| **Message models** | ✅ Included | ❌ Separate | ❌ |
| **Learning curve** | ⚡ Minutes | 📚 Hours | 📖 Medium |
| **OpenAI compatible** | ✅ | ✅ | ✅ |

**Use casual-llm when you want:**
- Lightweight, focused library (not a framework)
- Protocol-based design (no inheritance)
- Standard message models shared across your codebase
- Simple, predictable API

**Use LangChain when you need:**
- Full-featured framework with chains, agents, RAG
- Massive ecosystem of integrations
- Higher-level abstractions

## Part of the casual-* Ecosystem

- **[casual-mcp](https://github.com/AlexStansfield/casual-mcp)** - MCP server orchestration and tool calling
- **casual-llm** (this library) - LLM provider abstraction
- **casual-memory** (coming soon) - Memory intelligence with conflict detection

All casual-* libraries share the same philosophy: lightweight, protocol-based, easy to use.

## API Reference

### Core Classes

#### `LLMProvider` (Protocol)
```python
class LLMProvider(Protocol):
    async def chat(
        self,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
    ) -> AssistantMessage: ...
```

#### `ModelConfig`
```python
@dataclass
class ModelConfig:
    name: str  # Model name (e.g., "gpt-4o-mini", "qwen2.5:7b-instruct")
    provider: Provider  # Provider.OPENAI or Provider.OLLAMA
    base_url: str | None = None  # Optional API base URL
    api_key: str | None = None  # Optional API key
    temperature: float = 0.1  # Temperature for generation
```

#### `create_provider()`
```python
def create_provider(
    model_config: ModelConfig,
    timeout: float = 60.0,
    max_retries: int = 0,
    enable_metrics: bool = False,
) -> LLMProvider: ...
```

### Message Models

All message models are Pydantic `BaseModel` instances with full validation:

- `UserMessage(content: str | None)`
- `AssistantMessage(content: str | None, tool_calls: list[AssistantToolCall] | None = None)`
- `SystemMessage(content: str)`
- `ToolResultMessage(name: str, tool_call_id: str, content: str)`
- `ChatMessage` - Type alias for any message type

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- **GitHub**: https://github.com/AlexStansfield/casual-llm
- **PyPI**: https://pypi.org/project/casual-llm/
- **Issues**: https://github.com/AlexStansfield/casual-llm/issues
- **casual-mcp**: https://github.com/AlexStansfield/casual-mcp
