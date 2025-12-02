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
- 📊 **OpenAI-compatible** - Standard message format used across the industry
- 🔧 **Tool calling** - First-class support for function/tool calling
- 📈 **Usage tracking** - Track token usage for cost monitoring

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

provider = create_provider(config)

# Generate response
messages = [UserMessage(content="What is the capital of France?")]
response = await provider.chat(messages, response_format="text")
print(response.content)  # "The capital of France is Paris."

# Check token usage
usage = provider.get_usage()
print(f"Tokens used: {usage.total_tokens}")
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

# Check token usage
usage = provider.get_usage()
if usage:
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
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
from casual_llm import LLMProvider, ChatMessage, AssistantMessage, Tool, Usage

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

    def get_usage(self) -> Usage | None:
        """Return token usage from last call."""
        return self._last_usage

# Use it like any other provider
provider = MyCustomProvider(...)
response = await provider.chat(messages)
print(response.content)
```

## Advanced Usage

### Usage Tracking

Track token usage from API calls for cost monitoring:

```python
from casual_llm import create_provider, ModelConfig, Provider, UserMessage

config = ModelConfig(
    name="gpt-4o-mini",
    provider=Provider.OPENAI,
    api_key="sk-...",
)

provider = create_provider(config)

# Make a chat call
messages = [UserMessage(content="Hello!")]
response = await provider.chat(messages)

# Get usage statistics from the last call
usage = provider.get_usage()
if usage:
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
```

Both OpenAI and Ollama providers support usage tracking.

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

    def get_usage(self) -> Usage | None: ...
```

#### `ModelConfig`
```python
@dataclass
class ModelConfig:
    name: str  # Model name (e.g., "gpt-4o-mini", "qwen2.5:7b-instruct")
    provider: Provider  # Provider.OPENAI or Provider.OLLAMA
    base_url: str | None = None  # Optional API base URL
    api_key: str | None = None  # Optional API key
    temperature: float | None = None  # Temperature for generation (uses provider default if None)
```

#### `create_provider()`
```python
def create_provider(
    model_config: ModelConfig,
    timeout: float = 60.0,
) -> LLMProvider: ...
```

#### `Usage`
```python
class Usage(BaseModel):
    prompt_tokens: int  # Tokens in the prompt
    completion_tokens: int  # Tokens in the completion
    total_tokens: int  # Total tokens (computed automatically)
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

- **GitHub**: https://github.com/casualgenius/casual-llm
- **PyPI**: https://pypi.org/project/casual-llm/
- **Issues**: https://github.com/casualgenius/casual-llm/issues
- **casual-mcp**: https://github.com/AlexStansfield/casual-mcp
