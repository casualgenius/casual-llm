# Migration Guide: v0.4.x to v0.5.0

## Overview

Version 0.5.0 introduces a breaking architectural change that separates **API connection management** (Client) from **model configuration** (Model). This change enables more efficient multi-model usage and clearer separation of concerns.

## Why This Change?

### The Problem with v0.4.x

In v0.4.x, each model required its own provider instance, even when using the same API connection:

```python
# v0.4.x - Wasteful: Two separate API connections for the same provider
gpt4 = OpenAIProvider(model="gpt-4", api_key=key, temperature=0.7)
gpt35 = OpenAIProvider(model="gpt-3.5-turbo", api_key=key, temperature=0.5)
```

This approach was inefficient when:
- Using multiple models from the same provider (e.g., GPT-4 and GPT-3.5)
- Using services like OpenRouter that provide access to many models
- Switching between models dynamically

### The Solution in v0.5.0

v0.5.0 separates the concerns into two distinct classes:

1. **Client** - Manages the API connection (configured once per provider)
2. **Model** - Wraps a client with model-specific settings (can create many)

```python
# v0.5.0 - Efficient: One connection, multiple models
client = OpenAIClient(api_key=key)
gpt4 = Model(client, name="gpt-4", temperature=0.7)
gpt35 = Model(client, name="gpt-3.5-turbo", temperature=0.5)
```

### Benefits

- **Resource efficiency**: Single API connection shared across models
- **Clearer architecture**: Connection config vs model config are separate
- **Per-model usage tracking**: Each `Model` tracks its own token usage
- **Easier multi-model workflows**: Switch between models without reconnecting

---

## API Changes Summary

| v0.4.x | v0.5.0 |
|--------|--------|
| `OpenAIProvider` | `OpenAIClient` + `Model` |
| `OllamaProvider` | `OllamaClient` + `Model` |
| `AnthropicProvider` | `AnthropicClient` + `Model` |
| `ModelConfig` (combined) | `ClientConfig` + `ModelConfig` (split) |
| `create_provider()` | `create_client()` + `create_model()` |
| `LLMProvider` protocol | `LLMClient` protocol |

---

## Migration Examples

### Basic Usage

**v0.4.x:**
```python
from casual_llm import OpenAIProvider, UserMessage

provider = OpenAIProvider(
    model="gpt-4",
    api_key="sk-...",
    temperature=0.7,
)

response = await provider.chat([UserMessage(content="Hello")])
usage = provider.get_usage()
```

**v0.5.0:**
```python
from casual_llm import OpenAIClient, Model, UserMessage

client = OpenAIClient(api_key="sk-...")
model = Model(client, name="gpt-4", temperature=0.7)

response = await model.chat([UserMessage(content="Hello")])
usage = model.get_usage()
```

### Using ModelConfig

**v0.4.x:**
```python
from casual_llm import create_provider, ModelConfig, Provider

config = ModelConfig(
    name="gpt-4",
    provider=Provider.OPENAI,
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
    temperature=0.7,
)

provider = create_provider(config)
```

**v0.5.0:**
```python
from casual_llm import create_client, create_model, ClientConfig, ModelConfig, Provider

# Client config (connection settings)
client_config = ClientConfig(
    provider=Provider.OPENAI,
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
)

# Model config (model settings)
model_config = ModelConfig(
    name="gpt-4",
    temperature=0.7,
)

client = create_client(client_config)
model = create_model(client, model_config)
```

### Multiple Models (New Pattern)

**v0.5.0 enables efficient multi-model usage:**
```python
from casual_llm import OpenAIClient, Model, UserMessage

# One client connection
client = OpenAIClient(api_key="sk-...")

# Multiple models sharing the connection
gpt4 = Model(client, name="gpt-4", temperature=0.7)
gpt4_mini = Model(client, name="gpt-4o-mini", temperature=0.5)
gpt35 = Model(client, name="gpt-3.5-turbo")

# Use whichever model is appropriate
response = await gpt4.chat([UserMessage(content="Complex question...")])
response = await gpt4_mini.chat([UserMessage(content="Simple question...")])

# Each model tracks its own usage
print(gpt4.get_usage())      # Usage for gpt-4 calls
print(gpt4_mini.get_usage()) # Usage for gpt-4o-mini calls
```

### Ollama

**v0.4.x:**
```python
from casual_llm import OllamaProvider, UserMessage

provider = OllamaProvider(
    model="llama3.1",
    host="http://localhost:11434",
    temperature=0.7,
)

response = await provider.chat([UserMessage(content="Hello")])
```

**v0.5.0:**
```python
from casual_llm import OllamaClient, Model, UserMessage

client = OllamaClient(host="http://localhost:11434")
model = Model(client, name="llama3.1", temperature=0.7)

response = await model.chat([UserMessage(content="Hello")])
```

### Anthropic

**v0.4.x:**
```python
from casual_llm import AnthropicProvider, UserMessage

provider = AnthropicProvider(
    model="claude-3-5-sonnet-latest",
    api_key="sk-ant-...",
    temperature=0.7,
)

response = await provider.chat([UserMessage(content="Hello")])
```

**v0.5.0:**
```python
from casual_llm import AnthropicClient, Model, UserMessage

client = AnthropicClient(api_key="sk-ant-...")
model = Model(client, name="claude-3-5-sonnet-latest", temperature=0.7)

response = await model.chat([UserMessage(content="Hello")])
```

### Streaming

**v0.4.x:**
```python
async for chunk in provider.stream(messages):
    print(chunk.content, end="")
```

**v0.5.0:**
```python
async for chunk in model.stream(messages):
    print(chunk.content, end="")
```

---

## Class Reference

### New Classes

#### `Model`

The user-facing class for LLM interactions. Wraps a client with model-specific configuration.

```python
class Model:
    def __init__(
        self,
        client: LLMClient,
        name: str,
        temperature: float | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ): ...

    async def chat(
        self,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,  # Overrides instance temperature
    ) -> AssistantMessage: ...

    async def stream(
        self,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]: ...

    def get_usage(self) -> Usage | None: ...
```

#### `LLMClient` Protocol

The protocol that all client implementations follow. Clients expose internal `_chat` and `_stream` methods that accept the model name.

```python
class LLMClient(Protocol):
    async def _chat(
        self,
        model: str,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> tuple[AssistantMessage, Usage | None]: ...

    def _stream(
        self,
        model: str,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]: ...
```

### Renamed Classes

| Old Name | New Name |
|----------|----------|
| `OpenAIProvider` | `OpenAIClient` |
| `OllamaProvider` | `OllamaClient` |
| `AnthropicProvider` | `AnthropicClient` |
| `LLMProvider` | `LLMClient` |

### Split Configuration

**v0.4.x `ModelConfig`** contained both connection and model settings:
```python
@dataclass
class ModelConfig:
    name: str
    provider: Provider
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0
    temperature: float | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
```

**v0.5.0** splits this into two dataclasses:

```python
@dataclass
class ClientConfig:
    """Configuration for API connection."""
    provider: Provider
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelConfig:
    """Configuration for model settings."""
    name: str
    temperature: float | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
```

---

## Import Changes

### Removed Exports

These are no longer available in v0.5.0:

- `OpenAIProvider` (use `OpenAIClient` + `Model`)
- `OllamaProvider` (use `OllamaClient` + `Model`)
- `AnthropicProvider` (use `AnthropicClient` + `Model`)
- `LLMProvider` (use `LLMClient`)
- `create_provider()` (use `create_client()` + `create_model()`)

### New Exports

- `OpenAIClient`
- `OllamaClient`
- `AnthropicClient`
- `LLMClient`
- `Model`
- `ClientConfig`
- `create_client()`
- `create_model()`

### Full Import Example

```python
from casual_llm import (
    # Clients
    OpenAIClient,
    OllamaClient,
    AnthropicClient,
    LLMClient,  # Protocol for type hints

    # Model wrapper
    Model,

    # Configuration
    ClientConfig,
    ModelConfig,
    Provider,

    # Factory functions
    create_client,
    create_model,

    # Messages (unchanged)
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    ChatMessage,
    TextContent,
    ImageContent,
    StreamChunk,

    # Tools (unchanged)
    Tool,
    ToolParameter,

    # Usage (unchanged)
    Usage,
)
```

---

## Unchanged APIs

The following remain unchanged in v0.5.0:

- All message types (`UserMessage`, `AssistantMessage`, etc.)
- Tool definitions (`Tool`, `ToolParameter`)
- Message converters
- Tool converters
- Image utilities
- `Usage` class
- `Provider` enum
- Vision/multimodal support
- Streaming support
- JSON/Pydantic response formats

---

## Quick Migration Checklist

1. [ ] Replace `OpenAIProvider` with `OpenAIClient` + `Model`
2. [ ] Replace `OllamaProvider` with `OllamaClient` + `Model`
3. [ ] Replace `AnthropicProvider` with `AnthropicClient` + `Model`
4. [ ] Split `ModelConfig` into `ClientConfig` + `ModelConfig`
5. [ ] Replace `create_provider()` with `create_client()` + `create_model()`
6. [ ] Update imports to use new class names
7. [ ] Call `chat()`/`stream()`/`get_usage()` on `Model` instead of provider

---

## Questions?

If you encounter issues during migration, please open an issue at:
https://github.com/casualgenius/casual-llm/issues
