# Migration Guide: v0.4.x to v0.6.0

This guide covers migrating from the v0.4.x Provider-based API to the current Client/Model/ChatOptions architecture introduced in v0.5.0 and refined in v0.6.0.

## Overview

Version 0.5.0 introduced a breaking architectural change that separates **API connection management** (Client) from **model configuration** (Model). Version 0.6.0 further refined this with `ChatOptions` for unified request configuration.

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

### The Solution in v0.5.0+

The current architecture separates concerns into three distinct pieces:

1. **Client** - Manages the API connection (configured once per provider)
2. **Model** - Wraps a client with model-specific settings (can create many)
3. **ChatOptions** - Controls per-request behavior (temperature, tools, format, etc.)

```python
# v0.6.0 - Efficient: One connection, multiple models, flexible options
client = OpenAIClient(api_key=key)
gpt4 = Model(client, name="gpt-4", default_options=ChatOptions(temperature=0.7))
gpt35 = Model(client, name="gpt-3.5-turbo", default_options=ChatOptions(temperature=0.5))
```

### Benefits

- **Resource efficiency**: Single API connection shared across models
- **Clearer architecture**: Connection config vs model config vs request options are separate
- **Per-model usage tracking**: Each `Model` tracks its own token usage
- **Reusable option presets**: Define `ChatOptions` once, reuse across calls
- **Easier multi-model workflows**: Switch between models without reconnecting

---

## API Changes Summary

| v0.4.x | v0.6.0 |
|--------|--------|
| `OpenAIProvider` | `OpenAIClient` + `Model` |
| `OllamaProvider` | `OllamaClient` + `Model` |
| `AnthropicProvider` | `AnthropicClient` + `Model` |
| `ModelConfig` (combined) | `ClientConfig` + `ModelConfig` (split) |
| `create_provider()` | `create_client()` + `create_model()` |
| `LLMProvider` protocol | `LLMClient` protocol |
| Individual params (temperature, max_tokens, ...) | `ChatOptions` dataclass |

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

**v0.6.0:**
```python
from casual_llm import OpenAIClient, Model, ChatOptions, UserMessage

client = OpenAIClient(api_key="sk-...")
model = Model(client, name="gpt-4", default_options=ChatOptions(temperature=0.7))

response = await model.chat([UserMessage(content="Hello")])
usage = model.get_usage()
```

### JSON Responses

**v0.4.x:**
```python
response = await provider.chat(messages, response_format="json")
```

**v0.6.0:**
```python
response = await model.chat(messages, ChatOptions(response_format="json"))
```

### Tool Calling

**v0.4.x:**
```python
response = await provider.chat(messages, tools=my_tools, temperature=0.5)
```

**v0.6.0:**
```python
response = await model.chat(messages, ChatOptions(tools=my_tools, temperature=0.5))
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

**v0.6.0:**
```python
from casual_llm import create_client, create_model, ClientConfig, ModelConfig, ChatOptions, Provider

# Client config (connection settings)
client_config = ClientConfig(
    provider=Provider.OPENAI,
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
)

# Model config (model settings)
model_config = ModelConfig(
    name="gpt-4",
    default_options=ChatOptions(temperature=0.7),
)

client = create_client(client_config)
model = create_model(client, model_config)
```

### Multiple Models (New Pattern)

```python
from casual_llm import OpenAIClient, Model, ChatOptions, UserMessage

# One client connection
client = OpenAIClient(api_key="sk-...")

# Multiple models sharing the connection
gpt4 = Model(client, name="gpt-4", default_options=ChatOptions(temperature=0.7))
gpt4_mini = Model(client, name="gpt-4o-mini", default_options=ChatOptions(temperature=0.5))
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

**v0.6.0:**
```python
from casual_llm import OllamaClient, Model, ChatOptions, UserMessage

client = OllamaClient(host="http://localhost:11434")
model = Model(client, name="llama3.1", default_options=ChatOptions(temperature=0.7))

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

**v0.6.0:**
```python
from casual_llm import AnthropicClient, Model, ChatOptions, UserMessage

client = AnthropicClient(api_key="sk-ant-...")
model = Model(client, name="claude-3-5-sonnet-latest", default_options=ChatOptions(temperature=0.7))

response = await model.chat([UserMessage(content="Hello")])
```

### Streaming

**v0.4.x:**
```python
async for chunk in provider.stream(messages):
    print(chunk.content, end="")
```

**v0.6.0:**
```python
async for chunk in model.stream(messages):
    print(chunk.content, end="")

# With options
async for chunk in model.stream(messages, ChatOptions(temperature=0.9)):
    print(chunk.content, end="")
```

---

## Class Reference

### New Classes

#### `Model`

```python
class Model:
    def __init__(
        self,
        client: LLMClient,
        name: str,
        default_options: ChatOptions | None = None,
    ): ...

    async def chat(
        self,
        messages: list[ChatMessage],
        options: ChatOptions | None = None,
    ) -> AssistantMessage: ...

    async def stream(
        self,
        messages: list[ChatMessage],
        options: ChatOptions | None = None,
    ) -> AsyncIterator[StreamChunk]: ...

    def get_usage(self) -> Usage | None: ...
```

#### `LLMClient` Protocol

```python
class LLMClient(Protocol):
    async def _chat(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> tuple[AssistantMessage, Usage | None]: ...

    async def _stream(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> AsyncIterator[StreamChunk]: ...
```

#### `ChatOptions`

```python
@dataclass
class ChatOptions:
    response_format: Literal["json", "text"] | type[BaseModel] = "text"
    max_tokens: int | None = None
    tools: list[Tool] | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    tool_choice: Literal["auto", "none", "required"] | str | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    top_k: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
```

### Renamed Classes

| Old Name | New Name |
|----------|----------|
| `OpenAIProvider` | `OpenAIClient` |
| `OllamaProvider` | `OllamaClient` |
| `AnthropicProvider` | `AnthropicClient` |
| `LLMProvider` | `LLMClient` |

### Split Configuration

**v0.4.x `ModelConfig`** contained both connection and model settings. **v0.6.0** uses three separate structures:

```python
@dataclass
class ClientConfig:
    """Configuration for API connection."""
    provider: Provider | str
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelConfig:
    """Configuration for model settings."""
    name: str
    default_options: ChatOptions | None = None

@dataclass
class ChatOptions:
    """Per-request options (temperature, tools, format, etc.)."""
    # ... see above
```

---

## Import Changes

### Removed Exports

These are no longer available:

- `OpenAIProvider` (use `OpenAIClient` + `Model`)
- `OllamaProvider` (use `OllamaClient` + `Model`)
- `AnthropicProvider` (use `AnthropicClient` + `Model`)
- `LLMProvider` (use `LLMClient`)
- `create_provider()` (use `create_client()` + `create_model()`)

### New Exports

- `OpenAIClient`, `OllamaClient`, `AnthropicClient`
- `LLMClient`
- `Model`
- `ClientConfig`, `ModelConfig`, `ChatOptions`
- `create_client()`, `create_model()`

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
    ChatOptions,
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

The following remain unchanged:

- All message types (`UserMessage`, `AssistantMessage`, etc.)
- Tool definitions (`Tool`, `ToolParameter`)
- Message converters
- Tool converters
- Image utilities
- `Usage` class
- `Provider` enum
- Vision/multimodal support
- Streaming support

---

## v0.6.0 Additional Changes

Version 0.6.0 introduced these additional changes on top of v0.5.0:

- **`ChatOptions` dataclass**: Replaces individual parameters on `chat()`/`stream()` with a single options object
- **`ClientConfig.name`**: Optional client name for automatic API key lookup from `{NAME}_API_KEY` env vars
- **String provider support**: `ClientConfig.provider` accepts strings like `"openai"` in addition to `Provider` enum
- **`ollama` moved to optional dependency**: Install with `casual-llm[ollama]` — core package no longer requires it
- **`options.extra` safety**: Extra dict keys that conflict with core request parameters are now ignored with a warning
- **SSRF protection**: Image URL fetching validates against private IP ranges and non-http(s) schemes
- **Redirect safety**: Manual redirect handling with target validation (max 5 hops)
- **`max_tokens=0` fix**: Setting `max_tokens=0` is now correctly passed to providers instead of being silently ignored
- **Factory functions moved**: `create_client()` / `create_model()` moved to `casual_llm.factory` (top-level imports unchanged)

---

## Quick Migration Checklist

1. [ ] Replace `OpenAIProvider` with `OpenAIClient` + `Model`
2. [ ] Replace `OllamaProvider` with `OllamaClient` + `Model`
3. [ ] Replace `AnthropicProvider` with `AnthropicClient` + `Model`
4. [ ] Split `ModelConfig` into `ClientConfig` + `ModelConfig`
5. [ ] Replace `create_provider()` with `create_client()` + `create_model()`
6. [ ] Move individual params (temperature, max_tokens, etc.) into `ChatOptions`
7. [ ] Update imports to use new class names
8. [ ] Call `chat()`/`stream()`/`get_usage()` on `Model` instead of provider

---

## Questions?

If you encounter issues during migration, please open an issue at:
https://github.com/casualgenius/casual-llm/issues
