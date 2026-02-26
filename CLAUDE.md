# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**casual-llm** is a lightweight, protocol-based LLM provider abstraction library with standardized OpenAI-compatible message models. It's part of the casual-* ecosystem of lightweight AI tools.

### Architecture: Client / Model Pattern

The library uses a **two-layer** architecture:

1. **Client** (connection layer) - Manages API connections. One client per provider, shared across models.
   - `OllamaClient`, `OpenAIClient`, `AnthropicClient`
2. **Model** (interaction layer) - User-facing class wrapping a client with model-specific config.
   - `Model(client, name="gpt-4", default_options=ChatOptions(temperature=0.7))`

```python
from casual_llm import OpenAIClient, Model, ChatOptions, UserMessage

# Create client once (manages API connection)
client = OpenAIClient(api_key="...")

# Create multiple models sharing the same client
gpt4 = Model(client, name="gpt-4", default_options=ChatOptions(temperature=0.7))
gpt4o = Model(client, name="gpt-4o")

# Use models
response = await gpt4.chat([UserMessage(content="Hello")])
print(response.content)
```

### Key Design Principles

1. **Lightweight** - Minimal core dependencies (pydantic, httpx)
2. **Protocol-based** - Uses `typing.Protocol`, no inheritance required
3. **Type-safe** - Full type hints with py.typed marker (Python 3.10+)
4. **OpenAI-compatible** - Standard message format used across the industry
5. **Async-first** - Built for modern async Python
6. **Tool calling** - First-class support for function/tool calling
7. **Vision support** - Multimodal messages with images (URL and base64)
8. **Streaming** - Stream responses in real-time with AsyncIterator

### Repository Structure

```
casual-llm/
├── src/casual_llm/
│   ├── __init__.py              # Public API exports
│   ├── messages.py              # OpenAI-compatible message models
│   ├── tools.py                 # Tool and ToolParameter models
│   ├── config.py                # ChatOptions, ClientConfig, ModelConfig, Provider enum
│   ├── usage.py                 # Usage statistics model
│   ├── model.py                 # Model class (user-facing interaction layer)
│   ├── factory.py               # create_client() and create_model() factories
│   ├── message_converters/      # Message format converters per provider
│   │   ├── openai.py
│   │   ├── ollama.py
│   │   └── anthropic.py
│   ├── tool_converters/         # Tool format converters per provider
│   │   ├── openai.py
│   │   ├── ollama.py
│   │   └── anthropic.py
│   ├── utils/
│   │   └── image.py             # Image fetching and encoding utilities
│   ├── py.typed                 # PEP 561 type marker
│   └── providers/
│       ├── __init__.py          # Client exports
│       ├── base.py              # LLMClient protocol
│       ├── ollama.py            # OllamaClient (optional: casual-llm[ollama])
│       ├── openai.py            # OpenAIClient (optional: casual-llm[openai])
│       └── anthropic.py         # AnthropicClient (optional: casual-llm[anthropic])
├── tests/
├── examples/
└── docs/
```

---

## Development Workflow

This project uses **[uv](https://github.com/astral-sh/uv)** for dependency management (not pip/poetry).

### Python Version

- **Minimum**: Python 3.10
- **Recommended**: Python 3.11 or 3.12
- Uses modern type hints (`list[X]`, `dict[K,V]`, `X | Y`)
- Uses `from __future__ import annotations` where beneficial

### Setup

```bash
# Install core dependencies only
uv sync

# Install with all provider extras + dev dependencies
uv sync --all-extras
```

### Running Tests

```bash
uv run pytest tests/
uv run pytest tests/ --cov=casual_llm --cov-report=term-missing
uv run pytest tests/test_messages.py::test_user_message -v
```

### Code Quality

```bash
uv run ruff format src/ tests/ examples/
uv run ruff check src/ tests/ examples/
uv run mypy src/casual_llm
uv run pytest tests/

# All at once
uv run ruff format src/ tests/ examples/ && \
uv run ruff check src/ tests/ examples/ && \
uv run mypy src/casual_llm && \
uv run pytest tests/
```

---

## Core Components

### 1. Configuration (`src/casual_llm/config.py`)

**`ChatOptions`** - Dataclass for all chat/stream request options:

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
    frequency_penalty: float | None = None  # OpenAI, Ollama
    presence_penalty: float | None = None   # OpenAI, Ollama
    seed: int | None = None                 # OpenAI, Ollama
    top_k: int | None = None                # Anthropic, Ollama
    extra: dict[str, Any] = field(default_factory=dict)  # Provider-specific pass-through
    system_message_handling: Literal["passthrough", "merge"] | None = None
```

**`ClientConfig`** - Dataclass for client connection configuration:
- `provider`: `Provider` enum or string ("openai", "ollama", "anthropic")
- `name`: Optional name for automatic API key lookup (`{NAME}_API_KEY` env var)
- `base_url`, `api_key`, `timeout`, `extra_kwargs`
- `system_message_handling`: Optional `"passthrough"` or `"merge"`

**`ModelConfig`** - Dataclass for model configuration:
- `name`: Model identifier (e.g., "gpt-4", "llama3.1")
- `default_options`: Optional `ChatOptions` defaults
- `system_message_handling`: Optional `"passthrough"` or `"merge"`

**`Provider`** - Enum: `OPENAI`, `OLLAMA`, `ANTHROPIC`

### 2. Model Class (`src/casual_llm/model.py`)

The **user-facing** class for LLM interactions:

```python
class Model:
    def __init__(self, client: LLMClient, name: str, default_options: ChatOptions | None = None, system_message_handling: str | None = None): ...

    async def chat(self, messages: list[ChatMessage], options: ChatOptions | None = None) -> AssistantMessage: ...
    async def stream(self, messages: list[ChatMessage], options: ChatOptions | None = None) -> AsyncIterator[StreamChunk]: ...
    def get_usage(self) -> Usage | None: ...
```

- Wraps an `LLMClient` with model-specific configuration
- Merges `default_options` with per-call `options` (per-call non-None values win)
- Resolves `system_message_handling` chain: per-call ChatOptions > Model > Client > default
- Tracks per-model usage statistics via `get_usage()`

### 3. LLMClient Protocol (`src/casual_llm/providers/base.py`)

Internal protocol that provider clients implement:

```python
class LLMClient(Protocol):
    async def _chat(self, model: str, messages: list[ChatMessage], options: ChatOptions) -> tuple[AssistantMessage, Usage | None]: ...
    def _stream(self, model: str, messages: list[ChatMessage], options: ChatOptions) -> AsyncIterator[StreamChunk]: ...
```

**Key**: Uses `typing.Protocol` (PEP 544) — any class implementing these methods is compatible without inheritance.

### 4. Message Models (`src/casual_llm/messages.py`)

OpenAI-compatible Pydantic models:

- `UserMessage` - User message (text or multimodal: `str | list[TextContent | ImageContent]`)
- `AssistantMessage` - AI response (with optional `tool_calls`)
- `SystemMessage` - System prompt
- `ToolResultMessage` - Tool execution result
- `AssistantToolCall` / `AssistantToolCallFunction` - Tool call structures
- `TextContent` / `ImageContent` - Multimodal content blocks
- `StreamChunk` - Streaming response chunk (`content`, `finish_reason`)
- `ChatMessage` - TypeAlias for all message types

### 5. Tool Models (`src/casual_llm/tools.py`)

- `Tool` - Tool definition (name, description, parameters, required)
- `ToolParameter` - JSON Schema-based parameter definition (`extra="allow"` for additional schema fields)
- `Tool.input_schema` - Property returning full JSON Schema dict
- `Tool.from_input_schema()` - Create Tool from MCP-style inputSchema

### 6. Provider Implementations

All providers implement the `LLMClient` protocol. All are **optional dependencies**.

**OllamaClient** (`providers/ollama.py`) — `pip install casual-llm[ollama]`:
- Uses official `ollama.AsyncClient`
- Supports JSON/text/Pydantic model response formats
- Supports tool calling, vision (client-side base64 encoding), streaming
- Accepts `system_message_handling` for merge behavior

**OpenAIClient** (`providers/openai.py`) — `pip install casual-llm[openai]`:
- Uses `openai.AsyncOpenAI`
- Works with OpenAI API and compatible services (OpenRouter, etc.)
- Supports tool calling, vision (native URL/base64), streaming
- Accepts `system_message_handling` for merge behavior

**AnthropicClient** (`providers/anthropic.py`) — `pip install casual-llm[anthropic]`:
- Uses `anthropic.AsyncAnthropic`
- Extracts all system messages to separate `system` parameter as content blocks
- Supports tool calling, vision (native URL/base64), streaming
- Always uses content blocks for system messages (ignores `system_message_handling`)

### 7. Factory Functions (`src/casual_llm/factory.py`)

```python
def create_client(config: ClientConfig) -> LLMClient: ...
def create_model(client: LLMClient, config: ModelConfig) -> Model: ...
```

- `create_client()` resolves API keys from config or `{name}_API_KEY` env vars
- Handles optional dependencies with clear ImportError messages

### 8. Converters

**Message Converters** (`message_converters/`):
- `convert_messages_to_openai()` / `convert_messages_to_ollama()` / `convert_messages_to_anthropic()`
- `convert_tool_calls_from_openai()` / `convert_tool_calls_from_ollama()` / `convert_tool_calls_from_anthropic()`
- `extract_system_messages()` — Returns all system messages as Anthropic content block dicts
- `merge_system_messages()` — Merges multiple SystemMessages into one (used by OpenAI/Ollama merge mode)

**Tool Converters** (`tool_converters/`):
- `tool_to_openai()` / `tools_to_openai()` — OpenAI tool format
- `tool_to_ollama()` / `tools_to_ollama()` — Ollama tool format (same as OpenAI)
- `tool_to_anthropic()` / `tools_to_anthropic()` — Anthropic tool format

### 9. Image Utilities (`src/casual_llm/utils/image.py`)

- `fetch_image_as_base64(url)` — Async image fetch with SSRF protection, size limits, redirect validation
- `strip_base64_prefix()` / `add_base64_prefix()` — Data URI helpers
- Uses HTTP/2 for reliable fetching

---

## API Design Guidelines

### Public API (exported from `__init__.py`)

Core types: `ChatOptions`, `ClientConfig`, `ModelConfig`, `Provider`, `Model`, `LLMClient`
Clients: `OllamaClient`, `OpenAIClient`, `AnthropicClient`
Factories: `create_client`, `create_model`
Messages: `UserMessage`, `AssistantMessage`, `SystemMessage`, `ToolResultMessage`, `AssistantToolCall`, `AssistantToolCallFunction`, `StreamChunk`, `TextContent`, `ImageContent`, `ChatMessage`
Tools: `Tool`, `ToolParameter`
Usage: `Usage`
Converters: All converter functions (exported for advanced use cases)

### options.extra Safety

The `extra` dict in `ChatOptions` passes provider-specific kwargs to the API call. **Core parameters cannot be overwritten** — if an `extra` key conflicts with a core parameter (e.g., "model", "messages"), it is silently ignored with a warning log.

### Type Hints

- **Always** include type hints on all public APIs
- Modern Python 3.10+ syntax: `list[X]`, `dict[K, V]`, `X | None`
- `from __future__ import annotations` for forward references
- `typing.Protocol` for interfaces, `TypeAlias` for type aliases

### Async Patterns

- All LLM operations are async
- OllamaClient uses `ollama.AsyncClient`
- OpenAIClient uses `openai.AsyncOpenAI`
- AnthropicClient uses `anthropic.AsyncAnthropic`
- Test with `pytest-asyncio` (`asyncio_mode = "auto"` in pyproject.toml)

### Logging

```python
logger = logging.getLogger(__name__)
logger.debug("Processing %d messages", len(messages))  # Lazy formatting
```

---

## Common Tasks

### Adding a New Provider

1. Create `src/casual_llm/providers/your_provider.py`:

```python
from casual_llm.config import ChatOptions
from casual_llm.messages import ChatMessage, AssistantMessage, StreamChunk
from casual_llm.usage import Usage

class YourClient:
    def __init__(self, api_key: str | None = None, timeout: float = 60.0): ...

    async def _chat(
        self, model: str, messages: list[ChatMessage], options: ChatOptions,
    ) -> tuple[AssistantMessage, Usage | None]:
        # 1. Convert messages using message_converters
        # 2. Convert tools using tool_converters (if options.tools)
        # 3. Build request kwargs, apply options
        # 4. Call your LLM API
        # 5. Parse response, return (AssistantMessage, Usage)
        ...

    async def _stream(
        self, model: str, messages: list[ChatMessage], options: ChatOptions,
    ) -> AsyncIterator[StreamChunk]:
        # Yield StreamChunk objects as tokens arrive
        ...
```

2. Add try/except import in `providers/__init__.py`
3. Add to `create_client()` in `factory.py`
4. Add optional dependency in `pyproject.toml`
5. Write tests and example

### Adding a New Message Type

1. Add Pydantic model to `messages.py`
2. Update `ChatMessage` TypeAlias
3. Update all message converters
4. Add tests

---

## Dependencies

### Core (Required)

- `pydantic>=2.0.0` — Data validation and models
- `httpx[http2]>=0.28.1` — HTTP client for image fetching

### Optional (Provider SDKs)

- `ollama>=0.6.1` — `pip install casual-llm[ollama]`
- `openai>=1.0.0` — `pip install casual-llm[openai]`
- `anthropic>=0.20.0` — `pip install casual-llm[anthropic]`

### Dev

- `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`

---

## Testing Guidelines

- One test file per module
- Mock external APIs (never make real API calls in tests)
- Test both chat and stream paths
- Test tool calling, vision, and response formats
- Test `options.extra` doesn't overwrite core parameters
- Test usage tracking via `model.get_usage()`
- Descriptive test names: `test_user_message_with_multimodal_content()`

---

## Integration with casual-* Ecosystem

```
casual-mcp (orchestration, MCP server integration)
    |
casual-llm (clients, models, messages, tools)
```

Message and tool models are the single source of truth in casual-llm.

---

## Important Notes

### Don't Break Backwards Compatibility

Post-v1.0: Don't remove public APIs without deprecation, don't change signatures or return types, document breaking changes in CHANGELOG.md.

### Keep It Lightweight

Avoid unnecessary dependencies. All provider SDKs are optional. Focus on core abstraction.

### Protocol-Based Design

Use `typing.Protocol` for interfaces. Don't require inheritance. Support custom providers.

### Vision Support

- **OpenAI/Anthropic**: Native URL and base64 support — no client-side fetching
- **Ollama**: Requires client-side fetching via `fetch_image_as_base64()`
- Image utilities include SSRF protection (blocks private IPs, validates redirects)

---

## Quick Reference

```bash
uv run pytest tests/                                    # Run tests
uv run pytest tests/ --cov=casual_llm                   # With coverage
uv run ruff format src/ tests/ examples/                      # Format
uv run ruff check src/ tests/ examples/                 # Lint
uv run mypy src/casual_llm                              # Type check
```
