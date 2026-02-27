# API Reference

## Core Classes

### `Model`

The user-facing class for LLM interactions. Wraps a client with model-specific configuration.

```python
class Model:
    def __init__(
        self,
        client: LLMClient,
        name: str,
        default_options: ChatOptions | None = None,
        system_message_handling: Literal["passthrough", "merge"] | None = None,
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

**Option precedence:** Per-call `options` override `default_options`. Within each, non-`None` fields take precedence. The `extra` dicts are merged with per-call values winning on conflicts.

### `LLMClient` (Protocol)

The protocol that all client implementations follow. Uses `typing.Protocol` — no inheritance required.

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

## ChatOptions

Central configuration for chat/stream requests. All fields are optional — providers ignore params they don't support.

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
    frequency_penalty: float | None = None   # OpenAI, Ollama
    presence_penalty: float | None = None    # OpenAI, Ollama
    seed: int | None = None                  # OpenAI, Ollama
    top_k: int | None = None                 # Anthropic, Ollama
    extra: dict[str, Any] = field(default_factory=dict)
    system_message_handling: Literal["passthrough", "merge"] | None = None
```

**`system_message_handling`:** Controls how multiple `SystemMessage`s are handled before sending to the provider. `"passthrough"` (default) sends them as-is. `"merge"` combines all system messages into one, joined by `"\n\n"`. Anthropic always sends all system messages as separate content blocks regardless of this setting. See [Advanced Usage](advanced.md#multiple-system-messages) for details.

**Provider-specific fields:**

| Field | OpenAI | Anthropic | Ollama |
|-------|--------|-----------|--------|
| `temperature` | Yes | Yes | Yes |
| `max_tokens` | Yes | Yes (required, defaults to 4096) | Yes |
| `top_p` | Yes | Yes | Yes |
| `stop` | Yes (`stop`) | Yes (`stop_sequences`) | Yes (`stop`) |
| `tool_choice` | Yes | Yes | Yes |
| `frequency_penalty` | Yes | — | Yes |
| `presence_penalty` | Yes | — | Yes |
| `seed` | Yes | — | Yes |
| `top_k` | — | Yes | Yes |
| `response_format` | Native | Via system prompt | Native |

**Using `extra`:** Pass provider-specific kwargs not covered by first-class fields. Keys that conflict with core request parameters are ignored with a warning.

```python
# OpenAI-specific: enable logprobs
opts = ChatOptions(temperature=0.7, extra={"logprobs": True})
```

## Client Classes

### `OpenAIClient`

```python
OpenAIClient(
    api_key: str | None = None,      # Uses OPENAI_API_KEY env var if not provided
    base_url: str | None = None,     # Custom base URL for compatible APIs
    organization: str | None = None,
    timeout: float = 60.0,
    system_message_handling: str | None = None,  # "passthrough" or "merge"
)
```

### `OllamaClient`

```python
OllamaClient(
    host: str = "http://localhost:11434",
    timeout: float = 60.0,
    system_message_handling: str | None = None,  # "passthrough" or "merge"
)
```

### `AnthropicClient`

```python
AnthropicClient(
    api_key: str | None = None,  # Uses ANTHROPIC_API_KEY env var if not provided
    base_url: str | None = None,
    timeout: float = 60.0,
)
```

> **Note:** `AnthropicClient` does not accept `system_message_handling` — it always sends all system messages as separate content blocks to the Anthropic API.

## Configuration Classes

### `ClientConfig`

Used with the `create_client()` factory function.

```python
@dataclass
class ClientConfig:
    provider: Provider | str   # Provider.OPENAI, "openai", etc. (case-insensitive)
    name: str | None = None    # Optional name for auto API key lookup ({NAME}_API_KEY)
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    system_message_handling: Literal["passthrough", "merge"] | None = None
```

### `ModelConfig`

Used with the `create_model()` factory function.

```python
@dataclass
class ModelConfig:
    name: str                                    # Model name (e.g., "gpt-4o-mini")
    default_options: ChatOptions | None = None   # Default options for all requests
    system_message_handling: Literal["passthrough", "merge"] | None = None
```

## Factory Functions

```python
def create_client(config: ClientConfig) -> LLMClient: ...
def create_model(client: LLMClient, config: ModelConfig) -> Model: ...
```

## Message Models

All message models are Pydantic `BaseModel` instances with full validation.

### `UserMessage`

```python
UserMessage(
    content: str | list[TextContent | ImageContent] | None
)
```

### `AssistantMessage`

```python
AssistantMessage(
    content: str | None = None,
    tool_calls: list[AssistantToolCall] | None = None
)
```

### `SystemMessage`

```python
SystemMessage(content: str)
```

### `ToolResultMessage`

```python
ToolResultMessage(
    name: str,
    tool_call_id: str,
    content: str
)
```

### `TextContent`

```python
TextContent(text: str)
```

### `ImageContent`

```python
ImageContent(
    source: str | dict,          # URL string or {"type": "base64", "data": "..."}
    media_type: str = "image/jpeg"  # e.g., "image/jpeg", "image/png"
)
```

### `ChatMessage`

Type alias for any message type:

```python
ChatMessage = UserMessage | AssistantMessage | SystemMessage | ToolResultMessage
```

## Usage Model

```python
class Usage(BaseModel):
    prompt_tokens: int      # Tokens in the prompt
    completion_tokens: int  # Tokens in the completion
    total_tokens: int       # Total tokens (computed automatically)
```

## StreamChunk Model

```python
class StreamChunk(BaseModel):
    content: str              # Text content of this chunk
    finish_reason: str | None # "stop" when complete, None otherwise
```

## Tool Models

### `Tool`

```python
class Tool(BaseModel):
    name: str
    description: str
    parameters: dict[str, ToolParameter] = {}
    required: list[str] = []

    @property
    def input_schema(self) -> dict[str, Any]: ...

    @classmethod
    def from_input_schema(cls, name: str, description: str, input_schema: dict) -> Tool: ...
```

### `ToolParameter`

```python
class ToolParameter(BaseModel):
    type: str | None = None
    description: str | None = None
    enum: list[Any] | None = None
    items: dict[str, Any] | None = None
    properties: dict[str, ToolParameter] | None = None
    required: list[str] | None = None
    default: Any | None = None
    anyOf: list[dict[str, Any]] | None = None
    oneOf: list[dict[str, Any]] | None = None
    # Additional JSON Schema fields allowed via extra="allow"
```

## Provider Enum

```python
class Provider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
```
