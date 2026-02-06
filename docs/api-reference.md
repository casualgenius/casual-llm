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

### `LLMClient` (Protocol)

The protocol that all client implementations follow.

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

## Client Classes

### `OpenAIClient`

```python
OpenAIClient(
    api_key: str | None = None,      # Uses OPENAI_API_KEY env var if not provided
    base_url: str | None = None,     # Custom base URL for compatible APIs
    organization: str | None = None,
    timeout: float = 60.0,
)
```

### `OllamaClient`

```python
OllamaClient(
    host: str = "http://localhost:11434",
    timeout: float = 60.0,
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

## Configuration Classes

### `ClientConfig`

```python
@dataclass
class ClientConfig:
    provider: Provider  # Provider.OPENAI, Provider.OLLAMA, or Provider.ANTHROPIC
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
```

### `ModelConfig`

```python
@dataclass
class ModelConfig:
    name: str  # Model name (e.g., "gpt-4", "llama3.1")
    temperature: float | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
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
    content: str | None,
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
    media_type: str | None = None  # e.g., "image/jpeg"
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
    content: str | None       # Text content of this chunk
    finish_reason: str | None # "stop" when complete, None otherwise
```

## Provider Enum

```python
class Provider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
```
