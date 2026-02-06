# casual-llm

![PyPI](https://img.shields.io/pypi/v/casual-llm)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

**Lightweight LLM provider abstraction with standardized message models.**

Part of the [casual-*](https://github.com/AlexStansfield/casual-mcp) ecosystem of lightweight AI tools.

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

# Development dependencies
uv add casual-llm[dev]

# Or using pip
pip install casual-llm
pip install casual-llm[openai,anthropic]
```

## Quick Start

### Using Ollama

```python
from casual_llm import OllamaClient, Model, UserMessage

# Create client (manages API connection)
client = OllamaClient(host="http://localhost:11434")

# Create model (wraps client with model-specific settings)
model = Model(client, name="qwen2.5:7b-instruct", temperature=0.7)

# Generate response
messages = [UserMessage(content="What is the capital of France?")]
response = await model.chat(messages, response_format="text")
print(response.content)  # "The capital of France is Paris."

# Check token usage
usage = model.get_usage()
print(f"Tokens used: {usage.total_tokens}")
```

### Using OpenAI

```python
from casual_llm import OpenAIClient, Model, UserMessage

# Create client (manages API connection)
client = OpenAIClient(api_key="sk-...")  # or set OPENAI_API_KEY env var

# Create model
model = Model(client, name="gpt-4o-mini", temperature=0.7)

# Generate JSON response
messages = [UserMessage(content="List 3 colors as JSON")]
response = await model.chat(messages, response_format="json")
print(response.content)  # '{"colors": ["red", "blue", "green"]}'

# Check token usage
usage = model.get_usage()
if usage:
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
```

### Using Anthropic (Claude)

```python
from casual_llm import AnthropicClient, Model, UserMessage

# Create client (manages API connection)
client = AnthropicClient(api_key="sk-ant-...")  # or set ANTHROPIC_API_KEY env var

# Create model
model = Model(client, name="claude-3-5-sonnet-20241022", temperature=0.7)

# Generate response
messages = [UserMessage(content="Explain quantum computing in one sentence.")]
response = await model.chat(messages, response_format="text")
print(response.content)

# Check token usage
usage = model.get_usage()
if usage:
    print(f"Total tokens: {usage.total_tokens}")
```

### Using OpenAI-Compatible APIs (OpenRouter, LM Studio, etc.)

```python
from casual_llm import OpenAIClient, Model

# Create client with custom base URL
client = OpenAIClient(
    api_key="sk-or-...",
    base_url="https://openrouter.ai/api/v1"
)

# Create model for any model available through the service
model = Model(client, name="qwen/qwen-2.5-72b-instruct", temperature=0.7)
```

### Multi-Model Usage (New in v0.5.0)

Share a single API connection across multiple models:

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

### Vision Support

Send images to vision-capable models (GPT-4o, Claude 3.5 Sonnet, llava):

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

# Or send a base64-encoded image
import base64

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

### Streaming Responses

Stream responses in real-time for better UX:

```python
from casual_llm import OpenAIClient, Model, UserMessage

client = OpenAIClient(api_key="sk-...")
model = Model(client, name="gpt-4o")

messages = [UserMessage(content="Write a short poem about coding.")]

# Stream the response
async for chunk in model.stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)

print()  # New line after streaming

# Check usage after streaming
usage = model.get_usage()
if usage:
    print(f"\nTokens used: {usage.total_tokens}")
```

## Examples

Looking for more examples? Check out the [`examples/`](examples) directory for comprehensive demonstrations of all features:

- **[`basic_ollama.py`](examples/basic_ollama.py)** - Get started with Ollama (local LLMs)
- **[`basic_openai.py`](examples/basic_openai.py)** - Use OpenAI API and compatible services
- **[`basic_anthropic.py`](examples/basic_anthropic.py)** - Work with Claude models
- **[`vision_example.py`](examples/vision_example.py)** - Send images to vision-capable models
- **[`stream_example.py`](examples/stream_example.py)** - Stream responses in real-time
- **[`tool_calling.py`](examples/tool_calling.py)** - Complete tool/function calling workflow
- **[`message_formatting.py`](examples/message_formatting.py)** - All message types and structures

See the **[Examples README](examples/README.md)** for detailed descriptions, requirements, and usage instructions for each example.

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
    TextContent,  # For multimodal messages
    ImageContent,  # For vision support
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

## Custom Clients

Implement the `LLMClient` protocol to add your own provider:

```python
from typing import Literal, AsyncIterator
from pydantic import BaseModel
from casual_llm import (
    LLMClient,
    Model,
    ChatMessage,
    AssistantMessage,
    StreamChunk,
    Tool,
    Usage,
)

class MyCustomClient:
    """Custom LLM client implementation."""

    async def _chat(
        self,
        model: str,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> tuple[AssistantMessage, Usage | None]:
        # Your implementation here
        # 1. Convert messages using message_converters
        # 2. Convert tools using tool_converters (if tools provided)
        # 3. Handle vision content (ImageContent) if present
        # 4. Call your LLM API
        # 5. Parse response including tool_calls if present
        # 6. Return (AssistantMessage, Usage)
        ...

    def _stream(
        self,
        model: str,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        # Your streaming implementation here
        ...
        yield StreamChunk(content="chunk", finish_reason=None)

# Use it with Model class
client = MyCustomClient(...)
model = Model(client, name="my-model", temperature=0.7)
response = await model.chat(messages)
print(response.content)
```

## Advanced Usage

### Using Configuration Classes

For more structured configuration, use `ClientConfig` and `ModelConfig`:

```python
from casual_llm import (
    create_client,
    create_model,
    ClientConfig,
    ModelConfig,
    Provider,
    UserMessage,
)

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

# Create client and model using factory functions
client = create_client(client_config)
model = create_model(client, model_config)

# Use as normal
response = await model.chat([UserMessage(content="Hello!")])
```

### Per-Model Usage Tracking

Each `Model` instance tracks its own token usage:

```python
from casual_llm import OpenAIClient, Model, UserMessage

client = OpenAIClient(api_key="sk-...")

# Create multiple models
gpt4 = Model(client, name="gpt-4")
gpt35 = Model(client, name="gpt-3.5-turbo")

# Make calls with each model
await gpt4.chat([UserMessage(content="Hello!")])
await gpt35.chat([UserMessage(content="Hello!")])
await gpt4.chat([UserMessage(content="How are you?")])

# Each model tracks its own usage (most recent call)
gpt4_usage = gpt4.get_usage()
gpt35_usage = gpt35.get_usage()

print(f"GPT-4 last call: {gpt4_usage.total_tokens} tokens")
print(f"GPT-3.5 last call: {gpt35_usage.total_tokens} tokens")
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
| **OpenAI compatible** | Yes | Yes | Yes |

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

- **[casual-mcp](https://github.com/AlexStansfield/casual-mcp)** - MCP server orchestration and tool calling
- **casual-llm** (this library) - LLM provider abstraction
- **casual-memory** (coming soon) - Memory intelligence with conflict detection

All casual-* libraries share the same philosophy: lightweight, protocol-based, easy to use.

## API Reference

### Core Classes

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

#### `LLMClient` (Protocol)

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

#### Client Classes

- `OpenAIClient(api_key, base_url, organization, timeout)` - OpenAI and compatible APIs
- `OllamaClient(host, timeout)` - Ollama local LLMs
- `AnthropicClient(api_key, base_url, timeout)` - Anthropic Claude models

#### Configuration Classes

```python
@dataclass
class ClientConfig:
    provider: Provider  # Provider.OPENAI, Provider.OLLAMA, or Provider.ANTHROPIC
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelConfig:
    name: str  # Model name (e.g., "gpt-4", "llama3.1")
    temperature: float | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
```

#### Factory Functions

```python
def create_client(config: ClientConfig) -> LLMClient: ...
def create_model(client: LLMClient, config: ModelConfig) -> Model: ...
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

- `UserMessage(content: str | list[TextContent | ImageContent] | None)` - Supports simple text or multimodal content
- `AssistantMessage(content: str | None, tool_calls: list[AssistantToolCall] | None = None)`
- `SystemMessage(content: str)`
- `ToolResultMessage(name: str, tool_call_id: str, content: str)`
- `ChatMessage` - Type alias for any message type
- `TextContent(text: str)` - Text block for multimodal messages
- `ImageContent(source: str | dict, media_type: str | None = None)` - Image block for vision support

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- **GitHub**: https://github.com/casualgenius/casual-llm
- **PyPI**: https://pypi.org/project/casual-llm/
- **Issues**: https://github.com/casualgenius/casual-llm/issues
- **Migration Guide**: [MIGRATION-0.5.0.md](MIGRATION-0.5.0.md)
- **casual-mcp**: https://github.com/AlexStansfield/casual-mcp
