# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**casual-llm** is a lightweight, protocol-based LLM provider abstraction library with standardized OpenAI-compatible message models. It's part of the casual-* ecosystem of lightweight AI tools.

### Recent Updates (v0.4.1)

- **Anthropic Provider**: Full support for Claude models with vision and streaming
- **Vision/Multimodal Support**: Send images (URL or base64) to vision-capable models (GPT-4o, Claude 3.5, llava)
- **Streaming Support**: Real-time response streaming with `AsyncIterator[StreamChunk]` across all providers
- **Image Utilities**: HTTP/2-enabled image fetching with proper headers for reliable image downloads
- **Improved Test Coverage**: 95% coverage with comprehensive test suite

### Key Design Principles

1. **Lightweight** - Minimal dependencies (pydantic, ollama, httpx)
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
├── src/casual_llm/              # Main package
│   ├── __init__.py              # Public API exports (main entry point)
│   ├── messages.py              # OpenAI-compatible message models
│   ├── tools.py                 # Tool and ToolParameter models
│   ├── config.py                # ModelConfig and Provider enum
│   ├── usage.py                 # Usage statistics model
│   ├── message_converters/      # Message format converters
│   │   ├── openai.py            # OpenAI format converters
│   │   ├── ollama.py            # Ollama format converters
│   │   └── anthropic.py         # Anthropic format converters
│   ├── tool_converters/         # Tool format converters
│   │   ├── openai.py            # OpenAI tool format
│   │   ├── ollama.py            # Ollama tool format
│   │   └── anthropic.py         # Anthropic tool format
│   ├── utils/                   # Utility functions
│   │   └── image.py             # Image fetching and encoding utilities
│   ├── py.typed                 # PEP 561 type marker
│   └── providers/               # Provider implementations
│       ├── __init__.py          # Provider exports + create_provider() factory
│       ├── base.py              # LLMProvider protocol
│       ├── ollama.py            # OllamaProvider
│       ├── openai.py            # OpenAIProvider (optional dependency)
│       └── anthropic.py         # AnthropicProvider (optional dependency)
├── tests/                       # Test suite
│   ├── test_messages.py         # Message model tests
│   ├── test_tools.py            # Tool model tests
│   ├── test_providers.py        # Provider tests
│   └── test_image_utils.py      # Image utility tests
├── examples/                    # Working examples
│   ├── basic_ollama.py          # Ollama usage
│   ├── basic_openai.py          # OpenAI usage
│   ├── basic_anthropic.py       # Anthropic usage
│   ├── message_formatting.py    # All message types demo
│   ├── tool_calling.py          # Complete tool calling workflow
│   ├── vision_example.py        # Vision/multimodal examples
│   └── streaming_example.py     # Streaming response examples
├── docs/                        # Documentation
│   ├── decisions/               # Architecture Decision Records
│   │   └── 001-shelve-media-support.md
│   └── ...                      # API reference, guides
└── ...
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
# Install dependencies
uv sync

# Install with all extras (includes dev dependencies, openai, and anthropic)
uv sync --all-extras
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=casual_llm --cov-report=term-missing

# Run specific test
uv run pytest tests/test_messages.py::test_user_message -v
```

### Running Examples

```bash
# Message formatting example (no external dependencies)
uv run python examples/message_formatting.py

# Ollama example (requires Ollama running locally)
uv run python examples/basic_ollama.py

# Tool calling example (requires Ollama running locally)
uv run python examples/tool_calling.py

# OpenAI example (requires OPENAI_API_KEY env var)
uv run python examples/basic_openai.py

# Anthropic example (requires ANTHROPIC_API_KEY env var)
uv run python examples/basic_anthropic.py

# Vision example (requires provider and API key)
uv run python examples/vision_example.py

# Streaming example (requires provider and API key)
uv run python examples/streaming_example.py
```

### Code Quality

```bash
# Format code with black
uv run black src/ tests/ examples/

# Lint with ruff
uv run ruff check src/ tests/ examples/

# Type check with mypy
uv run mypy src/casual_llm

# Run all checks before committing
uv run black src/ tests/ examples/ && \
uv run ruff check src/ tests/ examples/ && \
uv run mypy src/casual_llm && \
uv run pytest tests/
```

---

## Core Components

### 1. Message Models (`src/casual_llm/messages.py`)

OpenAI-compatible Pydantic models for LLM conversations:

- `UserMessage` - Message from the user (supports text or multimodal content)
- `AssistantMessage` - Message from the AI (with optional tool_calls)
- `SystemMessage` - System prompt that sets behavior
- `ToolResultMessage` - Result from tool/function execution
- `AssistantToolCall` - Tool call structure
- `AssistantToolCallFunction` - Function details within a tool call
- `ChatMessage` - Type alias for any message type
- `TextContent` - Text block for multimodal messages
- `ImageContent` - Image block for vision support (URL or base64)
- `StreamChunk` - Chunk of streaming response

**Important**: These were moved from `casual-mcp` to create a single source of truth for message models across the casual-* ecosystem.

**Vision Support**: UserMessage content can be either a simple string or a list of TextContent/ImageContent objects for multimodal messages.

### 2. Tool Models (`src/casual_llm/tools.py`)

Models for defining LLM function calling tools:

- `Tool` - Tool definition with name, description, parameters, and required fields
- `ToolParameter` - JSON Schema-based parameter definition

**Design**: Tool names are flexible strings (not restricted to Python identifiers) to support various naming conventions (snake_case, kebab-case, dotted notation).

### 3. Provider Protocol (`src/casual_llm/providers/base.py`)

- `LLMProvider` - Protocol defining the interface for all providers
- The protocol defines `chat()` and `stream()` methods that return `AssistantMessage` and `AsyncIterator[StreamChunk]` respectively

**Key Design**: Uses `typing.Protocol` so custom providers don't need to inherit from a base class.

### 4. Configuration (`src/casual_llm/config.py`)

- `ModelConfig` - Dataclass for model configuration
- `Provider` - Enum (OPENAI, OLLAMA, ANTHROPIC)

### 5. Provider Implementations

**OllamaProvider** (`src/casual_llm/providers/ollama.py`):
- Uses the official `ollama` Python library with `AsyncClient`
- Supports JSON and text response formats
- Supports tool calling
- Supports vision (fetches images and converts to base64)
- Supports streaming
- Generates unique tool call IDs if Ollama doesn't provide them

**OpenAIProvider** (`src/casual_llm/providers/openai.py`):
- Works with OpenAI API and compatible services (OpenRouter, etc.)
- Optional dependency (`uv add casual-llm[openai]`)
- Async-first using `AsyncOpenAI` client
- Supports tool calling
- Supports vision (URLs and base64 images)
- Supports streaming

**AnthropicProvider** (`src/casual_llm/providers/anthropic.py`):
- Works with Anthropic API for Claude models
- Optional dependency (`uv add casual-llm[anthropic]`)
- Async-first using `AsyncAnthropic` client
- Supports tool calling
- Supports vision (URLs and base64 images natively)
- Supports streaming
- Uses separate system parameter (not in messages array)

### 6. Converters

**Message Converters**:
- `src/casual_llm/message_converters/openai.py` - OpenAI format conversion
- `src/casual_llm/message_converters/ollama.py` - Ollama format conversion
- `src/casual_llm/message_converters/anthropic.py` - Anthropic format conversion (with vision support)

Each converter handles:
- Message format conversion (including multimodal content)
- Tool call parsing from provider responses
- Provider-specific quirks (e.g., Anthropic's separate system parameter)

**Tool Converters**:
- `src/casual_llm/tool_converters/openai.py` - OpenAI tool format
- `src/casual_llm/tool_converters/ollama.py` - Ollama tool format
- `src/casual_llm/tool_converters/anthropic.py` - Anthropic tool format

**Note**: OpenAI and Ollama use the same tool format, so the converters share implementation.

### 7. Utilities

**Image Utilities** (`src/casual_llm/utils/image.py`):
- `fetch_image_as_base64()` - Fetch image from URL and convert to base64
- `strip_base64_prefix()` - Remove data URI prefix from base64 strings
- `add_base64_prefix()` - Add data URI prefix to base64 strings
- Uses HTTP/2 for reliable fetching from sites like Wikipedia
- Includes User-Agent header for bot detection avoidance
- Configurable size limits and timeout

### 8. Factory Function (`create_provider()`)

Located in `src/casual_llm/providers/__init__.py`:
- Creates provider instances from `ModelConfig`
- Handles optional dependencies gracefully
- Provides clear error messages

---

## API Design Guidelines

### Public vs Private

**Public API** (exported from `src/casual_llm/__init__.py`):
- All message models (UserMessage, AssistantMessage, etc.)
- All tool models (Tool, ToolParameter)
- Provider protocol and implementations
- Configuration (ModelConfig, Provider)
- `create_provider()` factory
- Message and tool converters

**Internal API** (not exported):
- Provider implementation details
- Internal helper functions

### Type Hints

- **Always** include type hints on all public APIs
- Use `typing.Protocol` for interfaces
- Use `TypeAlias` for type aliases (e.g., `ChatMessage`)
- Use modern Python 3.10+ syntax: `list[X]`, `dict[K, V]`, `X | None`
- Use `Literal` for string enums (e.g., `Literal["json", "text"]`)
- Add `from __future__ import annotations` when needed for forward references

### Async Patterns

- All LLM operations are async (`async def chat()`, `async def stream()`)
- OllamaProvider uses `ollama.AsyncClient`
- OpenAIProvider uses `openai.AsyncOpenAI`
- AnthropicProvider uses `anthropic.AsyncAnthropic`
- Test async functions with `pytest-asyncio`

### Method Signatures

The LLMProvider protocol defines two core methods:

**chat()** - Generate a complete response:

```python
async def chat(
    self,
    messages: list[ChatMessage],
    response_format: Literal["json", "text"] | type[BaseModel] = "text",
    max_tokens: int | None = None,
    tools: list[Tool] | None = None,
    temperature: float | None = None,
) -> AssistantMessage:
    """
    Generate a chat response from the LLM.

    Always returns AssistantMessage (not str).
    """
```

**stream()** - Stream response chunks:

```python
async def stream(
    self,
    messages: list[ChatMessage],
    response_format: Literal["json", "text"] | type[BaseModel] = "text",
    max_tokens: int | None = None,
    tools: list[Tool] | None = None,
    temperature: float | None = None,
) -> AsyncIterator[StreamChunk]:
    """
    Stream chat response chunks in real-time.

    Yields StreamChunk objects as tokens are generated.
    """
```

**get_usage()** - Get token usage from last call:

```python
def get_usage(self) -> Usage | None:
    """
    Get token usage statistics from the last chat() or stream() call.

    Returns Usage object with token counts, or None if no calls have been made.
    """
```

---

## Common Tasks

### Adding a New Provider

1. **Create provider file**: `src/casual_llm/providers/your_provider.py`

```python
from __future__ import annotations

from typing import Literal, AsyncIterator
from pydantic import BaseModel
from casual_llm.messages import ChatMessage, AssistantMessage, StreamChunk
from casual_llm.tools import Tool
from casual_llm.usage import Usage

class YourProvider:
    """Your provider implementation."""

    async def chat(
        self,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> AssistantMessage:
        # Implementation here
        # 1. Convert messages using message_converters
        # 2. Convert tools using tool_converters (if tools provided)
        # 3. Handle vision content (ImageContent) if present
        # 4. Call your LLM API
        # 5. Parse response including tool_calls if present
        # 6. Store usage statistics
        # 7. Return AssistantMessage
        ...

    async def stream(
        self,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        # Streaming implementation
        # 1. Convert messages and tools
        # 2. Call your LLM API with streaming enabled
        # 3. Yield StreamChunk objects as tokens arrive
        # 4. Update usage statistics
        ...
        yield StreamChunk(content="chunk", finish_reason=None)

    def get_usage(self) -> Usage | None:
        """Return token usage from last call."""
        return self._last_usage
```

2. **Export from providers/__init__.py**:
```python
from casual_llm.providers.your_provider import YourProvider

__all__ = [..., "YourProvider"]
```

3. **Add to create_provider()** if needed
4. **Write tests**: `tests/test_your_provider.py`
5. **Add example**: `examples/basic_your_provider.py`
6. **Update README.md** with usage example

### Adding a New Message Type

1. **Add to `src/casual_llm/messages.py`**:
```python
class NewMessage(BaseModel):
    """Description of new message type."""
    role: Literal["new_role"] = "new_role"
    content: str
```

2. **Update `ChatMessage` TypeAlias**:
```python
ChatMessage: TypeAlias = (
    AssistantMessage | SystemMessage | ToolResultMessage |
    UserMessage | NewMessage
)
```

3. **Update message converters** to handle the new type
4. **Add tests** to `tests/test_messages.py`
5. **Update examples** if relevant

### Updating Dependencies

```bash
# Add new dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Add optional dependency (edit pyproject.toml manually)
[project.optional-dependencies]
feature = ["package-name>=1.0.0"]
```

Then run `uv sync` to update lock file.

---

## Testing Guidelines

### Test Structure

- One test file per module: `test_messages.py`, `test_providers.py`, `test_tools.py`
- Use descriptive test names: `test_user_message_with_none_content()`
- Group related tests with classes (e.g., `TestOllamaProvider`)

### What to Test

**Message Models**:
- Creation and validation
- Serialization (`model_dump()`, `model_dump_json()`)
- Type aliases work correctly
- Edge cases (None values, empty lists, etc.)
- Multimodal content (TextContent, ImageContent)

**Tool Models**:
- Parameter definitions
- Nested objects and arrays
- Tool name flexibility (various naming conventions)
- `input_schema` property
- `from_input_schema()` class method

**Providers**:
- Mock external APIs (ollama.AsyncClient, openai.AsyncOpenAI, anthropic.AsyncAnthropic)
- Test both JSON and text response formats
- Test Pydantic model response formats
- Test tool calling (with and without tools)
- Test vision support (URL and base64 images)
- Test streaming (verify StreamChunk yields)
- Test usage tracking (get_usage())
- Test error handling
- Always verify returned type is AssistantMessage or StreamChunk

**Converters**:
- Message conversion to/from provider formats
- Tool conversion to provider formats
- Tool call parsing from provider responses
- Vision content conversion (ImageContent handling)

**Utilities**:
- Image fetching with various scenarios (success, errors, timeouts)
- HTTP/2 support verification
- Size limit validation
- Base64 encoding/decoding

### Running Specific Tests

```bash
# Single test function
uv run pytest tests/test_messages.py::test_user_message -v

# Single test class
uv run pytest tests/test_providers.py::TestOllamaProvider -v

# Single test file
uv run pytest tests/test_messages.py -v

# With output
uv run pytest tests/ -v -s
```

---

## Dependencies

### Core Dependencies (Required)

- `pydantic>=2.0.0` - Data validation and models
- `ollama>=0.6.1` - Official Ollama Python library
- `httpx[http2]>=0.28.1` - HTTP client with HTTP/2 support for image fetching

### Optional Dependencies

- `openai>=1.0.0` - For OpenAIProvider (install with `casual-llm[openai]`)
- `anthropic>=0.20.0` - For AnthropicProvider (install with `casual-llm[anthropic]`)

### Dev Dependencies

- `pytest>=7.0.0` - Testing framework
- `pytest-asyncio>=0.21.0` - Async test support
- `pytest-cov>=4.0.0` - Coverage reporting
- `black>=23.0.0` - Code formatting
- `ruff>=0.1.0` - Fast linting
- `mypy>=1.0.0` - Type checking

---

## Release Process

### Version Numbering

Follow semantic versioning (semver):
- **Patch** (0.1.X): Bug fixes, documentation
- **Minor** (0.X.0): New features, backwards compatible
- **Major** (X.0.0): Breaking changes

### Creating a Release

1. **Update version** in `pyproject.toml` and `src/casual_llm/__init__.py`
2. **Update CHANGELOG.md** with changes
3. **Run all checks**:
```bash
uv run black src/ tests/ examples/
uv run ruff check src/ tests/ examples/
uv run mypy src/casual_llm
uv run pytest tests/
```

4. **Build package**:
```bash
uv add --dev build twine
uv run python -m build
```

5. **Create git tag**:
```bash
git tag v0.X.Y
git push origin v0.X.Y
```

6. **Publish to PyPI**:
```bash
uv run twine upload dist/*
```

---

## Integration with casual-* Ecosystem

### Relationship to casual-mcp

**casual-llm** provides the foundational message models, tool models, and provider abstractions that **casual-mcp** builds upon:

```
casual-mcp (orchestration, MCP server integration)
    ↓ depends on
casual-llm (providers, messages, tools)
```

**Important**: Message and tool models were moved FROM casual-mcp TO casual-llm to create a single source of truth. casual-mcp should re-export them for backwards compatibility.

### Future: casual-memory

The planned **casual-memory** library will also depend on casual-llm:

```
casual-memory (memory extraction, conflict detection)
    ↓ depends on
casual-llm (providers, messages)
```

---

## Troubleshooting

### Import Errors

```python
# ❌ Wrong
from casual_llm.providers.ollama import OllamaProvider

# ✅ Correct
from casual_llm import OllamaProvider
```

### OpenAI Provider Not Found

```bash
# Install optional dependency
uv add casual-llm[openai]
# or
uv sync --all-extras
```

### Tests Failing

```bash
# Make sure dev dependencies are installed
uv sync --all-extras

# Check if pytest is available
uv run pytest --version

# Run with verbose output
uv run pytest tests/ -v -s
```

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve

# Pull a model
ollama pull qwen2.5:7b-instruct
```

---

## Code Style

### Imports

```python
# Standard library
import logging
import uuid
from typing import Any, Literal

# Third-party
from pydantic import BaseModel, Field
from ollama import AsyncClient

# Local
from casual_llm.messages import ChatMessage, AssistantMessage
from casual_llm.tools import Tool
```

### Type Hints

Use modern Python 3.10+ syntax:

```python
# ✅ Correct (Python 3.10+)
def process(items: list[str]) -> dict[str, int]:
    ...

# ❌ Old style (avoid)
from typing import List, Dict
def process(items: List[str]) -> Dict[str, int]:
    ...
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)  # ✅ Correct

# Use lazy formatting for performance
logger.debug("Processing %d messages", len(messages))

# ❌ Avoid f-strings in logging (evaluates even if logging disabled)
logger.debug(f"Processing {len(messages)} messages")
```

### Docstrings

Use Google-style docstrings:

```python
def create_provider(
    model_config: ModelConfig,
    timeout: float = 60.0,
) -> LLMProvider:
    """
    Factory function to create an LLM provider.

    Args:
        model_config: Model configuration with provider details
        timeout: HTTP timeout in seconds (default: 60.0)

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider type is not supported
        ImportError: If openai package is not installed for OpenAI provider

    Examples:
        >>> config = ModelConfig(name="gpt-4", provider=Provider.OPENAI)
        >>> provider = create_provider(config)
    """
```

---

## Important Notes

### Don't Break Backwards Compatibility

Once published to PyPI:
- Don't remove public APIs without deprecation period
- Don't change function signatures (add optional params only)
- Don't change message model field names
- Don't change return types
- Document breaking changes in CHANGELOG.md

### Keep It Lightweight

- Avoid adding dependencies unless absolutely necessary
- Keep optional dependencies optional (e.g., openai)
- Don't add features that belong in casual-mcp or casual-memory
- Focus on core provider abstraction and message models

### Protocol-Based Design

- Use `typing.Protocol` for interfaces
- Don't require inheritance from base classes
- Make it easy to create custom providers
- Support any provider that implements the protocol

### Tool Calling

- Tool names are flexible strings (not restricted to Python identifiers)
- Support various naming conventions (snake_case, kebab-case, dotted)
- Always return AssistantMessage from `chat()`, never just a string
- Generate unique tool call IDs if provider doesn't provide them

### Vision Support

- **OpenAI and Anthropic**: Support both URL and base64 images natively - no client-side fetching needed
- **Ollama**: Requires client-side fetching and base64 encoding (handled by image utilities)
- Use `ImageContent` for images in multimodal messages
- Use `TextContent` for text in multimodal messages
- Image utilities use HTTP/2 for reliable fetching from sites like Wikipedia
- Include proper User-Agent headers to avoid bot detection

### Media Output & Audio (Shelved)

Image output and audio I/O support was researched and shelved (Feb 2026). See [ADR-001](docs/decisions/001-shelve-media-support.md) for details and design decisions for future implementation.

### Streaming

- All providers must implement both `chat()` and `stream()` methods
- Streaming yields `StreamChunk` objects with `content` and `finish_reason`
- Usage statistics should be updated after streaming completes
- Stream method signature matches chat method for consistency

---

## Getting Help

- **Issues**: https://github.com/casualgenius/casual-llm/issues
- **Discussions**: https://github.com/casualgenius/casual-llm/discussions
- **casual-mcp**: https://github.com/AlexStansfield/casual-mcp

---

## Quick Reference

### Run Tests
```bash
uv run pytest tests/
```

### Format Code
```bash
uv run black src/ tests/ examples/
```

### Type Check
```bash
uv run mypy src/casual_llm
```

### Run Examples
```bash
# Basic examples
uv run python examples/message_formatting.py
uv run python examples/basic_ollama.py
uv run python examples/basic_openai.py
uv run python examples/basic_anthropic.py

# Advanced examples
uv run python examples/tool_calling.py
uv run python examples/vision_example.py
uv run python examples/streaming_example.py
```

### Build Package
```bash
uv run python -m build
```

---

**Remember**: casual-llm is about being lightweight, protocol-based, and easy to use. Keep it simple! 🎯
