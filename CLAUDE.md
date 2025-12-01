# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**casual-llm** is a lightweight, protocol-based LLM provider abstraction library with standardized OpenAI-compatible message models. It's part of the casual-* ecosystem of lightweight AI tools.

### Key Design Principles

1. **Lightweight** - Minimal dependencies (pydantic, ollama)
2. **Protocol-based** - Uses `typing.Protocol`, no inheritance required
3. **Type-safe** - Full type hints with py.typed marker (Python 3.10+)
4. **OpenAI-compatible** - Standard message format used across the industry
5. **Async-first** - Built for modern async Python
6. **Tool calling** - First-class support for function/tool calling

### Repository Structure

```
casual-llm/
├── src/casual_llm/              # Main package
│   ├── __init__.py              # Public API exports (main entry point)
│   ├── messages.py              # OpenAI-compatible message models
│   ├── tools.py                 # Tool and ToolParameter models
│   ├── config.py                # ModelConfig and Provider enum
│   ├── message_converters.py    # Message format converters
│   ├── tool_converters.py       # Tool format converters
│   ├── utils.py                 # JSON extraction utilities
│   ├── py.typed                 # PEP 561 type marker
│   └── providers/               # Provider implementations
│       ├── __init__.py          # Provider exports + create_provider() factory
│       ├── base.py              # LLMProvider protocol
│       ├── ollama.py            # OllamaProvider (with retry logic)
│       └── openai.py            # OpenAIProvider (optional dependency)
├── tests/                       # Test suite
│   ├── test_messages.py         # Message model tests
│   ├── test_tools.py            # Tool model tests
│   └── test_providers.py        # Provider tests
├── examples/                    # Working examples
│   ├── basic_ollama.py          # Ollama usage
│   ├── basic_openai.py          # OpenAI usage
│   ├── message_formatting.py    # All message types demo
│   ├── tool_definitions.py      # Tool definition examples
│   └── tool_calling.py          # Complete tool calling workflow
└── docs/                        # Documentation (future)
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

# Install with all extras (includes dev dependencies and openai)
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

# Tool definitions example
uv run python examples/tool_definitions.py

# Ollama example (requires Ollama running locally)
uv run python examples/basic_ollama.py

# Tool calling example (requires Ollama running locally)
uv run python examples/tool_calling.py

# OpenAI example (requires OPENAI_API_KEY env var)
uv run python examples/basic_openai.py
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

- `UserMessage` - Message from the user
- `AssistantMessage` - Message from the AI (with optional tool_calls)
- `SystemMessage` - System prompt that sets behavior
- `ToolResultMessage` - Result from tool/function execution
- `AssistantToolCall` - Tool call structure
- `AssistantToolCallFunction` - Function details within a tool call
- `ChatMessage` - Type alias for any message type

**Important**: These were moved from `casual-mcp` to create a single source of truth for message models across the casual-* ecosystem.

### 2. Tool Models (`src/casual_llm/tools.py`)

Models for defining LLM function calling tools:

- `Tool` - Tool definition with name, description, parameters, and required fields
- `ToolParameter` - JSON Schema-based parameter definition

**Design**: Tool names are flexible strings (not restricted to Python identifiers) to support various naming conventions (snake_case, kebab-case, dotted notation).

### 3. Provider Protocol (`src/casual_llm/providers/base.py`)

- `LLMProvider` - Protocol defining the interface for all providers
- The protocol defines a single `chat()` method that returns `AssistantMessage`

**Key Design**: Uses `typing.Protocol` so custom providers don't need to inherit from a base class.

### 4. Configuration (`src/casual_llm/config.py`)

- `ModelConfig` - Dataclass for model configuration
- `Provider` - Enum (OPENAI, OLLAMA)

### 5. Provider Implementations

**OllamaProvider** (`src/casual_llm/providers/ollama.py`):
- Uses the official `ollama` Python library with `AsyncClient`
- Built-in retry logic with exponential backoff
- Optional metrics tracking (success/failure counts)
- Supports both JSON and text response formats
- Supports tool calling
- Generates unique tool call IDs if Ollama doesn't provide them

**OpenAIProvider** (`src/casual_llm/providers/openai.py`):
- Works with OpenAI API and compatible services (OpenRouter, etc.)
- Optional dependency (`uv add casual-llm[openai]`)
- Async-first using `AsyncOpenAI` client
- Supports tool calling

### 6. Converters

**Message Converters** (`src/casual_llm/message_converters.py`):
- `convert_messages_to_openai()` - Convert ChatMessage list to OpenAI format
- `convert_messages_to_ollama()` - Convert ChatMessage list to Ollama format
- `convert_tool_calls_from_openai()` - Parse OpenAI tool calls
- `convert_tool_calls_from_ollama()` - Parse Ollama tool calls

**Tool Converters** (`src/casual_llm/tool_converters.py`):
- `tool_to_ollama()` - Convert Tool to Ollama format
- `tools_to_ollama()` - Convert multiple tools
- `tool_to_openai()` - Convert Tool to OpenAI format
- `tools_to_openai()` - Convert multiple tools

**Note**: OpenAI and Ollama use the same tool format, so the converters share implementation.

### 7. Factory Function (`create_provider()`)

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
- Utilities like `extract_json_from_markdown()`

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

- All LLM operations are async (`async def chat()`)
- OllamaProvider uses `ollama.AsyncClient`
- OpenAIProvider uses `openai.AsyncOpenAI`
- Test async functions with `pytest-asyncio`

### Method Signatures

The core method is `chat()`:

```python
async def chat(
    self,
    messages: list[ChatMessage],
    response_format: Literal["json", "text"] = "text",
    max_tokens: int | None = None,
    tools: list[Tool] | None = None,
) -> AssistantMessage:
    """
    Generate a chat response from the LLM.

    Always returns AssistantMessage (not str).
    """
```

---

## Common Tasks

### Adding a New Provider

1. **Create provider file**: `src/casual_llm/providers/your_provider.py`

```python
from __future__ import annotations

from typing import Literal
from casual_llm.messages import ChatMessage, AssistantMessage
from casual_llm.tools import Tool

class YourProvider:
    """Your provider implementation."""

    async def chat(
        self,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
    ) -> AssistantMessage:
        # Implementation here
        # 1. Convert messages using message_converters
        # 2. Convert tools using tool_converters (if tools provided)
        # 3. Call your LLM API
        # 4. Parse response including tool_calls if present
        # 5. Return AssistantMessage
        ...
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

**Tool Models**:
- Parameter definitions
- Nested objects and arrays
- Tool name flexibility (various naming conventions)
- `input_schema` property
- `from_input_schema()` class method

**Providers**:
- Mock external APIs (ollama.AsyncClient, openai.AsyncOpenAI)
- Test both JSON and text response formats
- Test tool calling (with and without tools)
- Test error handling
- Test retry logic (OllamaProvider)
- Always verify returned type is AssistantMessage

**Converters**:
- Message conversion to/from provider formats
- Tool conversion to provider formats
- Tool call parsing from provider responses

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

### Optional Dependencies

- `openai>=1.0.0` - For OpenAIProvider (install with `casual-llm[openai]`)

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
    max_retries: int = 0,
    enable_metrics: bool = False,
) -> LLMProvider:
    """
    Factory function to create an LLM provider.

    Args:
        model_config: Model configuration with provider details
        timeout: HTTP timeout in seconds (default: 60.0)
        max_retries: Number of retry attempts for transient failures
        enable_metrics: Enable metrics tracking (Ollama only)

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

---

## Getting Help

- **Issues**: https://github.com/AlexStansfield/casual-llm/issues
- **Discussions**: https://github.com/AlexStansfield/casual-llm/discussions
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

### Run Example
```bash
uv run python examples/message_formatting.py
uv run python examples/tool_calling.py
```

### Build Package
```bash
uv run python -m build
```

---

**Remember**: casual-llm is about being lightweight, protocol-based, and easy to use. Keep it simple! 🎯
