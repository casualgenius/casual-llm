# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**casual-llm** is a lightweight, protocol-based LLM provider abstraction library with standardized OpenAI-compatible message models. It's part of the casual-* ecosystem of lightweight AI tools.

### Key Design Principles

1. **Lightweight** - Minimal dependencies (pydantic, httpx)
2. **Protocol-based** - Uses `typing.Protocol`, no inheritance required
3. **Type-safe** - Full type hints with py.typed marker
4. **OpenAI-compatible** - Standard message format used across the industry
5. **Async-first** - Built for modern async Python

### Repository Structure

```
casual-llm/
├── src/casual_llm/              # Main package
│   ├── __init__.py              # Public API exports (main entry point)
│   ├── messages.py              # OpenAI-compatible message models
│   ├── utils.py                 # JSON extraction utilities
│   ├── py.typed                 # PEP 561 type marker
│   └── providers/               # Provider implementations
│       ├── __init__.py          # Provider exports + create_provider() factory
│       ├── base.py              # LLMProvider protocol + ModelConfig
│       ├── ollama.py            # OllamaProvider (with retry logic)
│       └── openai.py            # OpenAIProvider (optional dependency)
├── tests/                       # Test suite
│   └── test_messages.py         # Message model tests
├── examples/                    # Working examples
│   ├── basic_ollama.py          # Ollama usage
│   ├── basic_openai.py          # OpenAI usage
│   └── message_formatting.py    # All message types demo
└── docs/                        # Documentation (future)
```

---

## Development Workflow

This project uses **[uv](https://github.com/astral-sh/uv)** for dependency management (not pip/poetry).

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

# Ollama example (requires Ollama running locally)
uv run python examples/basic_ollama.py

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
- `ChatMessage` - Type alias for any message type

**Important**: These were moved from `casual-mcp` to create a single source of truth for message models across the casual-* ecosystem.

### 2. Provider Protocol (`src/casual_llm/providers/base.py`)

- `LLMProvider` - Protocol defining the interface for all providers
- `LLMMessage` - Simple message format for internal use
- `ModelConfig` - Configuration dataclass for models
- `Provider` - Enum (OPENAI, OLLAMA)

**Key Design**: Uses `typing.Protocol` so custom providers don't need to inherit from a base class.

### 3. Provider Implementations

**OllamaProvider** (`src/casual_llm/providers/ollama.py`):
- Connects to local Ollama server
- Built-in retry logic with exponential backoff
- Optional metrics tracking (success/failure counts)
- Supports both JSON and text response formats

**OpenAIProvider** (`src/casual_llm/providers/openai.py`):
- Works with OpenAI API and compatible services (OpenRouter, etc.)
- Optional dependency (`pip install casual-llm[openai]`)
- Async-first using `AsyncOpenAI` client

### 4. Factory Function (`create_provider()`)

Located in `src/casual_llm/providers/__init__.py`:
- Creates provider instances from `ModelConfig`
- Handles optional dependencies gracefully
- Provides clear error messages

---

## API Design Guidelines

### Public vs Private

**Public API** (exported from `src/casual_llm/__init__.py`):
- All message models
- Provider protocol and implementations
- `create_provider()` factory
- Utilities like `extract_json_from_markdown()`

**Internal API** (not exported):
- Provider implementation details
- Internal helper functions

### Type Hints

- **Always** include type hints on all public APIs
- Use `typing.Protocol` for interfaces
- Use `TypeAlias` for type aliases (e.g., `ChatMessage`)
- Include `Optional[X]` for optional parameters
- Use `Literal` for string enums (e.g., `Literal["json", "text"]`)

### Async Patterns

- All LLM operations are async (`async def generate()`)
- Use `httpx.AsyncClient` (not `requests`)
- Use `async with` for context managers
- Test async functions with `pytest-asyncio`

---

## Common Tasks

### Adding a New Provider

1. **Create provider file**: `src/casual_llm/providers/your_provider.py`

```python
from typing import List, Literal, Optional
from casual_llm.providers.base import LLMMessage

class YourProvider:
    """Your provider implementation."""

    async def generate(
        self,
        messages: List[LLMMessage],
        response_format: Literal["json", "text"] = "text",
        max_tokens: Optional[int] = None,
    ) -> str:
        # Implementation here
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

3. **Add tests** to `tests/test_messages.py`
4. **Update examples** if relevant

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

- One test file per module: `test_messages.py`, `test_providers.py`, etc.
- Use descriptive test names: `test_user_message_with_none_content()`
- Group related tests with comments or classes

### What to Test

**Message Models**:
- Creation and validation
- Serialization (`model_dump()`, `model_dump_json()`)
- Type aliases work correctly
- Edge cases (None values, empty lists, etc.)

**Providers**:
- Mock external APIs (httpx, OpenAI client)
- Test both JSON and text response formats
- Test error handling
- Test retry logic (OllamaProvider)

### Running Specific Tests

```bash
# Single test function
uv run pytest tests/test_messages.py::test_user_message -v

# Single test file
uv run pytest tests/test_messages.py -v

# With output
uv run pytest tests/ -v -s
```

---

## Dependencies

### Core Dependencies (Required)

- `pydantic>=2.0.0` - Data validation and models
- `httpx>=0.27.0` - Async HTTP client

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

**casual-llm** provides the foundational message models and provider abstractions that **casual-mcp** builds upon:

```
casual-mcp (orchestration, tool calling)
    ↓ depends on
casual-llm (providers, messages)
```

**Important**: Message models were moved FROM casual-mcp TO casual-llm to create a single source of truth. casual-mcp should re-export them for backwards compatibility.

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

---

## Code Style

### Imports

```python
# Standard library
import logging
from typing import List, Optional

# Third-party
import httpx
from pydantic import BaseModel

# Local
from casual_llm.providers.base import LLMMessage
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)  # ✅ Correct

# ❌ Wrong (old "dixie" pattern from ai-assistant)
logger = logging.getLogger("dixie.casual_llm.module")
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
- Document breaking changes in CHANGELOG.md

### Keep It Lightweight

- Avoid adding dependencies unless absolutely necessary
- Keep optional dependencies optional
- Don't add features that belong in casual-mcp or casual-memory

### Protocol-Based Design

- Use `typing.Protocol` for interfaces
- Don't require inheritance from base classes
- Make it easy to create custom providers

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
```

### Build Package
```bash
uv run python -m build
```

---

**Remember**: casual-llm is about being lightweight, protocol-based, and easy to use. Keep it simple! 🎯
