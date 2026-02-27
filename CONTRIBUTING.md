# Contributing to casual-llm

Thank you for your interest in contributing to casual-llm! This document provides guidelines for contributing to the project.

## Development Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/casualgenius/casual-llm.git
cd casual-llm

# Install dependencies with uv
uv sync

# Install with development dependencies and all providers
uv sync --all-extras
```

## Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=casual_llm --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_messages.py -v
```

## Code Quality

### Formatting and Linting

We use [ruff](https://github.com/astral-sh/ruff) for both formatting and linting:

```bash
# Format all code
uv run ruff format src/ tests/ examples/

# Check formatting without making changes
uv run ruff format --check src/ tests/ examples/

# Run linter
uv run ruff check src/ tests/ examples/

# Auto-fix issues
uv run ruff check --fix src/ tests/ examples/
```

### Type Checking

We use [mypy](https://github.com/python/mypy) for static type checking:

```bash
# Run type checker
uv run mypy src/casual_llm
```

### Run All Checks

```bash
# Format, lint, type check, and test
uv run ruff format src/ tests/ examples/
uv run ruff check src/ tests/ examples/
uv run mypy src/casual_llm
uv run pytest tests/
```

## Architecture Overview

casual-llm uses a **Client/Model/ChatOptions** architecture:

- **Client** (`LLMClient` protocol) — manages API connections (`OpenAIClient`, `OllamaClient`, `AnthropicClient`)
- **Model** — wraps a client with model-specific configuration, provides `chat()` and `stream()`
- **ChatOptions** — dataclass controlling per-request behavior (temperature, tools, format, etc.)

```
User code → Model.chat(messages, options) → Client._chat(model, messages, options) → Provider API
```

## Contribution Guidelines

### Before Submitting a PR

1. **Write tests** - All new features should include tests
2. **Update documentation** - Update relevant docs if adding new features
3. **Format code** - Run ruff format and ruff check
4. **Type hints** - Add type hints to all new code (Python 3.10+ syntax)
5. **Run tests** - Ensure all tests pass

### PR Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run all checks (format, lint, type check, test)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add retry logic to OpenAI client

- Implement exponential backoff for transient failures
- Add max_retries parameter to OpenAIClient
- Update tests to cover retry scenarios
```

## Adding a New Provider

To add a new LLM provider:

1. Create `src/casual_llm/providers/your_provider.py`
2. Implement the `LLMClient` protocol (`_chat` and `_stream` methods)
3. Create message converter in `src/casual_llm/message_converters/your_provider.py`
4. Create tool converter in `src/casual_llm/tool_converters/your_provider.py`
5. Add to `src/casual_llm/providers/__init__.py`
6. Update `create_client()` factory in `src/casual_llm/factory.py`
7. Add tests in `tests/test_your_provider.py`
8. Add example in `examples/basic_your_provider.py`
9. Update documentation

### Example Provider Template

```python
from __future__ import annotations

from typing import Any, AsyncIterator

from casual_llm.config import ChatOptions
from casual_llm.messages import ChatMessage, AssistantMessage, StreamChunk
from casual_llm.usage import Usage


class YourClient:
    """Your LLM client implementation."""

    def __init__(self, api_key: str | None = None, timeout: float = 60.0):
        # Initialize your SDK client
        ...

    async def _chat(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> tuple[AssistantMessage, Usage | None]:
        # 1. Convert messages to provider format
        # 2. Build request kwargs from options
        # 3. Call your LLM API
        # 4. Parse response into AssistantMessage
        # 5. Extract usage statistics
        # 6. Return (AssistantMessage, Usage)
        ...

    async def _stream(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> AsyncIterator[StreamChunk]:
        # 1. Convert messages to provider format
        # 2. Call your LLM API with streaming
        # 3. Yield StreamChunk objects as tokens arrive
        ...
        yield StreamChunk(content="chunk", finish_reason=None)
```

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues and discussions first

## Code of Conduct

- Be respectful and constructive
- Focus on the code, not the person
- Welcome newcomers and help them learn
- Follow Python community standards

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
