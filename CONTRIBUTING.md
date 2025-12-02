# Contributing to casual-llm

Thank you for your interest in contributing to casual-llm! This document provides guidelines for contributing to the project.

## Development Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/casualgenius/casual-llm.git
cd casual-llm

# Install dependencies with uv
uv sync

# Install with development dependencies
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

### Formatting

We use [black](https://github.com/psf/black) for code formatting:

```bash
# Format all code
uv run black src/ tests/ examples/

# Check formatting without making changes
uv run black --check src/ tests/ examples/
```

### Linting

We use [ruff](https://github.com/astral-sh/ruff) for linting:

```bash
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
uv run black src/ tests/ examples/
uv run ruff check src/ tests/ examples/
uv run mypy src/casual_llm
uv run pytest tests/
```

## Running Examples

```bash
# Run Ollama example (requires Ollama installed)
uv run python examples/basic_ollama.py

# Run OpenAI example (requires OPENAI_API_KEY)
uv run python examples/basic_openai.py

# Run message formatting example (no dependencies)
uv run python examples/message_formatting.py
```

## Contribution Guidelines

### Before Submitting a PR

1. **Write tests** - All new features should include tests
2. **Update documentation** - Update README.md if adding new features
3. **Format code** - Run black and ruff
4. **Type hints** - Add type hints to all new code
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
Add retry logic to OpenAI provider

- Implement exponential backoff for transient failures
- Add max_retries parameter to create_provider()
- Update tests to cover retry scenarios
```

## Adding a New Provider

To add a new LLM provider:

1. Create `src/casual_llm/providers/your_provider.py`
2. Implement the `LLMProvider` protocol
3. Add to `src/casual_llm/providers/__init__.py`
4. Update `create_provider()` factory function
5. Add tests in `tests/test_your_provider.py`
6. Add example in `examples/basic_your_provider.py`
7. Update README.md with usage example

### Example Provider Template

```python
from typing import List, Literal, Optional
from casual_llm.providers.base import LLMMessage

class YourProvider:
    """Your LLM provider implementation."""

    async def generate(
        self,
        messages: List[LLMMessage],
        response_format: Literal["json", "text"] = "text",
        max_tokens: Optional[int] = None,
    ) -> str:
        # Your implementation here
        ...
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
