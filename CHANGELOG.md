# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-01-XX

### Added
- **Pydantic model support for structured output**: `response_format` parameter now accepts Pydantic `BaseModel` classes for JSON Schema-based structured output
- Both `OllamaProvider` and `OpenAIProvider` support Pydantic models for automatic schema generation

### Changed
- **Message converters refactored into packages**: Split `message_converters.py` and `tool_converters.py` into separate packages with `openai.py` and `ollama.py` modules
- **Type safety improvements**: Added proper SDK types (`ChatCompletionMessageToolCall`, `ChatCompletionToolParam`) for better type checking
- **Temperature defaults to None**: Changed default temperature from 0.1 to None (uses provider defaults)
- **Usage model uses computed field**: Refactored `Usage.total_tokens` to use Pydantic's `@computed_field` instead of custom `__init__`

### Removed
- **Retry logic and metrics tracking**: Removed `max_retries` and `enable_metrics` parameters from `OllamaProvider` to keep the library lightweight and focused
- **Removed `tool_definitions.py` example**: Consolidated examples to focus on practical usage patterns

### Fixed
- Tool calling with Ollama: Fixed argument conversion between JSON strings and dicts for proper round-trip handling

## [0.1.0] - 2025-01-03

Initial release of casual-llm - a lightweight, protocol-based LLM provider abstraction library.

### Added

**Core Features:**
- `LLMProvider` protocol defining a unified interface for LLM interactions
- `OllamaProvider` implementation using the official ollama library
- `OpenAIProvider` implementation (optional dependency)
- `create_provider()` factory function for easy provider instantiation
- Full type hint support with py.typed marker
- Async-first design using modern async/await patterns

**Message Models:**
- OpenAI-compatible message models: `UserMessage`, `AssistantMessage`, `SystemMessage`, `ToolResultMessage`
- `ChatMessage` type alias for any message type
- `AssistantToolCall` and `AssistantToolCallFunction` for tool calling
- All message models built on Pydantic for validation and type safety

**Tool Calling Support:**
- `Tool` and `ToolParameter` models for defining function/tool schemas
- Tool format converters: `tool_to_ollama()`, `tools_to_ollama()`, `tool_to_openai()`, `tools_to_openai()`
- Support for nested objects, arrays, and enums in tool parameters
- Support for custom JSON Schema validation fields (minLength, maxLength, pattern, etc.)
- `Tool.from_input_schema()` factory method for MCP-style tool definitions
- `Tool.input_schema` property for exporting to MCP format

**Message Converters:**
- `convert_messages_to_openai()` - Convert ChatMessage list to OpenAI format
- `convert_messages_to_ollama()` - Convert ChatMessage list to Ollama format
- `convert_tool_calls_from_openai()` - Parse tool calls from OpenAI responses
- `convert_tool_calls_from_ollama()` - Parse tool calls from Ollama responses

**Usage Tracking:**
- `Usage` model for tracking token consumption (prompt_tokens, completion_tokens, total_tokens)
- `get_usage()` method on all providers to retrieve usage from last API call
- Automatic token tracking for both OpenAI and Ollama providers

**Examples:**
- `examples/basic_ollama.py` - Basic Ollama usage with text and JSON responses
- `examples/basic_openai.py` - Basic OpenAI usage with system messages
- `examples/message_formatting.py` - All message types demonstration
- `examples/tool_calling.py` - Complete tool calling workflow

**Configuration:**
- `ModelConfig` dataclass for provider configuration
- `Provider` enum (OPENAI, OLLAMA)
- Support for custom base URLs (for OpenRouter, LM Studio, etc.)
- Optional temperature parameter (defaults to provider defaults if not specified)

### Design Principles

- **Protocol-based**: Uses `typing.Protocol` - no inheritance required
- **Lightweight**: Minimal dependencies (pydantic, ollama)
- **Type-safe**: Full type hints throughout
- **OpenAI-compatible**: Standard message format used across the industry
- **Python 3.10+**: Modern Python with lowercase generic type hints (list[X], dict[K,V])

### Dependencies

- `pydantic>=2.0.0` - Data validation and models
- `ollama>=0.6.1` - Official Ollama Python library
- `openai>=1.0.0` - Optional, for OpenAI provider

[0.2.0]: https://github.com/casualgenius/casual-llm/releases/tag/v0.2.0
[0.1.0]: https://github.com/casualgenius/casual-llm/releases/tag/v0.1.0
