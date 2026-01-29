# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.3] - 2024-01-28

### Fixed
- Support for tool parameters that use `anyOf` / `oneOf` for type

## [0.4.2] - 2024-01-28

### Added
- README.md to examples folder to explain how to use them

### Changed
- Updated main README.md to document vision support, anthropic provider and streaming support

## [0.4.1] - 2024-12-30

### Fixed
- Bumped version (forgot to for 0.4.0) for pypi release

## [0.4.0] - 2024-12-29

### Added
- Provider for Anthropic

## [0.3.0] - 2024-12-26

### Added
- **Vision/Multimodal Support**: Full support for vision-capable models with image content
  - `ImageContent` and `TextContent` models for multimodal messages
  - Support for both URL-based and base64-encoded images
  - Automatic image fetching and encoding for Ollama provider
  - Works with OpenAI (gpt-4o, gpt-4.1-nano) and Ollama (llava) vision models
  - Example: `examples/vision_example.py` demonstrating vision capabilities with both providers
- **Streaming Support**: Stream responses from LLMs in real-time
  - `stream()` method on all providers returning `AsyncIterator[StreamChunk]`
  - `StreamChunk` model for streaming response chunks
  - Support for streaming with both OpenAI and Ollama providers
  - Example: `examples/stream_example.py` demonstrating streaming with multi-turn conversations
- **Image Utilities**: Helper functions for image processing
  - `fetch_image_as_base64()` - Download and encode images from URLs
  - `strip_base64_prefix()` - Remove data URI prefixes from base64 strings
  - `add_base64_prefix()` - Add data URI prefixes to base64 strings

### Changed
- **Core dependency added**: `httpx[http2]>=0.28.1` now required for reliable image fetching
- **Test coverage**: Added 23 comprehensive tests for image utilities (95% coverage)
- **Total test count**: Now 182 tests (up from 159)

### Fixed
- Image fetching from Wikipedia and similar sites (HTTP/2 and User-Agent headers)
- Error messages now properly indicate `httpx[http2]` installation requirement

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

[0.3.0]: https://github.com/casualgenius/casual-llm/releases/tag/v0.3.0
[0.2.0]: https://github.com/casualgenius/casual-llm/releases/tag/v0.2.0
[0.1.0]: https://github.com/casualgenius/casual-llm/releases/tag/v0.1.0
