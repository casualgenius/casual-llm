# ADR-001: Shelve Media Output & Audio Support

**Date**: 2026-02-24
**Status**: Shelved
**Branch**: `feat/better-media-support` (closed, no changes)

## Context

For v1.0.0 we investigated adding image output, audio input, and audio output support to casual-llm. The goal was to make multimodal capabilities comprehensive beyond the existing vision (image input) support.

## Research Findings

### Audio I/O
- Only available via OpenAI Chat Completions API (`modalities: ["text", "audio"]`)
- Not supported in OpenAI Responses API (listed as "coming soon")
- No Anthropic or Ollama support

### Image Output
- Only available via OpenAI Responses API (`image_generation` tool)
- Not available in Chat Completions API

### Migration Concern
- Most OpenAI-compatible providers (OpenRouter, Together, Groq, etc.) implement Chat Completions, not Responses API
- Migrating the OpenAI provider to Responses API would break compatibility with these providers

### Net Result
Audio and image output require different, incompatible APIs. Neither can be added cleanly without significant trade-offs.

## Decision

Shelve this work. Revisit when the API landscape consolidates.

## Revisit Conditions

- OpenAI Responses API adds audio support
- OpenAI-compatible providers adopt the Responses API
- A clean dual-API strategy becomes viable (e.g., Chat Completions as default, Responses API opt-in)

## Design Decisions (for future implementation)

These decisions were made during research and should be followed when this work resumes:

1. **`AssistantMessage.content` type**: Should become `str | list[TextContent | ImageContent] | None` for multimodal output
2. **Reuse `ImageContent`**: The existing model works for both input and output (no new model needed)
3. **Error on unsupported**: Providers that don't support a requested media type should raise errors, not skip silently
4. **StreamChunk stays text-only**: Media output only via `chat()`, not streaming
