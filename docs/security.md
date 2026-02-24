# Security

This document covers security considerations when using casual-llm.

## SSRF Protection

When using vision features with Ollama, casual-llm fetches images from URLs on the client side. To prevent Server-Side Request Forgery (SSRF) attacks, the image fetcher includes several protections:

- **Scheme validation**: Only `http` and `https` URLs are allowed. Schemes like `file://`, `ftp://`, and `data://` are blocked.
- **Private IP blocking**: Requests to private/internal IP ranges are blocked:
  - Loopback addresses (`127.0.0.0/8`, `::1`)
  - Private networks (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`)
  - Link-local addresses (`169.254.0.0/16`, `fe80::/10`)
  - Cloud metadata endpoints (e.g., `169.254.169.254`)
- **Redirect safety**: Redirects are followed manually (max 5 hops) with each redirect target validated against the same rules.
- **Size limits**: Images larger than 10 MB are rejected.

**Note:** OpenAI and Anthropic handle image URLs server-side. SSRF protection applies only to Ollama's client-side image fetching.

## API Key Handling

- **Environment variables**: All clients support API key resolution from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
- **Never log keys**: API keys should not appear in logs. Avoid passing keys through user-controlled input.
- **`ClientConfig.name` auto-resolution**: When using `ClientConfig` with a `name` field, the library checks `{NAME}_API_KEY` env vars automatically. Ensure the `name` field is not derived from untrusted user input.

## `options.extra` Safety

The `ChatOptions.extra` dict allows passing provider-specific kwargs to the underlying API. To prevent parameter injection:

- Keys in `extra` that conflict with core request parameters (`model`, `messages`, `system`, `tools`, etc.) are **silently ignored** with a warning logged.
- Never populate `extra` directly from untrusted user input without validation.

```python
# Safe: controlled by application code
opts = ChatOptions(extra={"logprobs": True})

# Unsafe: user-controlled extra dict
opts = ChatOptions(extra=user_provided_dict)  # Don't do this
```

## Input Validation

- **Message content**: casual-llm passes message content to provider APIs as-is. Sanitize user input before including it in messages if your application requires it.
- **Tool arguments**: Tool call arguments from LLM responses are JSON strings. Always validate and sanitize before executing tool calls.
- **`media_type` on ImageContent**: The `media_type` field accepts any string. For defense in depth, consider validating against known image MIME types before sending to providers.

## Dependency Security

casual-llm has minimal core dependencies:

- **pydantic** (>=2.0.0) — data validation
- **httpx** (>=0.28.1) — HTTP client for image fetching

Provider SDKs are optional:

- **ollama** — `casual-llm[ollama]`
- **openai** — `casual-llm[openai]`
- **anthropic** — `casual-llm[anthropic]`

Keep dependencies up to date with `uv sync` to receive security patches.

## Reporting Security Issues

If you discover a security vulnerability, please report it responsibly:

- **Do not** open a public GitHub issue
- Email security concerns to the maintainers
- See [GitHub Security Advisories](https://github.com/casualgenius/casual-llm/security) for reporting

## Best Practices

1. **Use environment variables** for API keys — never hardcode them
2. **Validate user input** before including in LLM messages
3. **Don't pass untrusted data** into `ChatOptions.extra`
4. **Validate tool call results** from LLM responses before executing
5. **Keep dependencies updated** for security patches
6. **Use HTTPS** for all API connections (default for all providers)
