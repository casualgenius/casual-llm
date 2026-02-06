# Quick Start Guide

This guide covers getting started with casual-llm using different providers.

## Using Ollama

Ollama runs LLMs locally on your machine.

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

## Using OpenAI

Works with OpenAI API and compatible services.

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

## Using Anthropic (Claude)

Works with Anthropic's Claude models.

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

## Using OpenAI-Compatible APIs

Works with OpenRouter, LM Studio, and other OpenAI-compatible services.

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

## Multi-Model Usage

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

## Next Steps

- [Vision Guide](vision.md) - Send images to vision-capable models
- [Streaming Guide](streaming.md) - Stream responses in real-time
- [Advanced Usage](advanced.md) - Custom clients, configuration classes
- [API Reference](api-reference.md) - Full API documentation
- [Examples Directory](../examples/) - Complete working examples
