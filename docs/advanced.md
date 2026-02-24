# Advanced Usage

## Custom Clients

Implement the `LLMClient` protocol to add your own provider:

```python
from typing import AsyncIterator
from casual_llm import (
    LLMClient,
    Model,
    ChatOptions,
    ChatMessage,
    AssistantMessage,
    StreamChunk,
    Usage,
)

class MyCustomClient:
    """Custom LLM client implementation."""

    async def _chat(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> tuple[AssistantMessage, Usage | None]:
        # Your implementation here
        # 1. Convert messages using message_converters
        # 2. Read options.temperature, options.max_tokens, etc.
        # 3. Call your LLM API
        # 4. Parse response including tool_calls if present
        # 5. Return (AssistantMessage, Usage)
        ...

    async def _stream(
        self,
        model: str,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> AsyncIterator[StreamChunk]:
        # Your streaming implementation here
        ...
        yield StreamChunk(content="chunk", finish_reason=None)

# Use it with Model class
client = MyCustomClient(...)
model = Model(client, name="my-model", default_options=ChatOptions(temperature=0.7))
response = await model.chat(messages)
print(response.content)
```

## Using Configuration Classes

For more structured configuration (e.g., loading from JSON/YAML configs):

```python
from casual_llm import (
    create_client,
    create_model,
    ClientConfig,
    ModelConfig,
    ChatOptions,
    Provider,
    UserMessage,
)

# Client config (connection settings)
client_config = ClientConfig(
    provider=Provider.OPENAI,  # or just "openai"
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
)

# Model config (model settings)
model_config = ModelConfig(
    name="gpt-4",
    default_options=ChatOptions(temperature=0.7),
)

# Create client and model using factory functions
client = create_client(client_config)
model = create_model(client, model_config)

# Use as normal
response = await model.chat([UserMessage(content="Hello!")])
```

### Automatic API Key Resolution

Use `ClientConfig.name` for automatic API key lookup from environment variables:

```python
# Will check OPENROUTER_API_KEY env var automatically
config = ClientConfig(
    name="openrouter",
    provider="openai",
    base_url="https://openrouter.ai/api/v1",
)
client = create_client(config)
```

## ChatOptions Presets and Overrides

Define reusable option presets and override them per-call:

```python
from casual_llm import OpenAIClient, Model, ChatOptions, UserMessage

client = OpenAIClient(api_key="sk-...")

# Model with default options
model = Model(
    client,
    name="gpt-4",
    default_options=ChatOptions(temperature=0.7, max_tokens=1000),
)

# Uses defaults (temperature=0.7, max_tokens=1000)
response = await model.chat([UserMessage(content="Hello!")])

# Override temperature for this call only
response = await model.chat(
    [UserMessage(content="Be creative!")],
    ChatOptions(temperature=0.95),
)
# Result: temperature=0.95, max_tokens=1000 (inherited from defaults)

# Provider-specific options via extra
response = await model.chat(
    [UserMessage(content="Explain this")],
    ChatOptions(extra={"logprobs": True, "top_logprobs": 5}),
)
```

## Structured Output with Pydantic

Use Pydantic models for validated structured output:

```python
from pydantic import BaseModel
from casual_llm import ChatOptions, UserMessage

class Person(BaseModel):
    name: str
    age: int
    occupation: str

response = await model.chat(
    [UserMessage(content="Tell me about a software engineer")],
    ChatOptions(response_format=Person),
)
print(response.content)  # Valid JSON matching Person schema
```

## Per-Model Usage Tracking

Each `Model` instance tracks its own token usage:

```python
from casual_llm import OpenAIClient, Model, UserMessage

client = OpenAIClient(api_key="sk-...")

# Create multiple models
gpt4 = Model(client, name="gpt-4")
gpt35 = Model(client, name="gpt-3.5-turbo")

# Make calls with each model
await gpt4.chat([UserMessage(content="Hello!")])
await gpt35.chat([UserMessage(content="Hello!")])
await gpt4.chat([UserMessage(content="How are you?")])

# Each model tracks its own usage (most recent call)
gpt4_usage = gpt4.get_usage()
gpt35_usage = gpt35.get_usage()

print(f"GPT-4 last call: {gpt4_usage.total_tokens} tokens")
print(f"GPT-3.5 last call: {gpt35_usage.total_tokens} tokens")
```

## Next Steps

- [API Reference](api-reference.md) - Full API documentation
- [Quick Start Guide](quick-start.md) - Provider setup
- [Security](security.md) - Security considerations
