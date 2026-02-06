# Advanced Usage

## Custom Clients

Implement the `LLMClient` protocol to add your own provider:

```python
from typing import Literal, AsyncIterator
from pydantic import BaseModel
from casual_llm import (
    LLMClient,
    Model,
    ChatMessage,
    AssistantMessage,
    StreamChunk,
    Tool,
    Usage,
)

class MyCustomClient:
    """Custom LLM client implementation."""

    async def _chat(
        self,
        model: str,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> tuple[AssistantMessage, Usage | None]:
        # Your implementation here
        # 1. Convert messages using message_converters
        # 2. Convert tools using tool_converters (if tools provided)
        # 3. Handle vision content (ImageContent) if present
        # 4. Call your LLM API
        # 5. Parse response including tool_calls if present
        # 6. Return (AssistantMessage, Usage)
        ...

    def _stream(
        self,
        model: str,
        messages: list[ChatMessage],
        response_format: Literal["json", "text"] | type[BaseModel] = "text",
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        # Your streaming implementation here
        ...
        yield StreamChunk(content="chunk", finish_reason=None)

# Use it with Model class
client = MyCustomClient(...)
model = Model(client, name="my-model", temperature=0.7)
response = await model.chat(messages)
print(response.content)
```

## Using Configuration Classes

For more structured configuration, use `ClientConfig` and `ModelConfig`:

```python
from casual_llm import (
    create_client,
    create_model,
    ClientConfig,
    ModelConfig,
    Provider,
    UserMessage,
)

# Client config (connection settings)
client_config = ClientConfig(
    provider=Provider.OPENAI,
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
)

# Model config (model settings)
model_config = ModelConfig(
    name="gpt-4",
    temperature=0.7,
)

# Create client and model using factory functions
client = create_client(client_config)
model = create_model(client, model_config)

# Use as normal
response = await model.chat([UserMessage(content="Hello!")])
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
