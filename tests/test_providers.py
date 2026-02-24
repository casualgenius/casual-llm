"""
Tests for LLM client and Model implementations.
"""

import os

import pytest
from pydantic import BaseModel
from unittest.mock import AsyncMock, MagicMock, patch
from casual_llm.config import ChatOptions, ClientConfig, ModelConfig, Provider
from casual_llm.providers import OllamaClient
from casual_llm.factory import create_client, create_model
from casual_llm.model import Model
from casual_llm.messages import UserMessage, AssistantMessage, SystemMessage, StreamChunk
from casual_llm.usage import Usage


# Test Pydantic models for JSON Schema tests
class PersonInfo(BaseModel):
    """Simple Pydantic model for testing"""

    name: str
    age: int


class Address(BaseModel):
    """Nested model for testing complex schemas"""

    street: str
    city: str
    zip_code: str


class PersonWithAddress(BaseModel):
    """Pydantic model with nested structure for testing"""

    name: str
    age: int
    address: Address


# Try to import OpenAI client - may not be available
try:
    from casual_llm.providers import OpenAIClient

    OPENAI_AVAILABLE = OpenAIClient is not None
except ImportError:
    OPENAI_AVAILABLE = False


class TestOllamaClient:
    """Tests for OllamaClient and Model"""

    @pytest.fixture
    def client(self):
        """Create an OllamaClient instance for testing"""
        return OllamaClient(
            host="http://localhost:11434",
            timeout=30.0,
        )

    @pytest.fixture
    def model(self, client):
        """Create a Model using the client"""
        return Model(
            client=client,
            name="qwen2.5:7b-instruct",
            default_options=ChatOptions(temperature=0.7),
        )

    @pytest.mark.asyncio
    async def test_generate_text_success(self, model):
        """Test successful text generation"""
        mock_response = MagicMock()
        mock_response.message.content = "Hello, I'm a test response!"
        mock_response.message.tool_calls = None

        with patch("ollama.AsyncClient.chat", new=AsyncMock(return_value=mock_response)):
            messages = [
                UserMessage(content="Hello"),
            ]

            result = await model.chat(messages, ChatOptions(response_format="text"))

            assert isinstance(result, AssistantMessage)
            assert result.content == "Hello, I'm a test response!"
            assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_generate_json_success(self, model):
        """Test successful JSON generation"""
        mock_response = MagicMock()
        mock_response.message.content = '{"name": "test", "value": 42}'
        mock_response.message.tool_calls = None

        with patch("ollama.AsyncClient.chat", new=AsyncMock(return_value=mock_response)):
            messages = [
                UserMessage(content="Give me JSON"),
            ]

            result = await model.chat(messages, ChatOptions(response_format="json"))

            assert isinstance(result, AssistantMessage)
            assert '{"name": "test", "value": 42}' in result.content

    @pytest.mark.asyncio
    async def test_generate_with_conversation(self, model):
        """Test generation with multi-turn conversation"""
        mock_response = MagicMock()
        mock_response.message.content = "Got it!"
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            messages = [
                SystemMessage(content="You are a helpful assistant"),
                UserMessage(content="Hello"),
                AssistantMessage(content="Hi there!"),
                UserMessage(content="How are you?"),
            ]

            result = await model.chat(messages, ChatOptions(response_format="text"))

            assert isinstance(result, AssistantMessage)
            assert result.content == "Got it!"
            # Verify the messages were passed
            call_args = mock_chat.call_args
            assert call_args is not None

    @pytest.mark.asyncio
    async def test_generate_with_none_content(self, model):
        """Test handling of messages with None content"""
        mock_response = MagicMock()
        mock_response.message.content = "Handled!"
        mock_response.message.tool_calls = None

        with patch("ollama.AsyncClient.chat", new=AsyncMock(return_value=mock_response)):
            messages = [
                UserMessage(content=None),  # None content
                AssistantMessage(content="Response"),
            ]

            result = await model.chat(messages, ChatOptions(response_format="text"))

            assert isinstance(result, AssistantMessage)
            assert result.content == "Handled!"

    @pytest.mark.asyncio
    async def test_temperature_override(self, model):
        """Test that per-call temperature overrides model default temperature"""
        # Model was created with temperature=0.7
        assert model.default_options.temperature == 0.7

        mock_response = MagicMock()
        mock_response.message.content = "Response"
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            messages = [UserMessage(content="Test")]

            # Call with overridden temperature
            await model.chat(messages, ChatOptions(temperature=0.1))

            # Verify the temperature passed to Ollama
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["options"]["temperature"] == 0.1

            # Call without override - should use model default temperature
            await model.chat(messages)
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["options"]["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_temperature_none_not_sent(self, client):
        """Test that temperature is not sent to API when None"""
        # Create model without temperature (defaults to None)
        model = Model(
            client=client,
            name="test-model",
        )
        assert model.default_options is None

        mock_response = MagicMock()
        mock_response.message.content = "Response"
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            messages = [UserMessage(content="Test")]

            # Call without temperature - should not include it in options
            await model.chat(messages)

            # Verify temperature was NOT included in options
            call_kwargs = mock_chat.call_args.kwargs
            assert "temperature" not in call_kwargs["options"]

    @pytest.mark.asyncio
    async def test_usage_tracking(self, model):
        """Test that usage statistics are tracked per-model"""
        # Check initial state
        assert model.get_usage() is None

        mock_response = MagicMock()
        mock_response.message.content = "Response"
        mock_response.message.tool_calls = None
        # Ollama uses prompt_eval_count and eval_count
        mock_response.prompt_eval_count = 10
        mock_response.eval_count = 20

        with patch("ollama.AsyncClient.chat", new=AsyncMock(return_value=mock_response)):
            messages = [UserMessage(content="Test")]
            await model.chat(messages)

            # Verify usage was tracked
            usage = model.get_usage()
            assert usage is not None
            assert isinstance(usage, Usage)
            assert usage.prompt_tokens == 10
            assert usage.completion_tokens == 20
            assert usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_json_schema_response_format(self, model):
        """Test that Pydantic model is correctly converted to JSON Schema for Ollama"""
        mock_response = MagicMock()
        mock_response.message.content = '{"name": "Alice", "age": 30}'
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            messages = [UserMessage(content="Give me person info")]

            result = await model.chat(messages, ChatOptions(response_format=PersonInfo))

            assert isinstance(result, AssistantMessage)
            assert '{"name": "Alice", "age": 30}' in result.content

            # Verify the format parameter contains the JSON Schema
            call_kwargs = mock_chat.call_args.kwargs
            assert "format" in call_kwargs
            schema = call_kwargs["format"]

            # Verify it's a dict (JSON Schema), not a string
            assert isinstance(schema, dict)

            # Verify schema has expected properties
            assert "properties" in schema
            assert "name" in schema["properties"]
            assert "age" in schema["properties"]
            assert schema["properties"]["name"]["type"] == "string"
            assert schema["properties"]["age"]["type"] == "integer"

    @pytest.mark.asyncio
    async def test_json_schema_nested_pydantic_model(self, model):
        """Test that complex nested Pydantic models work correctly"""
        mock_response = MagicMock()
        mock_response.message.content = (
            '{"name": "Bob", "age": 25, "address": '
            '{"street": "123 Main St", "city": "NYC", "zip_code": "10001"}}'
        )
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            messages = [UserMessage(content="Give me person with address")]

            result = await model.chat(messages, ChatOptions(response_format=PersonWithAddress))

            assert isinstance(result, AssistantMessage)

            # Verify the format parameter contains the nested JSON Schema
            call_kwargs = mock_chat.call_args.kwargs
            assert "format" in call_kwargs
            schema = call_kwargs["format"]

            # Verify it's a dict with properties
            assert isinstance(schema, dict)
            assert "properties" in schema

            # Verify nested structure is present (either through $defs or inline)
            # Pydantic v2 uses $defs for nested models
            if "$defs" in schema:
                assert "Address" in schema["$defs"]

    @pytest.mark.asyncio
    async def test_backward_compat_json_format(self, model):
        """Test that existing 'json' format still works (backward compatibility)"""
        mock_response = MagicMock()
        mock_response.message.content = '{"status": "ok"}'
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            messages = [UserMessage(content="Give me JSON")]

            result = await model.chat(messages, ChatOptions(response_format="json"))

            assert isinstance(result, AssistantMessage)
            assert '{"status": "ok"}' in result.content

            # Verify format is set to "json" string (not a schema dict)
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["format"] == "json"

    @pytest.mark.asyncio
    async def test_backward_compat_text_format(self, model):
        """Test that existing 'text' format still works (backward compatibility)"""
        mock_response = MagicMock()
        mock_response.message.content = "Plain text response"
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            messages = [UserMessage(content="Give me text")]

            result = await model.chat(messages, ChatOptions(response_format="text"))

            assert isinstance(result, AssistantMessage)
            assert result.content == "Plain text response"

            # Verify no format parameter is set for text
            call_kwargs = mock_chat.call_args.kwargs
            assert "format" not in call_kwargs

    @pytest.mark.asyncio
    async def test_stream_success(self, model):
        """Test successful streaming with multiple chunks"""

        async def mock_stream():
            """Mock async generator that yields stream chunks"""
            chunks = [
                MagicMock(message=MagicMock(content="Hello"), done=False),
                MagicMock(message=MagicMock(content=" world"), done=False),
                MagicMock(message=MagicMock(content="!"), done=True),
            ]
            for chunk in chunks:
                yield chunk

        with patch("ollama.AsyncClient.chat", new=AsyncMock(return_value=mock_stream())):
            messages = [UserMessage(content="Say hello")]

            collected_chunks = []
            async for chunk in model.stream(messages):
                collected_chunks.append(chunk)

            # Verify we got the expected chunks
            assert len(collected_chunks) == 3
            assert all(isinstance(c, StreamChunk) for c in collected_chunks)
            assert collected_chunks[0].content == "Hello"
            assert collected_chunks[1].content == " world"
            assert collected_chunks[2].content == "!"

            # Verify finish_reason is set on the last chunk
            assert collected_chunks[2].finish_reason == "stop"
            assert collected_chunks[0].finish_reason is None
            assert collected_chunks[1].finish_reason is None

    @pytest.mark.asyncio
    async def test_stream_empty_chunks(self, model):
        """Test that empty chunks are handled (not yielded)"""

        async def mock_stream():
            """Mock async generator with empty chunks interspersed"""
            chunks = [
                MagicMock(message=MagicMock(content="Hello"), done=False),
                MagicMock(message=MagicMock(content=""), done=False),  # Empty content
                MagicMock(message=MagicMock(content=None), done=False),  # None content
                MagicMock(message=None, done=False),  # No message at all
                MagicMock(message=MagicMock(content=" there"), done=True),
            ]
            for chunk in chunks:
                yield chunk

        with patch("ollama.AsyncClient.chat", new=AsyncMock(return_value=mock_stream())):
            messages = [UserMessage(content="Test")]

            collected_chunks = []
            async for chunk in model.stream(messages):
                collected_chunks.append(chunk)

            # Only non-empty chunks should be yielded
            assert len(collected_chunks) == 2
            assert collected_chunks[0].content == "Hello"
            assert collected_chunks[1].content == " there"

    @pytest.mark.asyncio
    async def test_stream_temperature_override(self, model):
        """Test that per-call temperature overrides model temperature during streaming"""
        # Model was created with temperature=0.7
        assert model.default_options.temperature == 0.7

        async def mock_stream():
            """Empty mock stream for testing parameters"""
            chunks = [
                MagicMock(message=MagicMock(content="Test"), done=True),
            ]
            for chunk in chunks:
                yield chunk

        mock_chat = AsyncMock(return_value=mock_stream())

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            messages = [UserMessage(content="Test")]

            # Call with overridden temperature
            async for _ in model.stream(messages, ChatOptions(temperature=0.2)):
                pass

            # Verify the temperature passed to Ollama
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["options"]["temperature"] == 0.2
            assert call_kwargs["stream"] is True

            # Reset mock for second call
            mock_chat.reset_mock()
            mock_chat.return_value = mock_stream()

            # Call without override - should use model temperature
            async for _ in model.stream(messages):
                pass

            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["options"]["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_extra_on_chat_options(self, client):
        """Test that extra dict on ChatOptions is passed to the API"""
        model = Model(client=client, name="qwen2.5:7b-instruct")

        mock_response = MagicMock()
        mock_response.message.content = "Response"
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            messages = [UserMessage(content="Test")]
            await model.chat(messages, ChatOptions(extra={"keep_alive": "10m"}))

            # Verify extra kwargs were passed through to the API call
            call_kwargs = mock_chat.call_args.kwargs
            assert "keep_alive" in call_kwargs
            assert call_kwargs["keep_alive"] == "10m"


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI client not installed")
class TestOpenAIClient:
    """Tests for OpenAIClient and Model"""

    @pytest.fixture
    def client(self):
        """Create an OpenAIClient instance for testing"""
        return OpenAIClient(
            api_key="sk-test-key",
        )

    @pytest.fixture
    def model(self, client):
        """Create a Model using the client"""
        return Model(
            client=client,
            name="gpt-4o-mini",
            default_options=ChatOptions(temperature=0.7),
        )

    @pytest.mark.asyncio
    async def test_generate_text_success(self, model):
        """Test successful text generation"""
        mock_completion = MagicMock()
        mock_message = MagicMock(content="Hello from OpenAI!", tool_calls=None)
        # Remove tool_calls attribute entirely to match real behavior
        del mock_message.tool_calls
        mock_completion.choices = [MagicMock(message=mock_message)]

        with patch.object(
            model._client.client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_completion),
        ):
            messages = [UserMessage(content="Hello")]
            result = await model.chat(messages, ChatOptions(response_format="text"))

            assert isinstance(result, AssistantMessage)
            assert result.content == "Hello from OpenAI!"
            assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_generate_json_success(self, model):
        """Test successful JSON generation"""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content='{"status": "ok"}'))]

        with patch.object(
            model._client.client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_completion),
        ):
            messages = [UserMessage(content="Give me JSON")]
            result = await model.chat(messages, ChatOptions(response_format="json"))

            assert isinstance(result, AssistantMessage)
            assert '{"status": "ok"}' in result.content

    @pytest.mark.asyncio
    async def test_generate_with_max_tokens(self, model):
        """Test generation with max_tokens parameter"""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Short response"))]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(model._client.client.chat.completions, "create", new=mock_create):
            messages = [UserMessage(content="Test")]
            result = await model.chat(messages, ChatOptions(response_format="text", max_tokens=50))

            assert isinstance(result, AssistantMessage)
            assert result.content == "Short response"
            # Verify max_tokens was passed
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 50

    @pytest.mark.asyncio
    async def test_message_conversion(self, model):
        """Test that ChatMessages are converted correctly to OpenAI format"""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Response"))]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(model._client.client.chat.completions, "create", new=mock_create):
            messages = [
                SystemMessage(content="You are helpful"),
                UserMessage(content="Hello"),
                AssistantMessage(content="Hi!"),
            ]

            await model.chat(messages, ChatOptions(response_format="text"))

            # Verify messages were converted to dict format
            call_kwargs = mock_create.call_args.kwargs
            chat_messages = call_kwargs["messages"]

            assert len(chat_messages) == 3
            assert chat_messages[0]["role"] == "system"
            assert chat_messages[0]["content"] == "You are helpful"
            assert chat_messages[1]["role"] == "user"
            assert chat_messages[2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_temperature_override(self, model):
        """Test that per-call temperature overrides model temperature"""
        # Model was created with temperature=0.7
        assert model.default_options.temperature == 0.7

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Response"))]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(model._client.client.chat.completions, "create", new=mock_create):
            messages = [UserMessage(content="Test")]

            # Call with overridden temperature
            await model.chat(messages, ChatOptions(temperature=0.1))

            # Verify the temperature passed to OpenAI
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.1

            # Call without override - should use model temperature
            await model.chat(messages)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_temperature_none_not_sent(self, client):
        """Test that temperature is not sent to API when None"""
        # Create model without temperature (defaults to None)
        model = Model(
            client=client,
            name="gpt-4o-mini",
        )
        assert model.default_options is None

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Response"))]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(model._client.client.chat.completions, "create", new=mock_create):
            messages = [UserMessage(content="Test")]

            # Call without temperature - should not include it in request
            await model.chat(messages)

            # Verify temperature was NOT included in request
            call_kwargs = mock_create.call_args.kwargs
            assert "temperature" not in call_kwargs

    @pytest.mark.asyncio
    async def test_usage_tracking(self, model):
        """Test that usage statistics are tracked per-model"""
        # Check initial state
        assert model.get_usage() is None

        mock_completion = MagicMock()
        mock_message = MagicMock(content="Response")
        del mock_message.tool_calls
        mock_completion.choices = [MagicMock(message=mock_message)]
        # OpenAI uses usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 15
        mock_usage.completion_tokens = 25
        mock_usage.total_tokens = 40
        mock_completion.usage = mock_usage

        with patch.object(
            model._client.client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_completion),
        ):
            messages = [UserMessage(content="Test")]
            await model.chat(messages)

            # Verify usage was tracked
            usage = model.get_usage()
            assert usage is not None
            assert isinstance(usage, Usage)
            assert usage.prompt_tokens == 15
            assert usage.completion_tokens == 25
            assert usage.total_tokens == 40

    @pytest.mark.asyncio
    async def test_json_schema_response_format(self, model):
        """Test that Pydantic model is correctly converted to JSON Schema for OpenAI"""
        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(message=MagicMock(content='{"name": "Alice", "age": 30}'))
        ]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(model._client.client.chat.completions, "create", new=mock_create):
            messages = [UserMessage(content="Give me person info")]

            result = await model.chat(messages, ChatOptions(response_format=PersonInfo))

            assert isinstance(result, AssistantMessage)
            assert '{"name": "Alice", "age": 30}' in result.content

            # Verify the response_format parameter contains the JSON Schema structure
            call_kwargs = mock_create.call_args.kwargs
            assert "response_format" in call_kwargs
            response_format = call_kwargs["response_format"]

            # Verify OpenAI json_schema format structure
            assert response_format["type"] == "json_schema"
            assert "json_schema" in response_format
            assert response_format["json_schema"]["name"] == "PersonInfo"
            assert "schema" in response_format["json_schema"]

            # Verify schema has expected properties
            schema = response_format["json_schema"]["schema"]
            assert "properties" in schema
            assert "name" in schema["properties"]
            assert "age" in schema["properties"]
            assert schema["properties"]["name"]["type"] == "string"
            assert schema["properties"]["age"]["type"] == "integer"

    @pytest.mark.asyncio
    async def test_stream_success(self, model):
        """Test successful streaming with multiple chunks and usage tracking"""

        async def mock_stream():
            """Mock async generator that yields stream chunks in OpenAI format"""
            chunks = [
                MagicMock(
                    choices=[MagicMock(delta=MagicMock(content="Hello"), finish_reason=None)],
                    usage=None,
                ),
                MagicMock(
                    choices=[MagicMock(delta=MagicMock(content=" world"), finish_reason=None)],
                    usage=None,
                ),
                MagicMock(
                    choices=[MagicMock(delta=MagicMock(content="!"), finish_reason="stop")],
                    usage=None,
                ),
                # Final chunk with usage data (no choices)
                MagicMock(
                    choices=[],
                    usage=MagicMock(prompt_tokens=10, completion_tokens=3),
                ),
            ]
            for chunk in chunks:
                yield chunk

        mock_create = AsyncMock(return_value=mock_stream())

        with patch.object(model._client.client.chat.completions, "create", new=mock_create):
            messages = [UserMessage(content="Say hello")]

            collected_chunks = []
            async for chunk in model.stream(messages):
                collected_chunks.append(chunk)

            # Content chunks + final usage chunk
            content_chunks = [c for c in collected_chunks if c.content]
            assert len(content_chunks) == 3
            assert all(isinstance(c, StreamChunk) for c in collected_chunks)
            assert content_chunks[0].content == "Hello"
            assert content_chunks[1].content == " world"
            assert content_chunks[2].content == "!"

            # Verify finish_reason is set on the last content chunk
            assert content_chunks[2].finish_reason == "stop"
            assert content_chunks[0].finish_reason is None
            assert content_chunks[1].finish_reason is None

            # Verify usage was captured
            usage = model.get_usage()
            assert usage is not None
            assert usage.prompt_tokens == 10
            assert usage.completion_tokens == 3
            assert usage.total_tokens == 13

            # Verify stream=True and stream_options were passed
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["stream"] is True
            assert call_kwargs["stream_options"] == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_extra_on_chat_options(self, client):
        """Test that extra dict on ChatOptions is passed to the API"""
        model = Model(client=client, name="gpt-4o-mini")

        mock_completion = MagicMock()
        mock_message = MagicMock(content="Response")
        del mock_message.tool_calls
        mock_completion.choices = [MagicMock(message=mock_message)]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(model._client.client.chat.completions, "create", new=mock_create):
            messages = [UserMessage(content="Test")]
            await model.chat(messages, ChatOptions(extra={"logprobs": True}))

            # Verify extra kwargs were passed through to the API call
            call_kwargs = mock_create.call_args.kwargs
            assert "logprobs" in call_kwargs
            assert call_kwargs["logprobs"] is True


class TestCreateClientFactory:
    """Tests for create_client() factory function"""

    def test_create_ollama_client(self):
        """Test creating an Ollama client"""
        config = ClientConfig(
            provider=Provider.OLLAMA,
            base_url="http://localhost:11434/api/chat",
        )

        client = create_client(config)

        assert isinstance(client, OllamaClient)
        assert client.timeout == 60.0  # default timeout

    def test_create_ollama_client_with_default_url(self):
        """Test creating Ollama client with default URL"""
        config = ClientConfig(
            provider=Provider.OLLAMA,
        )

        client = create_client(config)

        assert isinstance(client, OllamaClient)
        # Client should use default host
        assert client.host == "http://localhost:11434"

    def test_create_ollama_client_with_trailing_slash(self):
        """Test that trailing slashes are handled correctly"""
        # With trailing slash
        config_with_slash = ClientConfig(
            provider=Provider.OLLAMA,
            base_url="http://localhost:11434/",
        )
        client_with_slash = create_client(config_with_slash)

        # Without trailing slash
        config_without_slash = ClientConfig(
            provider=Provider.OLLAMA,
            base_url="http://localhost:11434",
        )
        client_without_slash = create_client(config_without_slash)

        # Both should produce the same host (trailing slash removed)
        assert client_with_slash.host == "http://localhost:11434"
        assert client_without_slash.host == "http://localhost:11434"
        assert client_with_slash.host == client_without_slash.host

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI client not installed")
    def test_create_openai_client(self):
        """Test creating an OpenAI client"""
        config = ClientConfig(
            provider=Provider.OPENAI,
            api_key="sk-test-key",
        )

        client = create_client(config)

        assert isinstance(client, OpenAIClient)

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI client not installed")
    def test_create_openai_client_with_base_url(self):
        """Test creating OpenAI client with custom base URL"""
        config = ClientConfig(
            provider=Provider.OPENAI,
            api_key="sk-test-key",
            base_url="https://api.openrouter.ai/api/v1",
        )

        client = create_client(config)

        assert isinstance(client, OpenAIClient)

    def test_create_client_unsupported_type(self):
        """Test that unsupported provider string raises ValueError at config creation"""
        with pytest.raises(ValueError, match="Unknown provider"):
            ClientConfig(provider="unsupported")

    def test_create_client_with_string_provider(self):
        """Test create_client works with string provider in ClientConfig"""
        config = ClientConfig(
            provider="ollama",
            base_url="http://localhost:11434",
        )
        client = create_client(config)
        assert isinstance(client, OllamaClient)


class TestCreateModelFactory:
    """Tests for create_model() factory function"""

    def test_create_model_with_config(self):
        """Test creating a Model from config"""
        client = OllamaClient(host="http://localhost:11434")
        config = ModelConfig(
            name="llama3.1",
            default_options=ChatOptions(temperature=0.5),
        )

        model = create_model(client, config)

        assert isinstance(model, Model)
        assert model.name == "llama3.1"
        assert model.default_options.temperature == 0.5

    def test_create_model_minimal_config(self):
        """Test creating a Model with minimal config"""
        client = OllamaClient(host="http://localhost:11434")
        config = ModelConfig(name="llama3.1")

        model = create_model(client, config)

        assert isinstance(model, Model)
        assert model.name == "llama3.1"
        assert model.default_options is None


class TestMultipleModelsPerClient:
    """Tests verifying multiple models can share a client"""

    def test_multiple_models_share_client(self):
        """Test that multiple models can use the same client"""
        client = OllamaClient(host="http://localhost:11434")

        model1 = Model(client, name="llama3.1", default_options=ChatOptions(temperature=0.7))
        model2 = Model(
            client, name="qwen2.5:7b-instruct", default_options=ChatOptions(temperature=0.5)
        )

        # Both models share the same client
        assert model1._client is model2._client
        # But have different configurations
        assert model1.name != model2.name
        assert model1.default_options.temperature != model2.default_options.temperature

    @pytest.mark.asyncio
    async def test_multiple_models_independent_usage_tracking(self):
        """Test that each model tracks its own usage independently"""
        client = OllamaClient(host="http://localhost:11434")

        model1 = Model(client, name="llama3.1")
        model2 = Model(client, name="qwen2.5:7b-instruct")

        # Both should start with no usage
        assert model1.get_usage() is None
        assert model2.get_usage() is None

        # Mock response for model1
        mock_response1 = MagicMock()
        mock_response1.message.content = "Response from model1"
        mock_response1.message.tool_calls = None
        mock_response1.prompt_eval_count = 10
        mock_response1.eval_count = 20

        # Mock response for model2
        mock_response2 = MagicMock()
        mock_response2.message.content = "Response from model2"
        mock_response2.message.tool_calls = None
        mock_response2.prompt_eval_count = 30
        mock_response2.eval_count = 40

        with patch("ollama.AsyncClient.chat", new=AsyncMock(return_value=mock_response1)):
            await model1.chat([UserMessage(content="Test 1")])

        # Only model1 should have usage
        assert model1.get_usage() is not None
        assert model1.get_usage().prompt_tokens == 10
        assert model2.get_usage() is None

        with patch("ollama.AsyncClient.chat", new=AsyncMock(return_value=mock_response2)):
            await model2.chat([UserMessage(content="Test 2")])

        # Now both have usage, independently tracked
        assert model1.get_usage().prompt_tokens == 10
        assert model2.get_usage().prompt_tokens == 30


class TestClientConfigProviderCoercion:
    """Tests for ClientConfig provider string coercion"""

    def test_string_coercion_openai(self):
        """Test that string 'openai' is coerced to Provider.OPENAI"""
        config = ClientConfig(provider="openai")
        assert config.provider == Provider.OPENAI
        assert isinstance(config.provider, Provider)

    def test_string_coercion_ollama(self):
        """Test that string 'ollama' is coerced to Provider.OLLAMA"""
        config = ClientConfig(provider="ollama")
        assert config.provider == Provider.OLLAMA

    def test_string_coercion_anthropic(self):
        """Test that string 'anthropic' is coerced to Provider.ANTHROPIC"""
        config = ClientConfig(provider="anthropic")
        assert config.provider == Provider.ANTHROPIC

    def test_enum_still_works(self):
        """Test that passing Provider enum directly still works"""
        config = ClientConfig(provider=Provider.OPENAI)
        assert config.provider == Provider.OPENAI

    def test_invalid_string_raises(self):
        """Test that invalid provider string raises ValueError"""
        with pytest.raises(ValueError, match="Unknown provider"):
            ClientConfig(provider="invalid_provider")

    def test_case_insensitive(self):
        """Test that provider strings are case-insensitive"""
        config = ClientConfig(provider="OPENAI")
        assert config.provider == Provider.OPENAI

        config2 = ClientConfig(provider="Ollama")
        assert config2.provider == Provider.OLLAMA

        config3 = ClientConfig(provider="Anthropic")
        assert config3.provider == Provider.ANTHROPIC

    def test_name_field_default_none(self):
        """Test that name defaults to None"""
        config = ClientConfig(provider="openai")
        assert config.name is None

    def test_name_field_set(self):
        """Test setting name field"""
        config = ClientConfig(name="openrouter", provider="openai")
        assert config.name == "openrouter"


class TestApiKeyResolution:
    """Tests for API key resolution in create_client"""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI client not installed")
    def test_explicit_api_key_wins(self):
        """Test that explicit api_key takes priority over env var"""
        config = ClientConfig(
            name="test",
            provider="openai",
            api_key="explicit-key",
        )

        with patch.dict(os.environ, {"TEST_API_KEY": "env-key"}):
            client = create_client(config)
            assert isinstance(client, OpenAIClient)
            # The explicit key should have been used
            assert client.client.api_key == "explicit-key"

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI client not installed")
    def test_name_based_env_var_lookup(self):
        """Test that {NAME}_API_KEY env var is used when no explicit key"""
        config = ClientConfig(
            name="openrouter",
            provider="openai",
        )

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}, clear=False):
            client = create_client(config)
            assert isinstance(client, OpenAIClient)
            assert client.client.api_key == "env-key"

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI client not installed")
    def test_no_name_no_key_passes_none(self):
        """Test that None is passed when no key and no name"""
        config = ClientConfig(provider="openai")

        with patch("casual_llm.factory.OpenAIClient") as MockClient:
            create_client(config)
            call_kwargs = MockClient.call_args.kwargs
            assert call_kwargs["api_key"] is None

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI client not installed")
    def test_name_env_var_not_found_passes_none(self):
        """Test that None is passed when name is set but env var doesn't exist"""
        config = ClientConfig(
            name="nonexistent",
            provider="openai",
        )

        # Ensure the env var doesn't exist
        env = os.environ.copy()
        env.pop("NONEXISTENT_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            with patch("casual_llm.factory.OpenAIClient") as MockClient:
                create_client(config)
                call_kwargs = MockClient.call_args.kwargs
                assert call_kwargs["api_key"] is None

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI client not installed")
    def test_name_hyphen_sanitized(self):
        """Test that hyphens in name are replaced with underscores for env var"""
        config = ClientConfig(
            name="open-router",
            provider="openai",
        )

        with patch.dict(os.environ, {"OPEN_ROUTER_API_KEY": "found-key"}, clear=False):
            client = create_client(config)
            assert isinstance(client, OpenAIClient)
            assert client.client.api_key == "found-key"

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI client not installed")
    def test_name_uppercased_for_env_var(self):
        """Test that name is uppercased for env var lookup"""
        config = ClientConfig(
            name="myService",
            provider="openai",
        )

        with patch.dict(os.environ, {"MYSERVICE_API_KEY": "found-key"}, clear=False):
            client = create_client(config)
            assert isinstance(client, OpenAIClient)
            assert client.client.api_key == "found-key"

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI client not installed")
    def test_name_dot_sanitized(self):
        """Test that dots in name are replaced with underscores for env var"""
        config = ClientConfig(
            name="my.service",
            provider="openai",
        )

        with patch.dict(os.environ, {"MY_SERVICE_API_KEY": "dot-key"}, clear=False):
            client = create_client(config)
            assert isinstance(client, OpenAIClient)
            assert client.client.api_key == "dot-key"
