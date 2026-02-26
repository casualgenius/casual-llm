"""
Tests for Anthropic LLM client implementation.
"""

import pytest
from pydantic import BaseModel
from unittest.mock import AsyncMock, MagicMock, patch
from casual_llm.config import ChatOptions, ClientConfig, ModelConfig, Provider
from casual_llm.factory import create_client, create_model
from casual_llm.model import Model
from casual_llm.messages import UserMessage, AssistantMessage, SystemMessage, StreamChunk
from casual_llm.usage import Usage
from casual_llm.tools import Tool, ToolParameter


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


# Try to import Anthropic client - may not be available
try:
    from casual_llm.providers import AnthropicClient

    ANTHROPIC_AVAILABLE = AnthropicClient is not None
except ImportError:
    ANTHROPIC_AVAILABLE = False


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic client not installed")
class TestAnthropicClient:
    """Tests for AnthropicClient and Model"""

    @pytest.fixture
    def client(self):
        """Create an AnthropicClient instance for testing"""
        return AnthropicClient(
            api_key="sk-ant-test-key",
        )

    @pytest.fixture
    def model(self, client):
        """Create a Model using the client"""
        return Model(
            client=client,
            name="claude-3-haiku-20240307",
            default_options=ChatOptions(temperature=0.7),
        )

    def _create_mock_response(
        self,
        content: str = "Hello from Claude!",
        tool_use_blocks: list | None = None,
        input_tokens: int = 10,
        output_tokens: int = 20,
    ):
        """Helper to create a mock Anthropic API response"""
        mock_response = MagicMock()

        # Create content blocks
        content_blocks = []
        if content:
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = content
            content_blocks.append(text_block)

        if tool_use_blocks:
            content_blocks.extend(tool_use_blocks)

        mock_response.content = content_blocks

        # Create usage object
        mock_usage = MagicMock()
        mock_usage.input_tokens = input_tokens
        mock_usage.output_tokens = output_tokens
        mock_response.usage = mock_usage

        return mock_response

    @pytest.mark.asyncio
    async def test_generate_text_success(self, model):
        """Test successful text generation"""
        mock_response = self._create_mock_response(content="Hello from Claude!")

        with patch.object(
            model._client.client.messages, "create", new=AsyncMock(return_value=mock_response)
        ):
            messages = [
                UserMessage(content="Hello"),
            ]

            result = await model.chat(messages, ChatOptions(response_format="text"))

            assert isinstance(result, AssistantMessage)
            assert result.content == "Hello from Claude!"
            assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_generate_json_success(self, model):
        """Test successful JSON generation"""
        mock_response = self._create_mock_response(content='{"name": "test", "value": 42}')

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            messages = [
                UserMessage(content="Give me JSON"),
            ]

            result = await model.chat(messages, ChatOptions(response_format="json"))

            assert isinstance(result, AssistantMessage)
            assert '{"name": "test", "value": 42}' in result.content

            # Verify JSON instruction was added to system prompt
            call_kwargs = mock_create.call_args.kwargs
            assert "system" in call_kwargs
            system_blocks = call_kwargs["system"]
            assert any("JSON" in b["text"] for b in system_blocks)

    @pytest.mark.asyncio
    async def test_generate_with_conversation(self, model):
        """Test generation with multi-turn conversation"""
        mock_response = self._create_mock_response(content="Got it!")

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            messages = [
                SystemMessage(content="You are a helpful assistant"),
                UserMessage(content="Hello"),
                AssistantMessage(content="Hi there!"),
                UserMessage(content="How are you?"),
            ]

            result = await model.chat(messages, ChatOptions(response_format="text"))

            assert isinstance(result, AssistantMessage)
            assert result.content == "Got it!"

            # Verify system message was extracted and passed separately as content blocks
            call_kwargs = mock_create.call_args.kwargs
            assert "system" in call_kwargs
            system_blocks = call_kwargs["system"]
            assert any("helpful assistant" in b["text"] for b in system_blocks)

            # Verify messages were passed (excluding system message)
            assert "messages" in call_kwargs
            assert call_kwargs["messages"] is not None

    @pytest.mark.asyncio
    async def test_generate_with_none_content(self, model):
        """Test handling of messages with None content"""
        mock_response = self._create_mock_response(content="Handled!")

        with patch.object(
            model._client.client.messages, "create", new=AsyncMock(return_value=mock_response)
        ):
            messages = [
                UserMessage(content=None),  # None content
                AssistantMessage(content="Response"),
            ]

            result = await model.chat(messages, ChatOptions(response_format="text"))

            assert isinstance(result, AssistantMessage)
            assert result.content == "Handled!"

    @pytest.mark.asyncio
    async def test_temperature_override(self, model):
        """Test that per-call temperature overrides model temperature"""
        # Model was created with temperature=0.7
        assert model.default_options.temperature == 0.7

        mock_response = self._create_mock_response(content="Response")
        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            messages = [UserMessage(content="Test")]

            # Call with overridden temperature
            await model.chat(messages, ChatOptions(temperature=0.1))

            # Verify the temperature passed to Anthropic
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
            name="claude-3-haiku-20240307",
        )
        assert model.default_options is None

        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Response"
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
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

        mock_response = self._create_mock_response(
            content="Response",
            input_tokens=15,
            output_tokens=25,
        )

        with patch.object(
            model._client.client.messages, "create", new=AsyncMock(return_value=mock_response)
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
        """Test that Pydantic model is correctly converted to JSON Schema for Anthropic"""
        mock_response = self._create_mock_response(content='{"name": "Alice", "age": 30}')

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            messages = [UserMessage(content="Give me person info")]

            result = await model.chat(messages, ChatOptions(response_format=PersonInfo))

            assert isinstance(result, AssistantMessage)
            assert '{"name": "Alice", "age": 30}' in result.content

            # Verify the system parameter contains the JSON Schema instruction
            call_kwargs = mock_create.call_args.kwargs
            assert "system" in call_kwargs
            system_blocks = call_kwargs["system"]
            system_text = " ".join(b["text"] for b in system_blocks)

            # Verify schema details are included
            assert "JSON" in system_text
            assert "schema" in system_text.lower()

    @pytest.mark.asyncio
    async def test_json_schema_nested_pydantic_model(self, model):
        """Test that complex nested Pydantic models work correctly"""
        mock_response = self._create_mock_response(
            content=(
                '{"name": "Bob", "age": 25, "address": '
                '{"street": "123 Main St", "city": "NYC", "zip_code": "10001"}}'
            )
        )

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            messages = [UserMessage(content="Give me person with address")]

            result = await model.chat(messages, ChatOptions(response_format=PersonWithAddress))

            assert isinstance(result, AssistantMessage)

            # Verify the system parameter contains the nested JSON Schema
            call_kwargs = mock_create.call_args.kwargs
            assert "system" in call_kwargs
            system_blocks = call_kwargs["system"]
            system_text = " ".join(b["text"] for b in system_blocks)

            # Verify schema and JSON instructions are present
            assert "JSON" in system_text

    @pytest.mark.asyncio
    async def test_backward_compat_json_format(self, model):
        """Test that existing 'json' format still works (backward compatibility)"""
        mock_response = self._create_mock_response(content='{"status": "ok"}')

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            messages = [UserMessage(content="Give me JSON")]

            result = await model.chat(messages, ChatOptions(response_format="json"))

            assert isinstance(result, AssistantMessage)
            assert '{"status": "ok"}' in result.content

            # Verify JSON instruction was added to system
            call_kwargs = mock_create.call_args.kwargs
            assert "system" in call_kwargs
            system_blocks = call_kwargs["system"]
            assert any("JSON" in b["text"] for b in system_blocks)

    @pytest.mark.asyncio
    async def test_backward_compat_text_format(self, model):
        """Test that existing 'text' format still works (backward compatibility)"""
        mock_response = self._create_mock_response(content="Plain text response")

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            messages = [UserMessage(content="Give me text")]

            result = await model.chat(messages, ChatOptions(response_format="text"))

            assert isinstance(result, AssistantMessage)
            assert result.content == "Plain text response"

            # Verify no JSON-related system parameter is set for text
            call_kwargs = mock_create.call_args.kwargs
            # System should not contain JSON instructions for text format
            if "system" in call_kwargs:
                system_blocks = call_kwargs["system"]
                assert not any("JSON" in b["text"] for b in system_blocks)

    @pytest.mark.asyncio
    async def test_max_tokens_passed(self, model):
        """Test that max_tokens is passed to the API"""
        mock_response = self._create_mock_response(content="Short response")

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            messages = [UserMessage(content="Test")]
            result = await model.chat(messages, ChatOptions(response_format="text", max_tokens=100))

            assert isinstance(result, AssistantMessage)
            assert result.content == "Short response"

            # Verify max_tokens was passed
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_default_max_tokens(self, model):
        """Test that default max_tokens is used when not specified"""
        mock_response = self._create_mock_response(content="Response")

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            messages = [UserMessage(content="Test")]
            await model.chat(messages)

            # Verify default max_tokens was passed (4096)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_stream_success(self, model):
        """Test successful streaming with multiple chunks and usage tracking"""

        class MockStreamManager:
            """Mock context manager for streaming"""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def __aiter__(self):
                # Yield content block delta events
                events = [
                    MagicMock(
                        type="content_block_delta",
                        delta=MagicMock(text="Hello"),
                    ),
                    MagicMock(
                        type="content_block_delta",
                        delta=MagicMock(text=" world"),
                    ),
                    MagicMock(
                        type="content_block_delta",
                        delta=MagicMock(text="!"),
                    ),
                    MagicMock(type="message_stop"),
                ]
                for event in events:
                    yield event

            async def get_final_message(self):
                return MagicMock(
                    usage=MagicMock(input_tokens=15, output_tokens=5),
                )

        with patch.object(
            model._client.client.messages, "stream", return_value=MockStreamManager()
        ):
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

            # Verify usage was captured
            usage = model.get_usage()
            assert usage is not None
            assert usage.prompt_tokens == 15
            assert usage.completion_tokens == 5
            assert usage.total_tokens == 20

    @pytest.mark.asyncio
    async def test_stream_empty_chunks(self, model):
        """Test that empty chunks are handled correctly"""

        class MockStreamManager:
            """Mock context manager for streaming with empty events"""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def __aiter__(self):
                events = [
                    MagicMock(
                        type="content_block_delta",
                        delta=MagicMock(text="Hello"),
                    ),
                    # Event without text attribute
                    MagicMock(
                        type="content_block_delta",
                        delta=MagicMock(spec=[]),  # No text attribute
                    ),
                    MagicMock(
                        type="content_block_delta",
                        delta=MagicMock(text=" there"),
                    ),
                    MagicMock(type="message_stop"),
                ]
                for event in events:
                    yield event

            async def get_final_message(self):
                return MagicMock(
                    usage=MagicMock(input_tokens=8, output_tokens=2),
                )

        with patch.object(
            model._client.client.messages, "stream", return_value=MockStreamManager()
        ):
            messages = [UserMessage(content="Test")]

            collected_chunks = []
            async for chunk in model.stream(messages):
                collected_chunks.append(chunk)

            # Content chunks + final usage chunk
            content_chunks = [c for c in collected_chunks if c.content]
            assert len(content_chunks) == 2
            assert content_chunks[0].content == "Hello"
            assert content_chunks[1].content == " there"

    @pytest.mark.asyncio
    async def test_stream_temperature_override(self, model):
        """Test that per-call temperature overrides model temperature during streaming"""
        # Model was created with temperature=0.7
        assert model.default_options.temperature == 0.7

        class MockStreamManager:
            """Mock context manager for streaming"""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def __aiter__(self):
                events = [
                    MagicMock(
                        type="content_block_delta",
                        delta=MagicMock(text="Test"),
                    ),
                    MagicMock(type="message_stop"),
                ]
                for event in events:
                    yield event

            async def get_final_message(self):
                return MagicMock(
                    usage=MagicMock(input_tokens=5, output_tokens=1),
                )

        mock_stream = MagicMock(return_value=MockStreamManager())

        with patch.object(model._client.client.messages, "stream", mock_stream):
            messages = [UserMessage(content="Test")]

            # Call with overridden temperature
            async for _ in model.stream(messages, ChatOptions(temperature=0.2)):
                pass

            # Verify the temperature passed to Anthropic
            call_kwargs = mock_stream.call_args.kwargs
            assert call_kwargs["temperature"] == 0.2

            # Reset mock for second call
            mock_stream.reset_mock()
            mock_stream.return_value = MockStreamManager()

            # Call without override - should use model temperature
            async for _ in model.stream(messages):
                pass

            call_kwargs = mock_stream.call_args.kwargs
            assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_tool_calls(self, model):
        """Test that tool calls are correctly parsed from response"""
        # Create a tool use block
        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "tool_call_123"
        tool_use_block.name = "get_weather"
        tool_use_block.input = {"location": "San Francisco"}

        mock_response = MagicMock()

        # Create text block
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Let me check the weather."

        mock_response.content = [text_block, tool_use_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            # Define a test tool
            test_tool = Tool(
                name="get_weather",
                description="Get weather for a location",
                parameters={
                    "location": ToolParameter(
                        type="string",
                        description="City name to get weather for",
                    ),
                },
                required=["location"],
            )

            messages = [UserMessage(content="What's the weather in San Francisco?")]

            result = await model.chat(messages, ChatOptions(tools=[test_tool]))

            assert isinstance(result, AssistantMessage)
            assert result.content == "Let me check the weather."
            assert result.tool_calls is not None
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].function.name == "get_weather"
            assert result.tool_calls[0].id == "tool_call_123"

    @pytest.mark.asyncio
    async def test_tools_passed_to_api(self, model):
        """Test that tools are correctly converted and passed to the API"""
        mock_response = self._create_mock_response(content="I can help with that.")

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            # Define a test tool
            test_tool = Tool(
                name="calculate",
                description="Calculate the sum of two numbers",
                parameters={
                    "a": ToolParameter(type="integer", description="First number"),
                    "b": ToolParameter(type="integer", description="Second number"),
                },
                required=["a", "b"],
            )

            messages = [UserMessage(content="Add 2 + 3")]

            await model.chat(messages, ChatOptions(tools=[test_tool]))

            # Verify tools were passed
            call_kwargs = mock_create.call_args.kwargs
            assert "tools" in call_kwargs
            assert len(call_kwargs["tools"]) == 1
            assert call_kwargs["tools"][0]["name"] == "calculate"

    @pytest.mark.asyncio
    async def test_system_message_combined_with_json_format(self, model):
        """Test that system message is preserved when JSON format is used"""
        mock_response = self._create_mock_response(content='{"result": "test"}')

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content="Give me JSON"),
            ]

            await model.chat(messages, ChatOptions(response_format="json"))

            # Verify both system message and JSON instruction are in system param
            call_kwargs = mock_create.call_args.kwargs
            assert "system" in call_kwargs
            system_blocks = call_kwargs["system"]
            system_text = " ".join(b["text"] for b in system_blocks)
            assert "helpful assistant" in system_text
            assert "JSON" in system_text

    @pytest.mark.asyncio
    async def test_extra_on_chat_options(self, client):
        """Test that extra dict on ChatOptions is passed to the API"""
        model = Model(client=client, name="claude-3-haiku-20240307")

        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Response"
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            messages = [UserMessage(content="Test")]
            await model.chat(messages, ChatOptions(extra={"metadata": {"user_id": "test123"}}))

            # Verify extra kwargs were passed
            call_kwargs = mock_create.call_args.kwargs
            assert "metadata" in call_kwargs
            assert call_kwargs["metadata"]["user_id"] == "test123"


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic client not installed")
class TestCreateAnthropicClientFactory:
    """Tests for create_client() factory function with Anthropic"""

    def test_create_anthropic_client(self):
        """Test creating an Anthropic client"""
        config = ClientConfig(
            provider=Provider.ANTHROPIC,
            api_key="sk-ant-test-key",
        )

        client = create_client(config)

        assert isinstance(client, AnthropicClient)

    def test_create_anthropic_client_with_base_url(self):
        """Test creating Anthropic client with custom base URL"""
        config = ClientConfig(
            provider=Provider.ANTHROPIC,
            api_key="sk-ant-test-key",
            base_url="https://custom.anthropic.endpoint/v1",
        )

        client = create_client(config)

        assert isinstance(client, AnthropicClient)

    def test_create_model_with_config(self):
        """Test creating a Model from config"""
        client = AnthropicClient(api_key="sk-ant-test-key")
        config = ModelConfig(
            name="claude-3-sonnet-20240229",
            default_options=ChatOptions(temperature=0.5),
        )

        model = create_model(client, config)

        assert isinstance(model, Model)
        assert model.name == "claude-3-sonnet-20240229"
        assert model.default_options.temperature == 0.5
