"""
Tests for Anthropic LLM provider implementation.
"""

import pytest
from pydantic import BaseModel
from unittest.mock import AsyncMock, MagicMock, patch
from casual_llm.config import ModelConfig, Provider
from casual_llm.providers import create_provider
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


# Try to import Anthropic provider - may not be available
try:
    from casual_llm.providers import AnthropicProvider

    ANTHROPIC_AVAILABLE = AnthropicProvider is not None
except ImportError:
    ANTHROPIC_AVAILABLE = False


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic provider not installed")
class TestAnthropicProvider:
    """Tests for AnthropicProvider"""

    @pytest.fixture
    def provider(self):
        """Create an AnthropicProvider instance for testing"""
        return AnthropicProvider(
            model="claude-3-haiku-20240307",
            api_key="sk-ant-test-key",
            temperature=0.7,
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
    async def test_generate_text_success(self, provider):
        """Test successful text generation"""
        mock_response = self._create_mock_response(content="Hello from Claude!")

        with patch.object(
            provider.client.messages, "create", new=AsyncMock(return_value=mock_response)
        ):
            messages = [
                UserMessage(content="Hello"),
            ]

            result = await provider.chat(messages, response_format="text")

            assert isinstance(result, AssistantMessage)
            assert result.content == "Hello from Claude!"
            assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_generate_json_success(self, provider):
        """Test successful JSON generation"""
        mock_response = self._create_mock_response(content='{"name": "test", "value": 42}')

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(provider.client.messages, "create", new=mock_create):
            messages = [
                UserMessage(content="Give me JSON"),
            ]

            result = await provider.chat(messages, response_format="json")

            assert isinstance(result, AssistantMessage)
            assert '{"name": "test", "value": 42}' in result.content

            # Verify JSON instruction was added to system prompt
            call_kwargs = mock_create.call_args.kwargs
            assert "system" in call_kwargs
            assert "JSON" in call_kwargs["system"]

    @pytest.mark.asyncio
    async def test_generate_with_conversation(self, provider):
        """Test generation with multi-turn conversation"""
        mock_response = self._create_mock_response(content="Got it!")

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(provider.client.messages, "create", new=mock_create):
            messages = [
                SystemMessage(content="You are a helpful assistant"),
                UserMessage(content="Hello"),
                AssistantMessage(content="Hi there!"),
                UserMessage(content="How are you?"),
            ]

            result = await provider.chat(messages, response_format="text")

            assert isinstance(result, AssistantMessage)
            assert result.content == "Got it!"

            # Verify system message was extracted and passed separately
            call_kwargs = mock_create.call_args.kwargs
            assert "system" in call_kwargs
            assert "helpful assistant" in call_kwargs["system"]

            # Verify messages were passed (excluding system message)
            assert "messages" in call_kwargs
            assert call_kwargs["messages"] is not None

    @pytest.mark.asyncio
    async def test_generate_with_none_content(self, provider):
        """Test handling of messages with None content"""
        mock_response = self._create_mock_response(content="Handled!")

        with patch.object(
            provider.client.messages, "create", new=AsyncMock(return_value=mock_response)
        ):
            messages = [
                UserMessage(content=None),  # None content
                AssistantMessage(content="Response"),
            ]

            result = await provider.chat(messages, response_format="text")

            assert isinstance(result, AssistantMessage)
            assert result.content == "Handled!"

    @pytest.mark.asyncio
    async def test_temperature_override(self, provider):
        """Test that per-call temperature overrides instance temperature"""
        # Provider was created with temperature=0.7
        assert provider.temperature == 0.7

        mock_response = self._create_mock_response(content="Response")
        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(provider.client.messages, "create", new=mock_create):
            messages = [UserMessage(content="Test")]

            # Call with overridden temperature
            await provider.chat(messages, temperature=0.1)

            # Verify the temperature passed to Anthropic
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.1

            # Call without override - should use instance temperature
            await provider.chat(messages)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_temperature_none_not_sent(self):
        """Test that temperature is not sent to API when None"""
        # Create provider without temperature (defaults to None)
        provider = AnthropicProvider(
            model="claude-3-haiku-20240307",
            api_key="sk-ant-test-key",
        )
        assert provider.temperature is None

        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Response"
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(provider.client.messages, "create", new=mock_create):
            messages = [UserMessage(content="Test")]

            # Call without temperature - should not include it in request
            await provider.chat(messages)

            # Verify temperature was NOT included in request
            call_kwargs = mock_create.call_args.kwargs
            assert "temperature" not in call_kwargs

    @pytest.mark.asyncio
    async def test_usage_tracking(self, provider):
        """Test that usage statistics are tracked"""
        # Check initial state
        assert provider.get_usage() is None

        mock_response = self._create_mock_response(
            content="Response",
            input_tokens=15,
            output_tokens=25,
        )

        with patch.object(
            provider.client.messages, "create", new=AsyncMock(return_value=mock_response)
        ):
            messages = [UserMessage(content="Test")]
            await provider.chat(messages)

            # Verify usage was tracked
            usage = provider.get_usage()
            assert usage is not None
            assert isinstance(usage, Usage)
            assert usage.prompt_tokens == 15
            assert usage.completion_tokens == 25
            assert usage.total_tokens == 40

    @pytest.mark.asyncio
    async def test_json_schema_response_format(self, provider):
        """Test that Pydantic model is correctly converted to JSON Schema for Anthropic"""
        mock_response = self._create_mock_response(content='{"name": "Alice", "age": 30}')

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(provider.client.messages, "create", new=mock_create):
            messages = [UserMessage(content="Give me person info")]

            result = await provider.chat(messages, response_format=PersonInfo)

            assert isinstance(result, AssistantMessage)
            assert '{"name": "Alice", "age": 30}' in result.content

            # Verify the system parameter contains the JSON Schema instruction
            call_kwargs = mock_create.call_args.kwargs
            assert "system" in call_kwargs
            system_prompt = call_kwargs["system"]

            # Verify schema details are included
            assert "JSON" in system_prompt
            assert "schema" in system_prompt.lower()

    @pytest.mark.asyncio
    async def test_json_schema_nested_pydantic_model(self, provider):
        """Test that complex nested Pydantic models work correctly"""
        mock_response = self._create_mock_response(
            content=(
                '{"name": "Bob", "age": 25, "address": '
                '{"street": "123 Main St", "city": "NYC", "zip_code": "10001"}}'
            )
        )

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(provider.client.messages, "create", new=mock_create):
            messages = [UserMessage(content="Give me person with address")]

            result = await provider.chat(messages, response_format=PersonWithAddress)

            assert isinstance(result, AssistantMessage)

            # Verify the system parameter contains the nested JSON Schema
            call_kwargs = mock_create.call_args.kwargs
            assert "system" in call_kwargs
            system_prompt = call_kwargs["system"]

            # Verify schema and JSON instructions are present
            assert "JSON" in system_prompt

    @pytest.mark.asyncio
    async def test_backward_compat_json_format(self, provider):
        """Test that existing 'json' format still works (backward compatibility)"""
        mock_response = self._create_mock_response(content='{"status": "ok"}')

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(provider.client.messages, "create", new=mock_create):
            messages = [UserMessage(content="Give me JSON")]

            result = await provider.chat(messages, response_format="json")

            assert isinstance(result, AssistantMessage)
            assert '{"status": "ok"}' in result.content

            # Verify JSON instruction was added to system
            call_kwargs = mock_create.call_args.kwargs
            assert "system" in call_kwargs
            assert "JSON" in call_kwargs["system"]

    @pytest.mark.asyncio
    async def test_backward_compat_text_format(self, provider):
        """Test that existing 'text' format still works (backward compatibility)"""
        mock_response = self._create_mock_response(content="Plain text response")

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(provider.client.messages, "create", new=mock_create):
            messages = [UserMessage(content="Give me text")]

            result = await provider.chat(messages, response_format="text")

            assert isinstance(result, AssistantMessage)
            assert result.content == "Plain text response"

            # Verify no JSON-related system parameter is set for text
            call_kwargs = mock_create.call_args.kwargs
            # System should not contain JSON instructions for text format
            if "system" in call_kwargs:
                assert "JSON" not in call_kwargs["system"]

    @pytest.mark.asyncio
    async def test_max_tokens_passed(self, provider):
        """Test that max_tokens is passed to the API"""
        mock_response = self._create_mock_response(content="Short response")

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(provider.client.messages, "create", new=mock_create):
            messages = [UserMessage(content="Test")]
            result = await provider.chat(messages, response_format="text", max_tokens=100)

            assert isinstance(result, AssistantMessage)
            assert result.content == "Short response"

            # Verify max_tokens was passed
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_default_max_tokens(self, provider):
        """Test that default max_tokens is used when not specified"""
        mock_response = self._create_mock_response(content="Response")

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(provider.client.messages, "create", new=mock_create):
            messages = [UserMessage(content="Test")]
            await provider.chat(messages)

            # Verify default max_tokens was passed (4096)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_stream_success(self, provider):
        """Test successful streaming with multiple chunks"""

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

        with patch.object(provider.client.messages, "stream", return_value=MockStreamManager()):
            messages = [UserMessage(content="Say hello")]

            collected_chunks = []
            async for chunk in provider.stream(messages):
                collected_chunks.append(chunk)

            # Verify we got the expected chunks
            assert len(collected_chunks) == 3
            assert all(isinstance(c, StreamChunk) for c in collected_chunks)
            assert collected_chunks[0].content == "Hello"
            assert collected_chunks[1].content == " world"
            assert collected_chunks[2].content == "!"

    @pytest.mark.asyncio
    async def test_stream_empty_chunks(self, provider):
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

        with patch.object(provider.client.messages, "stream", return_value=MockStreamManager()):
            messages = [UserMessage(content="Test")]

            collected_chunks = []
            async for chunk in provider.stream(messages):
                collected_chunks.append(chunk)

            # Only chunks with text content should be yielded
            assert len(collected_chunks) == 2
            assert collected_chunks[0].content == "Hello"
            assert collected_chunks[1].content == " there"

    @pytest.mark.asyncio
    async def test_stream_temperature_override(self, provider):
        """Test that per-call temperature overrides instance temperature during streaming"""
        # Provider was created with temperature=0.7
        assert provider.temperature == 0.7

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

        mock_stream = MagicMock(return_value=MockStreamManager())

        with patch.object(provider.client.messages, "stream", mock_stream):
            messages = [UserMessage(content="Test")]

            # Call with overridden temperature
            async for _ in provider.stream(messages, temperature=0.2):
                pass

            # Verify the temperature passed to Anthropic
            call_kwargs = mock_stream.call_args.kwargs
            assert call_kwargs["temperature"] == 0.2

            # Reset mock for second call
            mock_stream.reset_mock()
            mock_stream.return_value = MockStreamManager()

            # Call without override - should use instance temperature
            async for _ in provider.stream(messages):
                pass

            call_kwargs = mock_stream.call_args.kwargs
            assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_tool_calls(self, provider):
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

        with patch.object(provider.client.messages, "create", new=mock_create):
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

            result = await provider.chat(messages, tools=[test_tool])

            assert isinstance(result, AssistantMessage)
            assert result.content == "Let me check the weather."
            assert result.tool_calls is not None
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].function.name == "get_weather"
            assert result.tool_calls[0].id == "tool_call_123"

    @pytest.mark.asyncio
    async def test_tools_passed_to_api(self, provider):
        """Test that tools are correctly converted and passed to the API"""
        mock_response = self._create_mock_response(content="I can help with that.")

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(provider.client.messages, "create", new=mock_create):
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

            await provider.chat(messages, tools=[test_tool])

            # Verify tools were passed
            call_kwargs = mock_create.call_args.kwargs
            assert "tools" in call_kwargs
            assert len(call_kwargs["tools"]) == 1
            assert call_kwargs["tools"][0]["name"] == "calculate"

    @pytest.mark.asyncio
    async def test_system_message_combined_with_json_format(self, provider):
        """Test that system message is preserved when JSON format is used"""
        mock_response = self._create_mock_response(content='{"result": "test"}')

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(provider.client.messages, "create", new=mock_create):
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content="Give me JSON"),
            ]

            await provider.chat(messages, response_format="json")

            # Verify both system message and JSON instruction are in system param
            call_kwargs = mock_create.call_args.kwargs
            assert "system" in call_kwargs
            system_prompt = call_kwargs["system"]
            assert "helpful assistant" in system_prompt
            assert "JSON" in system_prompt

    @pytest.mark.asyncio
    async def test_extra_kwargs(self):
        """Test that extra_kwargs are passed to the API"""
        provider = AnthropicProvider(
            model="claude-3-haiku-20240307",
            api_key="sk-ant-test-key",
            extra_kwargs={"metadata": {"user_id": "test123"}},
        )

        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Response"
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(provider.client.messages, "create", new=mock_create):
            messages = [UserMessage(content="Test")]
            await provider.chat(messages)

            # Verify extra_kwargs were passed
            call_kwargs = mock_create.call_args.kwargs
            assert "metadata" in call_kwargs
            assert call_kwargs["metadata"]["user_id"] == "test123"


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic provider not installed")
class TestCreateAnthropicProviderFactory:
    """Tests for create_provider() factory function with Anthropic"""

    def test_create_anthropic_provider(self):
        """Test creating an Anthropic provider"""
        config = ModelConfig(
            name="claude-3-haiku-20240307",
            provider=Provider.ANTHROPIC,
            api_key="sk-ant-test-key",
        )

        provider = create_provider(config, timeout=30.0)

        assert isinstance(provider, AnthropicProvider)
        assert provider.model == "claude-3-haiku-20240307"

    def test_create_anthropic_provider_with_base_url(self):
        """Test creating Anthropic provider with custom base URL"""
        config = ModelConfig(
            name="claude-3-opus-20240229",
            provider=Provider.ANTHROPIC,
            api_key="sk-ant-test-key",
            base_url="https://custom.anthropic.endpoint/v1",
        )

        provider = create_provider(config)

        assert isinstance(provider, AnthropicProvider)
        assert provider.model == "claude-3-opus-20240229"

    def test_create_anthropic_provider_with_temperature(self):
        """Test creating Anthropic provider with temperature"""
        config = ModelConfig(
            name="claude-3-sonnet-20240229",
            provider=Provider.ANTHROPIC,
            api_key="sk-ant-test-key",
            temperature=0.5,
        )

        provider = create_provider(config)

        assert isinstance(provider, AnthropicProvider)
        assert provider.temperature == 0.5
