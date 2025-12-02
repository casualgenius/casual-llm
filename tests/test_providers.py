"""
Tests for LLM provider implementations.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from casual_llm.config import ModelConfig, Provider
from casual_llm.providers import OllamaProvider, create_provider
from casual_llm.messages import UserMessage, AssistantMessage, SystemMessage
from casual_llm.usage import Usage

# Try to import OpenAI provider - may not be available
try:
    from casual_llm.providers import OpenAIProvider

    OPENAI_AVAILABLE = OpenAIProvider is not None
except ImportError:
    OPENAI_AVAILABLE = False


class TestOllamaProvider:
    """Tests for OllamaProvider"""

    @pytest.fixture
    def provider(self):
        """Create an OllamaProvider instance for testing"""
        return OllamaProvider(
            model="qwen2.5:7b-instruct",
            host="http://localhost:11434",
            temperature=0.7,
            timeout=30.0,
            max_retries=2,
        )

    @pytest.mark.asyncio
    async def test_generate_text_success(self, provider):
        """Test successful text generation"""
        mock_response = MagicMock()
        mock_response.message.content = "Hello, I'm a test response!"
        mock_response.message.tool_calls = None

        with patch("ollama.AsyncClient.chat", new=AsyncMock(return_value=mock_response)):
            messages = [
                UserMessage(content="Hello"),
            ]

            result = await provider.chat(messages, response_format="text")

            assert isinstance(result, AssistantMessage)
            assert result.content == "Hello, I'm a test response!"
            assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_generate_json_success(self, provider):
        """Test successful JSON generation"""
        mock_response = MagicMock()
        mock_response.message.content = '{"name": "test", "value": 42}'
        mock_response.message.tool_calls = None

        with patch("ollama.AsyncClient.chat", new=AsyncMock(return_value=mock_response)):
            messages = [
                UserMessage(content="Give me JSON"),
            ]

            result = await provider.chat(messages, response_format="json")

            assert isinstance(result, AssistantMessage)
            assert '{"name": "test", "value": 42}' in result.content

    @pytest.mark.asyncio
    async def test_generate_with_conversation(self, provider):
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

            result = await provider.chat(messages, response_format="text")

            assert isinstance(result, AssistantMessage)
            assert result.content == "Got it!"
            # Verify the messages were passed
            call_args = mock_chat.call_args
            assert call_args is not None

    @pytest.mark.asyncio
    async def test_generate_with_none_content(self, provider):
        """Test handling of messages with None content"""
        mock_response = MagicMock()
        mock_response.message.content = "Handled!"
        mock_response.message.tool_calls = None

        with patch("ollama.AsyncClient.chat", new=AsyncMock(return_value=mock_response)):
            messages = [
                UserMessage(content=None),  # None content
                AssistantMessage(content="Response"),
            ]

            result = await provider.chat(messages, response_format="text")

            assert isinstance(result, AssistantMessage)
            assert result.content == "Handled!"

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self, provider):
        """Test retry logic on transient errors (ConnectionError/TimeoutError)"""
        mock_response_success = MagicMock()
        mock_response_success.message.content = "Success after retry"
        mock_response_success.message.tool_calls = None

        mock_chat = AsyncMock()
        # First call raises ConnectionError, second succeeds
        mock_chat.side_effect = [
            ConnectionError("Connection failed"),
            mock_response_success,
        ]

        # Mock asyncio.sleep to avoid delays in tests
        with patch("ollama.AsyncClient.chat", new=mock_chat):
            with patch("asyncio.sleep", new=AsyncMock()):
                messages = [UserMessage(content="Test")]
                result = await provider.chat(messages, response_format="text")

                assert isinstance(result, AssistantMessage)
                assert result.content == "Success after retry"
                assert mock_chat.call_count == 2  # Retried once

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, provider):
        """Test that errors are raised after max retries"""
        mock_chat = AsyncMock()
        # All calls fail with ConnectionError
        mock_chat.side_effect = ConnectionError("Connection failed")

        # Mock asyncio.sleep to avoid delays in tests
        with patch("ollama.AsyncClient.chat", new=mock_chat):
            with patch("asyncio.sleep", new=AsyncMock()):
                messages = [UserMessage(content="Test")]

                with pytest.raises(ConnectionError):
                    await provider.chat(messages, response_format="text")

                # Should have tried: initial + 2 retries = 3 times
                assert mock_chat.call_count == 3

    @pytest.mark.asyncio
    async def test_metrics_tracking(self):
        """Test that metrics are tracked when enabled"""
        provider = OllamaProvider(
            model="test-model",
            host="http://localhost:11434",
            enable_metrics=True,
        )

        mock_response = MagicMock()
        mock_response.message.content = "Success"
        mock_response.message.tool_calls = None

        with patch("ollama.AsyncClient.chat", new=AsyncMock(return_value=mock_response)):
            messages = [UserMessage(content="Test")]

            await provider.chat(messages, response_format="text")

            # Check metrics directly
            assert provider.success_count == 1
            assert provider.failure_count == 0

            # Check get_metrics() method
            metrics = provider.get_metrics()
            assert metrics["success_count"] == 1
            assert metrics["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_temperature_override(self, provider):
        """Test that per-call temperature overrides instance temperature"""
        # Provider was created with temperature=0.7
        assert provider.temperature == 0.7

        mock_response = MagicMock()
        mock_response.message.content = "Response"
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            messages = [UserMessage(content="Test")]

            # Call with overridden temperature
            await provider.chat(messages, temperature=0.1)

            # Verify the temperature passed to Ollama
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["options"]["temperature"] == 0.1

            # Call without override - should use instance temperature
            await provider.chat(messages)
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["options"]["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_temperature_none_not_sent(self):
        """Test that temperature is not sent to API when None"""
        # Create provider without temperature (defaults to None)
        provider = OllamaProvider(
            model="test-model",
            host="http://localhost:11434",
        )
        assert provider.temperature is None

        mock_response = MagicMock()
        mock_response.message.content = "Response"
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            messages = [UserMessage(content="Test")]

            # Call without temperature - should not include it in options
            await provider.chat(messages)

            # Verify temperature was NOT included in options
            call_kwargs = mock_chat.call_args.kwargs
            assert "temperature" not in call_kwargs["options"]

    @pytest.mark.asyncio
    async def test_usage_tracking(self, provider):
        """Test that usage statistics are tracked"""
        # Check initial state
        assert provider.get_usage() is None

        mock_response = MagicMock()
        mock_response.message.content = "Response"
        mock_response.message.tool_calls = None
        # Ollama uses prompt_eval_count and eval_count
        mock_response.prompt_eval_count = 10
        mock_response.eval_count = 20

        with patch("ollama.AsyncClient.chat", new=AsyncMock(return_value=mock_response)):
            messages = [UserMessage(content="Test")]
            await provider.chat(messages)

            # Verify usage was tracked
            usage = provider.get_usage()
            assert usage is not None
            assert isinstance(usage, Usage)
            assert usage.prompt_tokens == 10
            assert usage.completion_tokens == 20
            assert usage.total_tokens == 30


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI provider not installed")
class TestOpenAIProvider:
    """Tests for OpenAIProvider"""

    @pytest.fixture
    def provider(self):
        """Create an OpenAIProvider instance for testing"""
        return OpenAIProvider(
            model="gpt-4o-mini",
            api_key="sk-test-key",
            temperature=0.7,
        )

    @pytest.mark.asyncio
    async def test_generate_text_success(self, provider):
        """Test successful text generation"""
        mock_completion = MagicMock()
        mock_message = MagicMock(content="Hello from OpenAI!", tool_calls=None)
        # Remove tool_calls attribute entirely to match real behavior
        del mock_message.tool_calls
        mock_completion.choices = [MagicMock(message=mock_message)]

        with patch.object(
            provider.client.chat.completions, "create", new=AsyncMock(return_value=mock_completion)
        ):
            messages = [UserMessage(content="Hello")]
            result = await provider.chat(messages, response_format="text")

            assert isinstance(result, AssistantMessage)
            assert result.content == "Hello from OpenAI!"
            assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_generate_json_success(self, provider):
        """Test successful JSON generation"""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content='{"status": "ok"}'))]

        with patch.object(
            provider.client.chat.completions, "create", new=AsyncMock(return_value=mock_completion)
        ):
            messages = [UserMessage(content="Give me JSON")]
            result = await provider.chat(messages, response_format="json")

            assert isinstance(result, AssistantMessage)
            assert '{"status": "ok"}' in result.content

    @pytest.mark.asyncio
    async def test_generate_with_max_tokens(self, provider):
        """Test generation with max_tokens parameter"""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Short response"))]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(provider.client.chat.completions, "create", new=mock_create):
            messages = [UserMessage(content="Test")]
            result = await provider.chat(messages, response_format="text", max_tokens=50)

            assert isinstance(result, AssistantMessage)
            assert result.content == "Short response"
            # Verify max_tokens was passed
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 50

    @pytest.mark.asyncio
    async def test_message_conversion(self, provider):
        """Test that ChatMessages are converted correctly to OpenAI format"""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Response"))]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(provider.client.chat.completions, "create", new=mock_create):
            messages = [
                SystemMessage(content="You are helpful"),
                UserMessage(content="Hello"),
                AssistantMessage(content="Hi!"),
            ]

            await provider.chat(messages, response_format="text")

            # Verify messages were converted to dict format
            call_kwargs = mock_create.call_args.kwargs
            chat_messages = call_kwargs["messages"]

            assert len(chat_messages) == 3
            assert chat_messages[0]["role"] == "system"
            assert chat_messages[0]["content"] == "You are helpful"
            assert chat_messages[1]["role"] == "user"
            assert chat_messages[2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_temperature_override(self, provider):
        """Test that per-call temperature overrides instance temperature"""
        # Provider was created with temperature=0.7
        assert provider.temperature == 0.7

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Response"))]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(provider.client.chat.completions, "create", new=mock_create):
            messages = [UserMessage(content="Test")]

            # Call with overridden temperature
            await provider.chat(messages, temperature=0.1)

            # Verify the temperature passed to OpenAI
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
        provider = OpenAIProvider(
            model="gpt-4o-mini",
            api_key="sk-test-key",
        )
        assert provider.temperature is None

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Response"))]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(provider.client.chat.completions, "create", new=mock_create):
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
            provider.client.chat.completions, "create", new=AsyncMock(return_value=mock_completion)
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


class TestCreateProviderFactory:
    """Tests for create_provider() factory function"""

    def test_create_ollama_provider(self):
        """Test creating an Ollama provider"""
        config = ModelConfig(
            name="qwen2.5:7b-instruct",
            provider=Provider.OLLAMA,
            base_url="http://localhost:11434/api/chat",
        )

        provider = create_provider(config, timeout=60.0, max_retries=2)

        assert isinstance(provider, OllamaProvider)
        assert provider.model == "qwen2.5:7b-instruct"
        assert provider.timeout == 60.0
        assert provider.max_retries == 2

    def test_create_ollama_provider_with_default_url(self):
        """Test creating Ollama provider with default URL"""
        config = ModelConfig(
            name="llama2",
            provider=Provider.OLLAMA,
        )

        provider = create_provider(config)

        assert isinstance(provider, OllamaProvider)
        # Provider should use default host
        assert provider.host == "http://localhost:11434"

    def test_create_ollama_provider_with_trailing_slash(self):
        """Test that trailing slashes are handled correctly"""
        # With trailing slash
        config_with_slash = ModelConfig(
            name="llama2",
            provider=Provider.OLLAMA,
            base_url="http://localhost:11434/",
        )
        provider_with_slash = create_provider(config_with_slash)

        # Without trailing slash
        config_without_slash = ModelConfig(
            name="llama2",
            provider=Provider.OLLAMA,
            base_url="http://localhost:11434",
        )
        provider_without_slash = create_provider(config_without_slash)

        # Both should produce the same host (trailing slash removed)
        assert provider_with_slash.host == "http://localhost:11434"
        assert provider_without_slash.host == "http://localhost:11434"
        assert provider_with_slash.host == provider_without_slash.host

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI provider not installed")
    def test_create_openai_provider(self):
        """Test creating an OpenAI provider"""
        config = ModelConfig(
            name="gpt-4o-mini",
            provider=Provider.OPENAI,
            api_key="sk-test-key",
        )

        provider = create_provider(config, timeout=30.0)

        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4o-mini"

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI provider not installed")
    def test_create_openai_provider_with_base_url(self):
        """Test creating OpenAI provider with custom base URL"""
        config = ModelConfig(
            name="gpt-4",
            provider=Provider.OPENAI,
            api_key="sk-test-key",
            base_url="https://api.openrouter.ai/api/v1",
        )

        provider = create_provider(config)

        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4"

    def test_create_provider_unsupported_type(self):
        """Test that unsupported provider raises ValueError"""
        # Create a mock provider enum value
        config = ModelConfig(
            name="test-model",
            provider="unsupported",  # type: ignore
        )

        with pytest.raises(ValueError, match="Unsupported provider"):
            create_provider(config)

    def test_create_provider_metrics_enabled(self):
        """Test creating provider with metrics enabled"""
        config = ModelConfig(
            name="test-model",
            provider=Provider.OLLAMA,
        )

        provider = create_provider(config, enable_metrics=True)

        assert isinstance(provider, OllamaProvider)
        assert provider.enable_metrics is True
        # Verify metrics tracking works
        metrics = provider.get_metrics()
        assert "success_count" in metrics or metrics == {}  # Empty if no calls yet
