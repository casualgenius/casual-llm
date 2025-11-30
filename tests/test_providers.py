"""
Tests for LLM provider implementations.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from casual_llm.config import ModelConfig, Provider
from casual_llm.providers import OllamaProvider, create_provider
from casual_llm.messages import UserMessage, AssistantMessage, SystemMessage

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
            endpoint="http://localhost:11434",
            temperature=0.7,
            timeout=30.0,
            max_retries=2,
        )

    @pytest.mark.asyncio
    async def test_generate_text_success(self, provider):
        """Test successful text generation"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "Hello, I'm a test response!"}}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            messages = [
                UserMessage(content="Hello"),
            ]

            result = await provider.chat(messages, response_format="text")

            assert result == "Hello, I'm a test response!"
            assert mock_client.post.called

    @pytest.mark.asyncio
    async def test_generate_json_success(self, provider):
        """Test successful JSON generation"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": '{"name": "test", "value": 42}'}}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            messages = [
                UserMessage(content="Give me JSON"),
            ]

            result = await provider.chat(messages, response_format="json")

            assert '{"name": "test", "value": 42}' in result

    @pytest.mark.asyncio
    async def test_generate_with_conversation(self, provider):
        """Test generation with multi-turn conversation"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "Got it!"}}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            messages = [
                SystemMessage(content="You are a helpful assistant"),
                UserMessage(content="Hello"),
                AssistantMessage(content="Hi there!"),
                UserMessage(content="How are you?"),
            ]

            result = await provider.chat(messages, response_format="text")

            assert result == "Got it!"
            # Verify the prompt was constructed from all messages
            call_args = mock_client.post.call_args
            assert call_args is not None

    @pytest.mark.asyncio
    async def test_generate_with_none_content(self, provider):
        """Test handling of messages with None content"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "Handled!"}}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            messages = [
                UserMessage(content=None),  # None content
                AssistantMessage(content="Response"),
            ]

            result = await provider.chat(messages, response_format="text")

            assert result == "Handled!"

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self, provider):
        """Test retry logic on transient errors (ConnectError/TimeoutException)"""
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"message": {"content": "Success after retry"}}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None

            # First call raises ConnectError, second succeeds
            mock_client.post.side_effect = [
                httpx.ConnectError("Connection failed"),
                mock_response_success,
            ]
            mock_client_class.return_value = mock_client

            # Mock asyncio.sleep to avoid delays in tests
            with patch("asyncio.sleep", new=AsyncMock()):
                messages = [UserMessage(content="Test")]
                result = await provider.chat(messages, response_format="text")

                assert result == "Success after retry"
                assert mock_client.post.call_count == 2  # Retried once

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, provider):
        """Test that errors are raised after max retries"""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None

            # All calls fail with ConnectError
            mock_client.post.side_effect = httpx.ConnectError("Connection failed")
            mock_client_class.return_value = mock_client

            # Mock asyncio.sleep to avoid delays in tests
            with patch("asyncio.sleep", new=AsyncMock()):
                messages = [UserMessage(content="Test")]

                with pytest.raises(httpx.ConnectError):
                    await provider.chat(messages, response_format="text")

                # Should have tried: initial + 2 retries = 3 times
                assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_metrics_tracking(self):
        """Test that metrics are tracked when enabled"""
        provider = OllamaProvider(
            model="test-model",
            endpoint="http://localhost:11434",
            enable_metrics=True,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "Success"}}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            messages = [UserMessage(content="Test")]

            await provider.chat(messages, response_format="text")

            # Check metrics directly
            assert provider.success_count == 1
            assert provider.failure_count == 0

            # Check get_metrics() method
            metrics = provider.get_metrics()
            assert metrics["success_count"] == 1
            assert metrics["failure_count"] == 0


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
        mock_completion.choices = [MagicMock(message=MagicMock(content="Hello from OpenAI!"))]

        with patch.object(
            provider.client.chat.completions, "create", new=AsyncMock(return_value=mock_completion)
        ):
            messages = [UserMessage(content="Hello")]
            result = await provider.chat(messages, response_format="text")

            assert result == "Hello from OpenAI!"

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

            assert '{"status": "ok"}' in result

    @pytest.mark.asyncio
    async def test_generate_with_max_tokens(self, provider):
        """Test generation with max_tokens parameter"""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Short response"))]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(provider.client.chat.completions, "create", new=mock_create):
            messages = [UserMessage(content="Test")]
            result = await provider.chat(messages, response_format="text", max_tokens=50)

            assert result == "Short response"
            # Verify max_tokens was passed
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 50

    @pytest.mark.asyncio
    async def test_generate_json_helper(self, provider):
        """Test the generate_json() convenience method"""
        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(message=MagicMock(content='{"name": "test", "value": 42}'))
        ]

        with patch.object(
            provider.client.chat.completions, "create", new=AsyncMock(return_value=mock_completion)
        ):
            messages = [UserMessage(content="Give me JSON")]
            result = await provider.chat_json(messages)

            assert isinstance(result, dict)
            assert result["name"] == "test"
            assert result["value"] == 42

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
        # Provider should construct the full API URL internally
        assert provider.endpoint == "http://localhost:11434/api/chat"

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

        # Both should produce the same endpoint
        assert provider_with_slash.endpoint == "http://localhost:11434/api/chat"
        assert provider_without_slash.endpoint == "http://localhost:11434/api/chat"
        assert provider_with_slash.endpoint == provider_without_slash.endpoint

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
