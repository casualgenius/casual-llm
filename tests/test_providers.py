"""
Tests for LLM provider implementations.
"""

import pytest
from pydantic import BaseModel
from unittest.mock import AsyncMock, MagicMock, patch
from casual_llm.config import ModelConfig, Provider, RetryConfig
from casual_llm.providers import OllamaProvider, create_provider
from casual_llm.messages import UserMessage, AssistantMessage, SystemMessage
from casual_llm.usage import Usage

# Import httpx and Ollama exceptions for retry testing
import httpx
from ollama import RequestError, ResponseError

# Import OpenAI exceptions for retry testing
try:
    from openai import (
        RateLimitError,
        APIConnectionError,
        InternalServerError,
        AuthenticationError,
        BadRequestError,
    )

    OPENAI_EXCEPTIONS_AVAILABLE = True
except ImportError:
    OPENAI_EXCEPTIONS_AVAILABLE = False


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


# Try to import OpenAI provider - may not be available
try:
    from casual_llm.providers import OpenAIProvider

    OPENAI_AVAILABLE = OpenAIProvider is not None
except ImportError:
    OPENAI_AVAILABLE = False


class TestRetryConfig:
    """Tests for RetryConfig dataclass"""

    def test_retry_config_defaults(self):
        """Test that RetryConfig has correct default values"""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.backoff_factor == 2.0

    def test_retry_config_custom_values(self):
        """Test that custom values are stored correctly"""
        config = RetryConfig(max_attempts=5, backoff_factor=1.5)

        assert config.max_attempts == 5
        assert config.backoff_factor == 1.5

    def test_retry_config_partial_custom_values(self):
        """Test that partial custom values work with defaults"""
        # Only override max_attempts
        config1 = RetryConfig(max_attempts=10)
        assert config1.max_attempts == 10
        assert config1.backoff_factor == 2.0  # default

        # Only override backoff_factor
        config2 = RetryConfig(backoff_factor=3.0)
        assert config2.max_attempts == 3  # default
        assert config2.backoff_factor == 3.0

    def test_retry_config_edge_case_values(self):
        """Test edge case values for RetryConfig"""
        # Zero max_attempts (disables retry)
        config1 = RetryConfig(max_attempts=0)
        assert config1.max_attempts == 0

        # Backoff factor of 1.0 (constant delay)
        config2 = RetryConfig(backoff_factor=1.0)
        assert config2.backoff_factor == 1.0

        # Single attempt
        config3 = RetryConfig(max_attempts=1)
        assert config3.max_attempts == 1


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

    @pytest.mark.asyncio
    async def test_json_schema_response_format(self, provider):
        """Test that Pydantic model is correctly converted to JSON Schema for Ollama"""
        mock_response = MagicMock()
        mock_response.message.content = '{"name": "Alice", "age": 30}'
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            messages = [UserMessage(content="Give me person info")]

            result = await provider.chat(messages, response_format=PersonInfo)

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
    async def test_json_schema_nested_pydantic_model(self, provider):
        """Test that complex nested Pydantic models work correctly"""
        mock_response = MagicMock()
        mock_response.message.content = '{"name": "Bob", "age": 25, "address": {"street": "123 Main St", "city": "NYC", "zip_code": "10001"}}'
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            messages = [UserMessage(content="Give me person with address")]

            result = await provider.chat(messages, response_format=PersonWithAddress)

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
    async def test_backward_compat_json_format(self, provider):
        """Test that existing 'json' format still works (backward compatibility)"""
        mock_response = MagicMock()
        mock_response.message.content = '{"status": "ok"}'
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            messages = [UserMessage(content="Give me JSON")]

            result = await provider.chat(messages, response_format="json")

            assert isinstance(result, AssistantMessage)
            assert '{"status": "ok"}' in result.content

            # Verify format is set to "json" string (not a schema dict)
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["format"] == "json"

    @pytest.mark.asyncio
    async def test_backward_compat_text_format(self, provider):
        """Test that existing 'text' format still works (backward compatibility)"""
        mock_response = MagicMock()
        mock_response.message.content = "Plain text response"
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            messages = [UserMessage(content="Give me text")]

            result = await provider.chat(messages, response_format="text")

            assert isinstance(result, AssistantMessage)
            assert result.content == "Plain text response"

            # Verify no format parameter is set for text
            call_kwargs = mock_chat.call_args.kwargs
            assert "format" not in call_kwargs

    # --- Ollama Retry Behavior Tests ---

    @pytest.fixture
    def provider_with_retry(self):
        """Create an OllamaProvider instance with retry config for testing"""
        return OllamaProvider(
            model="qwen2.5:7b-instruct",
            host="http://localhost:11434",
            temperature=0.7,
            retry_config=RetryConfig(max_attempts=3, backoff_factor=2.0),
        )

    def _create_mock_connect_error(self):
        """Create a mock httpx.ConnectError for testing"""
        return httpx.ConnectError("Connection refused")

    def _create_mock_timeout_error(self):
        """Create a mock httpx.TimeoutException for testing"""
        return httpx.TimeoutException("Request timed out")

    def _create_mock_response_error(self):
        """Create a mock ollama ResponseError for testing"""
        return ResponseError("Internal server error")

    def _create_mock_request_error(self):
        """Create a mock ollama RequestError for testing (non-retryable)"""
        return RequestError("Invalid request parameters")

    def _create_mock_ollama_success_response(self, content="Success response"):
        """Create a mock successful Ollama response"""
        mock_response = MagicMock()
        mock_response.message.content = content
        mock_response.message.tool_calls = None
        mock_response.prompt_eval_count = 10
        mock_response.eval_count = 20
        return mock_response

    @pytest.mark.asyncio
    async def test_connection_error_retry(self, provider_with_retry):
        """Test that httpx.ConnectError triggers retry with exponential backoff"""
        connect_error = self._create_mock_connect_error()
        success_response = self._create_mock_ollama_success_response()

        # First two calls fail with ConnectError, third succeeds
        mock_chat = AsyncMock(
            side_effect=[connect_error, connect_error, success_response]
        )

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
                messages = [UserMessage(content="Hello")]
                result = await provider_with_retry.chat(messages)

                assert isinstance(result, AssistantMessage)
                assert result.content == "Success response"

                # Verify retry attempts
                assert mock_chat.call_count == 3

                # Verify exponential backoff delays (1s, 2s for factor=2.0)
                assert mock_sleep.call_count == 2
                mock_sleep.assert_any_call(1.0)  # First retry: 2^0 = 1
                mock_sleep.assert_any_call(2.0)  # Second retry: 2^1 = 2

    @pytest.mark.asyncio
    async def test_timeout_error_retry(self, provider_with_retry):
        """Test that httpx.TimeoutException triggers retry"""
        timeout_error = self._create_mock_timeout_error()
        success_response = self._create_mock_ollama_success_response()

        mock_chat = AsyncMock(side_effect=[timeout_error, success_response])

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
                messages = [UserMessage(content="Hello")]
                result = await provider_with_retry.chat(messages)

                assert isinstance(result, AssistantMessage)
                assert mock_chat.call_count == 2
                assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_response_error_retry(self, provider_with_retry):
        """Test that ollama ResponseError triggers retry"""
        response_error = self._create_mock_response_error()
        success_response = self._create_mock_ollama_success_response()

        mock_chat = AsyncMock(side_effect=[response_error, success_response])

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
                messages = [UserMessage(content="Hello")]
                result = await provider_with_retry.chat(messages)

                assert isinstance(result, AssistantMessage)
                assert mock_chat.call_count == 2
                assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_request_error_no_retry(self, provider_with_retry):
        """Test that ollama RequestError fails immediately without retry"""
        request_error = self._create_mock_request_error()

        mock_chat = AsyncMock(side_effect=request_error)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
                messages = [UserMessage(content="Hello")]

                with pytest.raises(RequestError):
                    await provider_with_retry.chat(messages)

                # Verify no retries occurred
                assert mock_chat.call_count == 1
                assert mock_sleep.call_count == 0

    @pytest.mark.asyncio
    async def test_ollama_success_after_retry(self, provider_with_retry):
        """Test that response is returned successfully after transient failures"""
        connect_error = self._create_mock_connect_error()
        success_response = self._create_mock_ollama_success_response("Finally succeeded!")

        # First call fails, second succeeds
        mock_chat = AsyncMock(side_effect=[connect_error, success_response])

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            with patch("asyncio.sleep", new=AsyncMock()):
                messages = [UserMessage(content="Hello")]
                result = await provider_with_retry.chat(messages)

                assert isinstance(result, AssistantMessage)
                assert result.content == "Finally succeeded!"
                assert mock_chat.call_count == 2

    @pytest.mark.asyncio
    async def test_ollama_max_attempts_exhausted(self, provider_with_retry):
        """Test that original exception is raised after all retries fail"""
        connect_error = self._create_mock_connect_error()

        # All 3 attempts fail
        mock_chat = AsyncMock(side_effect=connect_error)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
                messages = [UserMessage(content="Hello")]

                with pytest.raises(httpx.ConnectError):
                    await provider_with_retry.chat(messages)

                # Verify all attempts were made
                assert mock_chat.call_count == 3

                # Verify backoff delays (only 2 sleeps, before attempts 2 and 3)
                assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_ollama_no_retry_when_config_none(self, provider):
        """Test that no retries occur when retry_config is None"""
        connect_error = self._create_mock_connect_error()

        mock_chat = AsyncMock(side_effect=connect_error)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
                messages = [UserMessage(content="Hello")]

                with pytest.raises(httpx.ConnectError):
                    await provider.chat(messages)

                # Verify only one attempt (no retries)
                assert mock_chat.call_count == 1
                assert mock_sleep.call_count == 0

    @pytest.mark.asyncio
    async def test_ollama_custom_backoff_factor(self):
        """Test that custom backoff_factor is used correctly"""
        provider = OllamaProvider(
            model="qwen2.5:7b-instruct",
            host="http://localhost:11434",
            retry_config=RetryConfig(max_attempts=3, backoff_factor=3.0),
        )

        connect_error = httpx.ConnectError("Connection refused")

        mock_response = MagicMock()
        mock_response.message.content = "Success"
        mock_response.message.tool_calls = None
        mock_response.prompt_eval_count = 10
        mock_response.eval_count = 20

        mock_chat = AsyncMock(
            side_effect=[connect_error, connect_error, mock_response]
        )

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
                messages = [UserMessage(content="Hello")]
                result = await provider.chat(messages)

                assert isinstance(result, AssistantMessage)
                assert mock_chat.call_count == 3

                # Verify custom backoff factor: 3^0=1, 3^1=3
                mock_sleep.assert_any_call(1.0)  # First retry: 3^0 = 1
                mock_sleep.assert_any_call(3.0)  # Second retry: 3^1 = 3


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

    @pytest.mark.asyncio
    async def test_json_schema_response_format(self, provider):
        """Test that Pydantic model is correctly converted to JSON Schema for OpenAI"""
        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(message=MagicMock(content='{"name": "Alice", "age": 30}'))
        ]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(provider.client.chat.completions, "create", new=mock_create):
            messages = [UserMessage(content="Give me person info")]

            result = await provider.chat(messages, response_format=PersonInfo)

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
    async def test_json_schema_nested_pydantic_model(self, provider):
        """Test that complex nested Pydantic models work correctly"""
        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"name": "Bob", "age": 25, "address": {"street": "123 Main St", "city": "NYC", "zip_code": "10001"}}'
                )
            )
        ]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(provider.client.chat.completions, "create", new=mock_create):
            messages = [UserMessage(content="Give me person with address")]

            result = await provider.chat(messages, response_format=PersonWithAddress)

            assert isinstance(result, AssistantMessage)

            # Verify the response_format parameter contains the nested JSON Schema
            call_kwargs = mock_create.call_args.kwargs
            assert "response_format" in call_kwargs
            response_format = call_kwargs["response_format"]

            # Verify OpenAI json_schema format structure
            assert response_format["type"] == "json_schema"
            assert response_format["json_schema"]["name"] == "PersonWithAddress"

            schema = response_format["json_schema"]["schema"]
            assert "properties" in schema

            # Verify nested structure is present (either through $defs or inline)
            # Pydantic v2 uses $defs for nested models
            if "$defs" in schema:
                assert "Address" in schema["$defs"]

    @pytest.mark.asyncio
    async def test_backward_compat_json_format(self, provider):
        """Test that existing 'json' format still works (backward compatibility)"""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content='{"status": "ok"}'))]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(provider.client.chat.completions, "create", new=mock_create):
            messages = [UserMessage(content="Give me JSON")]

            result = await provider.chat(messages, response_format="json")

            assert isinstance(result, AssistantMessage)
            assert '{"status": "ok"}' in result.content

            # Verify response_format is set to json_object (not json_schema)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_backward_compat_text_format(self, provider):
        """Test that existing 'text' format still works (backward compatibility)"""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Plain text response"))]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(provider.client.chat.completions, "create", new=mock_create):
            messages = [UserMessage(content="Give me text")]

            result = await provider.chat(messages, response_format="text")

            assert isinstance(result, AssistantMessage)
            assert result.content == "Plain text response"

            # Verify no response_format parameter is set for text
            call_kwargs = mock_create.call_args.kwargs
            assert "response_format" not in call_kwargs

    # --- OpenAI Retry Behavior Tests ---

    @pytest.fixture
    def provider_with_retry(self):
        """Create an OpenAIProvider instance with retry config for testing"""
        return OpenAIProvider(
            model="gpt-4o-mini",
            api_key="sk-test-key",
            temperature=0.7,
            retry_config=RetryConfig(max_attempts=3, backoff_factor=2.0),
        )

    def _create_mock_rate_limit_error(self):
        """Create a mock RateLimitError for testing"""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        return RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body=None,
        )

    def _create_mock_internal_server_error(self):
        """Create a mock InternalServerError for testing"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}
        return InternalServerError(
            message="Internal server error",
            response=mock_response,
            body=None,
        )

    def _create_mock_api_connection_error(self):
        """Create a mock APIConnectionError for testing"""
        mock_request = MagicMock()
        return APIConnectionError(request=mock_request)

    def _create_mock_auth_error(self):
        """Create a mock AuthenticationError for testing"""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        return AuthenticationError(
            message="Invalid API key",
            response=mock_response,
            body=None,
        )

    def _create_mock_bad_request_error(self):
        """Create a mock BadRequestError for testing"""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {}
        return BadRequestError(
            message="Bad request",
            response=mock_response,
            body=None,
        )

    def _create_mock_success_response(self, content="Success response"):
        """Create a mock successful API response"""
        mock_message = MagicMock(content=content)
        del mock_message.tool_calls  # Remove to match real behavior
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=mock_message)]
        mock_completion.usage = None
        return mock_completion

    @pytest.mark.asyncio
    async def test_rate_limit_retry(self, provider_with_retry):
        """Test that RateLimitError triggers retry with exponential backoff"""
        rate_limit_error = self._create_mock_rate_limit_error()
        success_response = self._create_mock_success_response()

        # First two calls fail with RateLimitError, third succeeds
        mock_create = AsyncMock(
            side_effect=[rate_limit_error, rate_limit_error, success_response]
        )

        with patch.object(
            provider_with_retry.client.chat.completions, "create", new=mock_create
        ):
            with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
                messages = [UserMessage(content="Hello")]
                result = await provider_with_retry.chat(messages)

                assert isinstance(result, AssistantMessage)
                assert result.content == "Success response"

                # Verify retry attempts
                assert mock_create.call_count == 3

                # Verify exponential backoff delays (1s, 2s for factor=2.0)
                assert mock_sleep.call_count == 2
                mock_sleep.assert_any_call(1.0)  # First retry: 2^0 = 1
                mock_sleep.assert_any_call(2.0)  # Second retry: 2^1 = 2

    @pytest.mark.asyncio
    async def test_server_error_retry(self, provider_with_retry):
        """Test that InternalServerError triggers retry"""
        server_error = self._create_mock_internal_server_error()
        success_response = self._create_mock_success_response()

        mock_create = AsyncMock(side_effect=[server_error, success_response])

        with patch.object(
            provider_with_retry.client.chat.completions, "create", new=mock_create
        ):
            with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
                messages = [UserMessage(content="Hello")]
                result = await provider_with_retry.chat(messages)

                assert isinstance(result, AssistantMessage)
                assert mock_create.call_count == 2
                assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_connection_error_retry(self, provider_with_retry):
        """Test that APIConnectionError triggers retry"""
        connection_error = self._create_mock_api_connection_error()
        success_response = self._create_mock_success_response()

        mock_create = AsyncMock(side_effect=[connection_error, success_response])

        with patch.object(
            provider_with_retry.client.chat.completions, "create", new=mock_create
        ):
            with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
                messages = [UserMessage(content="Hello")]
                result = await provider_with_retry.chat(messages)

                assert isinstance(result, AssistantMessage)
                assert mock_create.call_count == 2
                assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_auth_error_no_retry(self, provider_with_retry):
        """Test that AuthenticationError fails immediately without retry"""
        auth_error = self._create_mock_auth_error()

        mock_create = AsyncMock(side_effect=auth_error)

        with patch.object(
            provider_with_retry.client.chat.completions, "create", new=mock_create
        ):
            with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
                messages = [UserMessage(content="Hello")]

                with pytest.raises(AuthenticationError):
                    await provider_with_retry.chat(messages)

                # Verify no retries occurred
                assert mock_create.call_count == 1
                assert mock_sleep.call_count == 0

    @pytest.mark.asyncio
    async def test_bad_request_no_retry(self, provider_with_retry):
        """Test that BadRequestError fails immediately without retry"""
        bad_request_error = self._create_mock_bad_request_error()

        mock_create = AsyncMock(side_effect=bad_request_error)

        with patch.object(
            provider_with_retry.client.chat.completions, "create", new=mock_create
        ):
            with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
                messages = [UserMessage(content="Hello")]

                with pytest.raises(BadRequestError):
                    await provider_with_retry.chat(messages)

                # Verify no retries occurred
                assert mock_create.call_count == 1
                assert mock_sleep.call_count == 0

    @pytest.mark.asyncio
    async def test_success_after_retry(self, provider_with_retry):
        """Test that response is returned successfully after transient failures"""
        rate_limit_error = self._create_mock_rate_limit_error()
        success_response = self._create_mock_success_response("Finally succeeded!")

        # First call fails, second succeeds
        mock_create = AsyncMock(side_effect=[rate_limit_error, success_response])

        with patch.object(
            provider_with_retry.client.chat.completions, "create", new=mock_create
        ):
            with patch("asyncio.sleep", new=AsyncMock()):
                messages = [UserMessage(content="Hello")]
                result = await provider_with_retry.chat(messages)

                assert isinstance(result, AssistantMessage)
                assert result.content == "Finally succeeded!"
                assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_max_attempts_exhausted(self, provider_with_retry):
        """Test that original exception is raised after all retries fail"""
        rate_limit_error = self._create_mock_rate_limit_error()

        # All 3 attempts fail
        mock_create = AsyncMock(side_effect=rate_limit_error)

        with patch.object(
            provider_with_retry.client.chat.completions, "create", new=mock_create
        ):
            with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
                messages = [UserMessage(content="Hello")]

                with pytest.raises(RateLimitError):
                    await provider_with_retry.chat(messages)

                # Verify all attempts were made
                assert mock_create.call_count == 3

                # Verify backoff delays (only 2 sleeps, before attempts 2 and 3)
                assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_when_config_none(self, provider):
        """Test that no retries occur when retry_config is None"""
        rate_limit_error = self._create_mock_rate_limit_error()

        mock_create = AsyncMock(side_effect=rate_limit_error)

        with patch.object(
            provider.client.chat.completions, "create", new=mock_create
        ):
            with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
                messages = [UserMessage(content="Hello")]

                with pytest.raises(RateLimitError):
                    await provider.chat(messages)

                # Verify only one attempt (no retries)
                assert mock_create.call_count == 1
                assert mock_sleep.call_count == 0

    @pytest.mark.asyncio
    async def test_custom_backoff_factor(self):
        """Test that custom backoff_factor is used correctly"""
        provider = OpenAIProvider(
            model="gpt-4o-mini",
            api_key="sk-test-key",
            retry_config=RetryConfig(max_attempts=3, backoff_factor=3.0),
        )

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        rate_limit_error = RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body=None,
        )

        mock_message = MagicMock(content="Success")
        del mock_message.tool_calls
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=mock_message)]
        mock_completion.usage = None

        mock_create = AsyncMock(
            side_effect=[rate_limit_error, rate_limit_error, mock_completion]
        )

        with patch.object(
            provider.client.chat.completions, "create", new=mock_create
        ):
            with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
                messages = [UserMessage(content="Hello")]
                result = await provider.chat(messages)

                assert isinstance(result, AssistantMessage)
                assert mock_create.call_count == 3

                # Verify custom backoff factor: 3^0=1, 3^1=3
                mock_sleep.assert_any_call(1.0)  # First retry: 3^0 = 1
                mock_sleep.assert_any_call(3.0)  # Second retry: 3^1 = 3


class TestCreateProviderFactory:
    """Tests for create_provider() factory function"""

    def test_create_ollama_provider(self):
        """Test creating an Ollama provider"""
        config = ModelConfig(
            name="qwen2.5:7b-instruct",
            provider=Provider.OLLAMA,
            base_url="http://localhost:11434/api/chat",
        )

        provider = create_provider(config, timeout=60.0)

        assert isinstance(provider, OllamaProvider)
        assert provider.model == "qwen2.5:7b-instruct"
        assert provider.timeout == 60.0

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
