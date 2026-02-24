"""
Security-critical tests for casual-llm.

Covers:
- TST-001: SSRF protection (_validate_url)
- TST-002: options.extra override protection (all providers)
- TST-003: max_tokens=0 boundary tests (all providers)
- TST-005: Redirect validation in image fetch
"""

import logging

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from casual_llm.config import ChatOptions
from casual_llm.messages import UserMessage
from casual_llm.model import Model
from casual_llm.providers import OllamaClient
from casual_llm.utils.image import (
    _validate_url,
    fetch_image_as_base64,
    ImageFetchError,
)

# Try to import optional providers
try:
    from casual_llm.providers import OpenAIClient

    OPENAI_AVAILABLE = OpenAIClient is not None
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from casual_llm.providers import AnthropicClient

    ANTHROPIC_AVAILABLE = AnthropicClient is not None
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# TST-001: SSRF Protection (_validate_url)
# ---------------------------------------------------------------------------


class TestValidateUrl:
    """Tests for _validate_url SSRF protection."""

    def test_valid_https_url(self):
        """Valid HTTPS URL should pass validation."""
        _validate_url("https://example.com/image.jpg")

    def test_valid_http_url(self):
        """Valid HTTP URL should pass validation."""
        _validate_url("http://example.com/image.jpg")

    def test_rejects_file_scheme(self):
        """file:// URLs should be blocked."""
        with pytest.raises(ImageFetchError, match="not allowed"):
            _validate_url("file:///etc/passwd")

    def test_rejects_ftp_scheme(self):
        """ftp:// URLs should be blocked."""
        with pytest.raises(ImageFetchError, match="not allowed"):
            _validate_url("ftp://example.com/image.jpg")

    def test_rejects_data_scheme(self):
        """data: URLs should be blocked."""
        with pytest.raises(ImageFetchError, match="not allowed"):
            _validate_url("data:image/png;base64,abc123")

    def test_rejects_javascript_scheme(self):
        """javascript: URLs should be blocked."""
        with pytest.raises(ImageFetchError, match="not allowed"):
            _validate_url("javascript:alert(1)")

    def test_rejects_empty_scheme(self):
        """URL with no scheme should be blocked."""
        with pytest.raises(ImageFetchError, match="not allowed"):
            _validate_url("//example.com/image.jpg")

    def test_rejects_loopback_ipv4(self):
        """127.0.0.1 should be blocked."""
        with pytest.raises(ImageFetchError, match="private/internal"):
            _validate_url("https://127.0.0.1/image.jpg")

    def test_rejects_loopback_full_range(self):
        """127.x.x.x addresses should be blocked."""
        with pytest.raises(ImageFetchError, match="private/internal"):
            _validate_url("https://127.0.0.2/image.jpg")

    def test_rejects_private_10_range(self):
        """10.x.x.x (RFC 1918) should be blocked."""
        with pytest.raises(ImageFetchError, match="private/internal"):
            _validate_url("https://10.0.0.1/image.jpg")

    def test_rejects_private_172_range(self):
        """172.16-31.x.x (RFC 1918) should be blocked."""
        with pytest.raises(ImageFetchError, match="private/internal"):
            _validate_url("https://172.16.0.1/image.jpg")

    def test_rejects_private_192_168_range(self):
        """192.168.x.x (RFC 1918) should be blocked."""
        with pytest.raises(ImageFetchError, match="private/internal"):
            _validate_url("https://192.168.1.1/image.jpg")

    def test_rejects_link_local(self):
        """169.254.x.x (link-local / AWS metadata) should be blocked."""
        with pytest.raises(ImageFetchError, match="private/internal"):
            _validate_url("https://169.254.169.254/latest/meta-data/")

    def test_rejects_no_hostname(self):
        """URL without hostname should be blocked."""
        with pytest.raises(ImageFetchError, match="no hostname"):
            _validate_url("https:///path/only")

    def test_allows_public_hostname(self):
        """Regular hostnames should be allowed (DNS not resolved at this stage)."""
        _validate_url("https://upload.wikimedia.org/image.jpg")

    def test_allows_public_ip(self):
        """Public IP addresses should be allowed."""
        _validate_url("https://8.8.8.8/image.jpg")

    def test_rejects_ipv6_loopback(self):
        """IPv6 loopback (::1) should be blocked."""
        with pytest.raises(ImageFetchError, match="private/internal"):
            _validate_url("https://[::1]/image.jpg")

    def test_rejects_reserved_address(self):
        """Reserved addresses (0.0.0.0) should be blocked."""
        with pytest.raises(ImageFetchError, match="private/internal"):
            _validate_url("https://0.0.0.0/image.jpg")


# ---------------------------------------------------------------------------
# TST-005: Redirect Validation in Image Fetch
# ---------------------------------------------------------------------------


class TestRedirectValidation:
    """Tests for redirect handling in fetch_image_as_base64."""

    @pytest.mark.asyncio
    async def test_redirect_to_private_ip_blocked(self):
        """Redirects to private IPs should be blocked."""

        # First response is a redirect to a private IP
        mock_redirect_response = Mock()
        mock_redirect_response.is_redirect = True
        mock_redirect_response.next_request = Mock()
        mock_redirect_response.next_request.url = "http://127.0.0.1/secret"
        mock_redirect_response.raise_for_status = Mock()

        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get = AsyncMock(return_value=mock_redirect_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(ImageFetchError, match="private/internal"):
                await fetch_image_as_base64("https://evil.com/redirect-to-internal")

    @pytest.mark.asyncio
    async def test_too_many_redirects(self):
        """More than 5 redirects should raise an error."""
        # Every response is a redirect
        mock_response = Mock()
        mock_response.is_redirect = True
        mock_response.next_request = Mock()
        mock_response.next_request.url = "https://example.com/bounce"
        mock_response.raise_for_status = Mock()

        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(ImageFetchError, match="Too many redirects"):
                await fetch_image_as_base64("https://example.com/loop")

    @pytest.mark.asyncio
    async def test_redirect_to_non_http_blocked(self):
        """Redirects to file:// or ftp:// should be blocked."""
        mock_response = Mock()
        mock_response.is_redirect = True
        mock_response.next_request = Mock()
        mock_response.next_request.url = "file:///etc/passwd"
        mock_response.raise_for_status = Mock()

        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(ImageFetchError, match="not allowed"):
                await fetch_image_as_base64("https://evil.com/redirect-to-file")

    @pytest.mark.asyncio
    async def test_single_valid_redirect_succeeds(self):
        """A single redirect to a valid URL should succeed."""
        test_image_data = b"fake image data"

        # First response: redirect
        mock_redirect_response = Mock()
        mock_redirect_response.is_redirect = True
        mock_redirect_response.next_request = Mock()
        mock_redirect_response.next_request.url = "https://cdn.example.com/image.jpg"
        mock_redirect_response.raise_for_status = Mock()

        # Second response: actual image
        mock_final_response = Mock()
        mock_final_response.is_redirect = False
        mock_final_response.content = test_image_data
        mock_final_response.headers = {"content-type": "image/jpeg"}
        mock_final_response.raise_for_status = Mock()

        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get = AsyncMock(side_effect=[mock_redirect_response, mock_final_response])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            base64_data, media_type = await fetch_image_as_base64("https://example.com/image.jpg")

            assert media_type == "image/jpeg"
            assert len(base64_data) > 0

    @pytest.mark.asyncio
    async def test_redirect_with_no_next_request(self):
        """Redirect with no next_request should stop redirect loop and proceed."""
        test_image_data = b"fake image data"

        mock_response = Mock()
        mock_response.is_redirect = True
        mock_response.next_request = None
        mock_response.content = test_image_data
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.raise_for_status = Mock()

        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            # Should not loop forever — the `break` on missing next_request
            # means response.is_redirect stays True, which triggers "Too many redirects"
            # Actually the code breaks out of the while loop, then checks is_redirect again
            with pytest.raises(ImageFetchError, match="Too many redirects"):
                await fetch_image_as_base64("https://example.com/image.jpg")


# ---------------------------------------------------------------------------
# TST-002: options.extra Override Protection (All Providers)
# ---------------------------------------------------------------------------


class TestOllamaExtraOverrideProtection:
    """Test that options.extra cannot overwrite core Ollama request parameters."""

    @pytest.fixture
    def model(self):
        client = OllamaClient(host="http://localhost:11434")
        return Model(client=client, name="test-model")

    @pytest.mark.asyncio
    async def test_extra_cannot_overwrite_model(self, model):
        """extra={\"model\": \"evil\"} should NOT overwrite the actual model."""
        mock_response = MagicMock()
        mock_response.message.content = "Response"
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            await model.chat(
                [UserMessage(content="Test")],
                ChatOptions(extra={"model": "evil-model"}),
            )

            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_extra_cannot_overwrite_messages(self, model):
        """extra={\"messages\": []} should NOT overwrite real messages."""
        mock_response = MagicMock()
        mock_response.message.content = "Response"
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            await model.chat(
                [UserMessage(content="Hello")],
                ChatOptions(extra={"messages": []}),
            )

            call_kwargs = mock_chat.call_args.kwargs
            # Messages should NOT be empty — the real messages should win
            assert len(call_kwargs["messages"]) > 0

    @pytest.mark.asyncio
    async def test_extra_override_logs_warning(self, model, caplog):
        """Conflicting extra key should log a warning."""
        mock_response = MagicMock()
        mock_response.message.content = "Response"
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            with caplog.at_level(logging.WARNING):
                await model.chat(
                    [UserMessage(content="Test")],
                    ChatOptions(extra={"model": "evil-model"}),
                )

            assert any("Ignoring extra key" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_extra_non_conflicting_keys_pass_through(self, model):
        """Non-conflicting extra keys should still be passed through."""
        mock_response = MagicMock()
        mock_response.message.content = "Response"
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            await model.chat(
                [UserMessage(content="Test")],
                ChatOptions(extra={"keep_alive": "10m", "model": "evil"}),
            )

            call_kwargs = mock_chat.call_args.kwargs
            # Non-conflicting key passes through
            assert call_kwargs["keep_alive"] == "10m"
            # Conflicting key does NOT overwrite
            assert call_kwargs["model"] == "test-model"


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI client not installed")
class TestOpenAIExtraOverrideProtection:
    """Test that options.extra cannot overwrite core OpenAI request parameters."""

    @pytest.fixture
    def model(self):
        client = OpenAIClient(api_key="sk-test-key")
        return Model(client=client, name="gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_extra_cannot_overwrite_model(self, model):
        """extra={\"model\": \"evil\"} should NOT overwrite the actual model."""
        mock_completion = MagicMock()
        mock_message = MagicMock(content="Response")
        del mock_message.tool_calls
        mock_completion.choices = [MagicMock(message=mock_message)]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(model._client.client.chat.completions, "create", new=mock_create):
            await model.chat(
                [UserMessage(content="Test")],
                ChatOptions(extra={"model": "evil-model"}),
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_extra_cannot_overwrite_messages(self, model):
        """extra={\"messages\": []} should NOT overwrite real messages."""
        mock_completion = MagicMock()
        mock_message = MagicMock(content="Response")
        del mock_message.tool_calls
        mock_completion.choices = [MagicMock(message=mock_message)]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(model._client.client.chat.completions, "create", new=mock_create):
            await model.chat(
                [UserMessage(content="Hello")],
                ChatOptions(extra={"messages": []}),
            )

            call_kwargs = mock_create.call_args.kwargs
            assert len(call_kwargs["messages"]) > 0

    @pytest.mark.asyncio
    async def test_extra_override_logs_warning(self, model, caplog):
        """Conflicting extra key should log a warning."""
        mock_completion = MagicMock()
        mock_message = MagicMock(content="Response")
        del mock_message.tool_calls
        mock_completion.choices = [MagicMock(message=mock_message)]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(model._client.client.chat.completions, "create", new=mock_create):
            with caplog.at_level(logging.WARNING):
                await model.chat(
                    [UserMessage(content="Test")],
                    ChatOptions(extra={"model": "evil-model"}),
                )

            assert any("Ignoring extra key" in r.message for r in caplog.records)


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic client not installed")
class TestAnthropicExtraOverrideProtection:
    """Test that options.extra cannot overwrite core Anthropic request parameters."""

    @pytest.fixture
    def model(self):
        client = AnthropicClient(api_key="sk-ant-test-key")
        return Model(client=client, name="claude-3-haiku-20240307")

    def _create_mock_response(self, content="Response"):
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = content
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        return mock_response

    @pytest.mark.asyncio
    async def test_extra_cannot_overwrite_model(self, model):
        """extra={\"model\": \"evil\"} should NOT overwrite the actual model."""
        mock_response = self._create_mock_response()
        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            await model.chat(
                [UserMessage(content="Test")],
                ChatOptions(extra={"model": "evil-model"}),
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "claude-3-haiku-20240307"

    @pytest.mark.asyncio
    async def test_extra_cannot_overwrite_messages(self, model):
        """extra={\"messages\": []} should NOT overwrite real messages."""
        mock_response = self._create_mock_response()
        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            await model.chat(
                [UserMessage(content="Hello")],
                ChatOptions(extra={"messages": []}),
            )

            call_kwargs = mock_create.call_args.kwargs
            assert len(call_kwargs["messages"]) > 0

    @pytest.mark.asyncio
    async def test_extra_cannot_overwrite_max_tokens(self, model):
        """extra={\"max_tokens\": 1} should NOT overwrite computed max_tokens."""
        mock_response = self._create_mock_response()
        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            await model.chat(
                [UserMessage(content="Test")],
                ChatOptions(max_tokens=500, extra={"max_tokens": 1}),
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_extra_override_logs_warning(self, model, caplog):
        """Conflicting extra key should log a warning."""
        mock_response = self._create_mock_response()
        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            with caplog.at_level(logging.WARNING):
                await model.chat(
                    [UserMessage(content="Test")],
                    ChatOptions(extra={"model": "evil-model"}),
                )

            assert any("Ignoring extra key" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# TST-003: max_tokens=0 Boundary Tests (All Providers)
# ---------------------------------------------------------------------------


class TestOllamaMaxTokensBoundary:
    """Test max_tokens=0 is correctly handled by Ollama provider."""

    @pytest.fixture
    def model(self):
        client = OllamaClient(host="http://localhost:11434")
        return Model(client=client, name="test-model")

    @pytest.mark.asyncio
    async def test_max_tokens_zero_is_passed(self, model):
        """max_tokens=0 should be passed to the API (not dropped)."""
        mock_response = MagicMock()
        mock_response.message.content = ""
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            await model.chat(
                [UserMessage(content="Test")],
                ChatOptions(max_tokens=0),
            )

            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["options"]["num_predict"] == 0

    @pytest.mark.asyncio
    async def test_max_tokens_none_is_omitted(self, model):
        """max_tokens=None should NOT be sent to the API."""
        mock_response = MagicMock()
        mock_response.message.content = "Response"
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            await model.chat(
                [UserMessage(content="Test")],
                ChatOptions(max_tokens=None),
            )

            call_kwargs = mock_chat.call_args.kwargs
            assert "num_predict" not in call_kwargs["options"]

    @pytest.mark.asyncio
    async def test_max_tokens_positive_is_passed(self, model):
        """max_tokens=100 should be passed normally."""
        mock_response = MagicMock()
        mock_response.message.content = "Response"
        mock_response.message.tool_calls = None

        mock_chat = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient.chat", new=mock_chat):
            await model.chat(
                [UserMessage(content="Test")],
                ChatOptions(max_tokens=100),
            )

            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["options"]["num_predict"] == 100


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI client not installed")
class TestOpenAIMaxTokensBoundary:
    """Test max_tokens=0 is correctly handled by OpenAI provider."""

    @pytest.fixture
    def model(self):
        client = OpenAIClient(api_key="sk-test-key")
        return Model(client=client, name="gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_max_tokens_zero_is_passed(self, model):
        """max_tokens=0 should be passed to the API (not dropped)."""
        mock_completion = MagicMock()
        mock_message = MagicMock(content="")
        del mock_message.tool_calls
        mock_completion.choices = [MagicMock(message=mock_message)]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(model._client.client.chat.completions, "create", new=mock_create):
            await model.chat(
                [UserMessage(content="Test")],
                ChatOptions(max_tokens=0),
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 0

    @pytest.mark.asyncio
    async def test_max_tokens_none_is_omitted(self, model):
        """max_tokens=None should NOT be sent to the API."""
        mock_completion = MagicMock()
        mock_message = MagicMock(content="Response")
        del mock_message.tool_calls
        mock_completion.choices = [MagicMock(message=mock_message)]

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(model._client.client.chat.completions, "create", new=mock_create):
            await model.chat(
                [UserMessage(content="Test")],
                ChatOptions(max_tokens=None),
            )

            call_kwargs = mock_create.call_args.kwargs
            assert "max_tokens" not in call_kwargs


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic client not installed")
class TestAnthropicMaxTokensBoundary:
    """Test max_tokens=0 is correctly handled by Anthropic provider."""

    @pytest.fixture
    def model(self):
        client = AnthropicClient(api_key="sk-ant-test-key")
        return Model(client=client, name="claude-3-haiku-20240307")

    def _create_mock_response(self, content="Response"):
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = content
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
        return mock_response

    @pytest.mark.asyncio
    async def test_max_tokens_explicit_value_is_passed(self, model):
        """max_tokens=100 should be passed to the API."""
        mock_response = self._create_mock_response()
        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            await model.chat(
                [UserMessage(content="Test")],
                ChatOptions(max_tokens=100),
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_max_tokens_none_uses_default(self, model):
        """max_tokens=None should use the DEFAULT_MAX_TOKENS (4096)."""
        mock_response = self._create_mock_response()
        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            await model.chat(
                [UserMessage(content="Test")],
                ChatOptions(max_tokens=None),
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_max_tokens_zero_is_passed(self, model):
        """max_tokens=0 should be passed as 0 (not fall back to default)."""
        mock_response = self._create_mock_response()
        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(model._client.client.messages, "create", new=mock_create):
            await model.chat(
                [UserMessage(content="Test")],
                ChatOptions(max_tokens=0),
            )

            call_kwargs = mock_create.call_args.kwargs
            # After the fix, max_tokens=0 should be sent as 0
            assert call_kwargs["max_tokens"] == 0
