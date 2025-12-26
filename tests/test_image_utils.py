"""
Tests for image utility functions.

Tests cover:
- fetch_image_as_base64: URL fetching with various scenarios
- strip_base64_prefix: Data URI prefix handling
- add_base64_prefix: Data URI creation
"""

import base64
import pytest
from unittest.mock import AsyncMock, Mock, patch
import httpx

from casual_llm.utils.image import (
    fetch_image_as_base64,
    strip_base64_prefix,
    add_base64_prefix,
    ImageFetchError,
    DEFAULT_TIMEOUT,
    MAX_IMAGE_SIZE,
)


class TestStripBase64Prefix:
    """Tests for strip_base64_prefix function."""

    def test_strip_prefix_with_png(self):
        """Test stripping data URI prefix from PNG."""
        data_uri = "data:image/png;base64,abc123xyz"
        result = strip_base64_prefix(data_uri)
        assert result == "abc123xyz"

    def test_strip_prefix_with_jpeg(self):
        """Test stripping data URI prefix from JPEG."""
        data_uri = "data:image/jpeg;base64,def456uvw"
        result = strip_base64_prefix(data_uri)
        assert result == "def456uvw"

    def test_no_prefix_returns_unchanged(self):
        """Test that data without prefix is returned unchanged."""
        raw_data = "abc123xyz"
        result = strip_base64_prefix(raw_data)
        assert result == "abc123xyz"

    def test_empty_string(self):
        """Test handling of empty string."""
        result = strip_base64_prefix("")
        assert result == ""

    def test_prefix_without_base64_marker(self):
        """Test data URI without base64 marker."""
        data_uri = "data:image/png,abc123"
        result = strip_base64_prefix(data_uri)
        assert result == "data:image/png,abc123"  # Should return unchanged


class TestAddBase64Prefix:
    """Tests for add_base64_prefix function."""

    def test_add_prefix_default_png(self):
        """Test adding prefix with default PNG media type."""
        result = add_base64_prefix("abc123")
        assert result == "data:image/png;base64,abc123"

    def test_add_prefix_jpeg(self):
        """Test adding prefix with JPEG media type."""
        result = add_base64_prefix("xyz789", "image/jpeg")
        assert result == "data:image/jpeg;base64,xyz789"

    def test_add_prefix_webp(self):
        """Test adding prefix with WebP media type."""
        result = add_base64_prefix("webpdata", "image/webp")
        assert result == "data:image/webp;base64,webpdata"

    def test_add_prefix_empty_data(self):
        """Test adding prefix to empty data."""
        result = add_base64_prefix("")
        assert result == "data:image/png;base64,"


class TestFetchImageAsBase64:
    """Tests for fetch_image_as_base64 function."""

    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        """Test successful image fetch."""
        # Create mock response
        test_image_data = b"fake image data"
        mock_response = Mock()
        mock_response.content = test_image_data
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.raise_for_status = Mock()

        # Mock httpx.AsyncClient
        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            base64_data, media_type = await fetch_image_as_base64(
                "https://example.com/image.jpg"
            )

            # Verify result
            expected_b64 = base64.b64encode(test_image_data).decode("ascii")
            assert base64_data == expected_b64
            assert media_type == "image/jpeg"

            # Verify httpx was called with correct parameters
            mock_client.get.assert_called_once_with("https://example.com/image.jpg")

    @pytest.mark.asyncio
    async def test_fetch_with_content_type_charset(self):
        """Test fetch with content-type that includes charset."""
        test_image_data = b"fake image data"
        mock_response = Mock()
        mock_response.content = test_image_data
        mock_response.headers = {"content-type": "image/png; charset=utf-8"}
        mock_response.raise_for_status = Mock()

        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            base64_data, media_type = await fetch_image_as_base64(
                "https://example.com/image.png"
            )

            assert media_type == "image/png"

    @pytest.mark.asyncio
    async def test_fetch_with_invalid_content_type(self):
        """Test fetch with non-image content-type falls back to image/jpeg."""
        test_image_data = b"fake image data"
        mock_response = Mock()
        mock_response.content = test_image_data
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = Mock()

        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            base64_data, media_type = await fetch_image_as_base64(
                "https://example.com/image.jpg"
            )

            assert media_type == "image/jpeg"  # Fallback

    @pytest.mark.asyncio
    async def test_fetch_with_missing_content_type(self):
        """Test fetch with missing content-type header."""
        test_image_data = b"fake image data"
        mock_response = Mock()
        mock_response.content = test_image_data
        mock_response.headers = {}
        mock_response.raise_for_status = Mock()

        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            base64_data, media_type = await fetch_image_as_base64(
                "https://example.com/image.jpg"
            )

            assert media_type == "image/jpeg"  # Default

    @pytest.mark.asyncio
    async def test_fetch_http_error(self):
        """Test fetch with HTTP error (404, 403, etc.)."""
        mock_response = Mock()
        mock_response.status_code = 404

        async def mock_get(*args, **kwargs):
            raise httpx.HTTPStatusError(
                "404 Not Found", request=Mock(), response=mock_response
            )

        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(ImageFetchError) as exc_info:
                await fetch_image_as_base64("https://example.com/missing.jpg")

            assert "HTTP error" in str(exc_info.value)
            assert "404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_timeout(self):
        """Test fetch with timeout."""
        async def mock_get(*args, **kwargs):
            raise httpx.TimeoutException("Request timed out")

        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(ImageFetchError) as exc_info:
                await fetch_image_as_base64("https://example.com/slow.jpg")

            assert "Timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_request_error(self):
        """Test fetch with general request error."""
        async def mock_get(*args, **kwargs):
            raise httpx.RequestError("Connection failed")

        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(ImageFetchError) as exc_info:
                await fetch_image_as_base64("https://example.com/error.jpg")

            assert "Error fetching image" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_image_too_large_from_header(self):
        """Test fetch with image size exceeding limit (from content-length header)."""
        mock_response = Mock()
        mock_response.headers = {
            "content-length": str(MAX_IMAGE_SIZE + 1000),
            "content-type": "image/jpeg",
        }
        mock_response.raise_for_status = Mock()

        async def mock_get(*args, **kwargs):
            return mock_response

        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(ImageFetchError) as exc_info:
                await fetch_image_as_base64("https://example.com/huge.jpg")

            assert "exceeds maximum allowed size" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_image_too_large_actual_content(self):
        """Test fetch with actual content size exceeding limit."""
        # Create image data that exceeds MAX_IMAGE_SIZE
        large_data = b"x" * (MAX_IMAGE_SIZE + 1000)
        mock_response = Mock()
        mock_response.content = large_data
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.raise_for_status = Mock()

        async def mock_get(*args, **kwargs):
            return mock_response

        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(ImageFetchError) as exc_info:
                await fetch_image_as_base64("https://example.com/huge.jpg")

            assert "exceeds maximum allowed size" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_with_custom_timeout(self):
        """Test fetch with custom timeout parameter."""
        test_image_data = b"fake image data"
        mock_response = Mock()
        mock_response.content = test_image_data
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.raise_for_status = Mock()

        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            await fetch_image_as_base64("https://example.com/image.jpg", timeout=60.0)

            # Verify AsyncClient was created with custom timeout
            mock_client_class.assert_called_once()
            call_kwargs = mock_client_class.call_args[1]
            assert call_kwargs["timeout"] == 60.0

    @pytest.mark.asyncio
    async def test_fetch_with_custom_max_size(self):
        """Test fetch with custom max_size parameter."""
        # Test 1: Should succeed with large max_size
        test_image_data = b"x" * 1000
        mock_response = Mock()
        mock_response.content = test_image_data
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.raise_for_status = Mock()

        async def mock_get(*args, **kwargs):
            return mock_response

        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            # Should succeed with large max_size
            await fetch_image_as_base64(
                "https://example.com/image.jpg", max_size=2000
            )

        # Test 2: Should fail with small max_size
        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(ImageFetchError):
                await fetch_image_as_base64(
                    "https://example.com/image.jpg", max_size=500
                )

    @pytest.mark.asyncio
    async def test_fetch_uses_http2(self):
        """Test that fetch uses HTTP/2."""
        test_image_data = b"fake image data"
        mock_response = Mock()
        mock_response.content = test_image_data
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.raise_for_status = Mock()

        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            await fetch_image_as_base64("https://example.com/image.jpg")

            # Verify AsyncClient was created with http2=True
            mock_client_class.assert_called_once()
            call_kwargs = mock_client_class.call_args[1]
            assert call_kwargs["http2"] is True

    @pytest.mark.asyncio
    async def test_fetch_uses_user_agent(self):
        """Test that fetch uses proper User-Agent header."""
        test_image_data = b"fake image data"
        mock_response = Mock()
        mock_response.content = test_image_data
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.raise_for_status = Mock()

        with patch("casual_llm.utils.image.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            await fetch_image_as_base64("https://example.com/image.jpg")

            # Verify AsyncClient was created with User-Agent header
            mock_client_class.assert_called_once()
            call_kwargs = mock_client_class.call_args[1]
            assert "User-Agent" in call_kwargs["headers"]
            assert "Mozilla" in call_kwargs["headers"]["User-Agent"]

    @pytest.mark.asyncio
    async def test_httpx_not_available(self):
        """Test error when httpx is not installed."""
        with patch("casual_llm.utils.image.HTTPX_AVAILABLE", False):
            with pytest.raises(ImageFetchError) as exc_info:
                await fetch_image_as_base64("https://example.com/image.jpg")

            assert "httpx is required" in str(exc_info.value)
            assert "httpx[http2]" in str(exc_info.value)
