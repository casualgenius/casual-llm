"""
Image utilities for fetching and processing images.

Provides async utilities for downloading images from URLs and converting
them to base64 format for use in multimodal LLM messages.
"""

import base64
import ipaddress
import logging
from urllib.parse import urlparse

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImageFetchError(Exception):
    """Raised when image fetching fails."""

    pass


# Default timeout for image fetching (in seconds)
DEFAULT_TIMEOUT = 30.0

# Maximum image size in bytes (10 MB)
MAX_IMAGE_SIZE = 10 * 1024 * 1024

# Allowed URL schemes
_ALLOWED_SCHEMES = {"http", "https"}

# Allowed image MIME types
_ALLOWED_MEDIA_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/svg+xml",
    "image/bmp",
    "image/tiff",
}


def _validate_url(url: str) -> None:
    """Validate that a URL is safe to fetch.

    Blocks private/internal IP addresses to prevent SSRF attacks.

    Args:
        url: The URL to validate.

    Raises:
        ImageFetchError: If the URL is not safe to fetch.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        raise ImageFetchError(f"Invalid URL: {url}")

    # Check scheme
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ImageFetchError(
            f"URL scheme {parsed.scheme!r} is not allowed. "
            f"Only {', '.join(sorted(_ALLOWED_SCHEMES))} are supported."
        )

    # Check for empty or missing hostname
    hostname = parsed.hostname
    if not hostname:
        raise ImageFetchError(f"URL has no hostname: {url}")

    # Block private/internal IP addresses
    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            raise ImageFetchError(
                f"URL points to a private/internal address ({hostname}), which is not allowed."
            )
    except ValueError:
        # Not an IP address literal — it's a hostname, which is fine.
        # DNS resolution to private IPs is harder to block here but
        # we at least prevent direct IP-based SSRF.
        pass


async def fetch_image_as_base64(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    max_size: int = MAX_IMAGE_SIZE,
) -> tuple[str, str]:
    """Fetch an image from a URL and return it as base64-encoded data.

    Downloads the image from the given URL and returns the base64-encoded
    content along with the detected media type.

    Args:
        url: The URL of the image to fetch.
        timeout: Request timeout in seconds. Defaults to 30 seconds.
        max_size: Maximum allowed image size in bytes. Defaults to 10 MB.

    Returns:
        A tuple of (base64_data, media_type) where:
            - base64_data: The raw base64-encoded image data (no data: prefix)
            - media_type: The MIME type of the image (e.g., "image/jpeg")

    Raises:
        ImageFetchError: If the image cannot be fetched, is too large,
            or if httpx is not installed.

    Example:
        >>> base64_data, media_type = await fetch_image_as_base64("https://example.com/image.jpg")
        >>> print(media_type)
        image/jpeg
    """
    if not HTTPX_AVAILABLE:
        raise ImageFetchError(
            "httpx is required for fetching images from URLs. "
            "Install it with: pip install 'httpx[http2]'"
        )

    # Validate URL before fetching (SSRF prevention)
    _validate_url(url)

    try:
        # Use a browser-like User-Agent and HTTP/2 to avoid being blocked by sites like Wikipedia
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        async with httpx.AsyncClient(
            timeout=timeout,
            headers=headers,
            http2=True,
            follow_redirects=False,
        ) as client:
            response = await client.get(url)

            # Handle redirects manually to validate targets
            redirect_count = 0
            while response.is_redirect and redirect_count < 5:
                redirect_count += 1
                redirect_url = str(response.next_request.url) if response.next_request else None
                if not redirect_url:
                    break
                _validate_url(redirect_url)
                response = await client.get(redirect_url)

            if response.is_redirect:
                raise ImageFetchError("Too many redirects when fetching image")

            response.raise_for_status()

            # Check content length if available (before downloading body)
            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    declared_size = int(content_length)
                    if declared_size > max_size:
                        raise ImageFetchError(
                            f"Image size ({declared_size} bytes) exceeds "
                            f"maximum allowed size ({max_size} bytes)"
                        )
                except ValueError:
                    pass  # Malformed content-length, skip check

            # Read content and check actual size
            content = response.content
            if len(content) > max_size:
                raise ImageFetchError(
                    f"Image size ({len(content)} bytes) exceeds "
                    f"maximum allowed size ({max_size} bytes)"
                )

            # Extract media type from Content-Type header
            content_type = response.headers.get("content-type", "image/jpeg")
            # Remove any charset or boundary info (e.g., "image/jpeg; charset=utf-8")
            media_type = content_type.split(";")[0].strip()

            # Validate that it looks like an image type
            if not media_type.startswith("image/"):
                media_type = "image/jpeg"

            # Encode to base64
            base64_data = base64.b64encode(content).decode("ascii")

            return base64_data, media_type

    except ImageFetchError:
        raise
    except httpx.HTTPStatusError as e:
        raise ImageFetchError(f"HTTP error fetching image: {e.response.status_code}") from e
    except httpx.TimeoutException:
        raise ImageFetchError("Timeout fetching image") from None
    except httpx.RequestError as e:
        raise ImageFetchError(f"Error fetching image: {e}") from e


def strip_base64_prefix(data: str) -> str:
    """Strip the data URI prefix from a base64-encoded string.

    Removes the 'data:<media_type>;base64,' prefix if present,
    returning only the raw base64 data.

    Args:
        data: A base64 string, optionally with a data URI prefix.

    Returns:
        The raw base64 data without any prefix.

    Example:
        >>> strip_base64_prefix("data:image/png;base64,abc123")
        'abc123'
        >>> strip_base64_prefix("abc123")
        'abc123'
    """
    # Check for data URI format: data:<media_type>;base64,<data>
    if data.startswith("data:") and ";base64," in data:
        # Split on ";base64," and return the data portion
        return data.split(";base64,", 1)[1]
    return data


def add_base64_prefix(base64_data: str, media_type: str = "image/png") -> str:
    """Add a data URI prefix to raw base64 data.

    Creates a complete data URI by prepending the appropriate prefix
    to raw base64-encoded data.

    Args:
        base64_data: The raw base64-encoded data (without prefix).
        media_type: The MIME type of the data. Defaults to "image/png".

    Returns:
        A complete data URI string.

    Example:
        >>> add_base64_prefix("abc123", "image/png")
        'data:image/png;base64,abc123'
        >>> add_base64_prefix("xyz789", "image/jpeg")
        'data:image/jpeg;base64,xyz789'
    """
    return f"data:{media_type};base64,{base64_data}"
