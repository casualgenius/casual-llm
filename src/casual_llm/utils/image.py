"""
Image utilities for fetching and processing images.

Provides async utilities for downloading images from URLs and converting
them to base64 format for use in multimodal LLM messages.
"""

import base64

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class ImageFetchError(Exception):
    """Raised when image fetching fails."""

    pass


# Default timeout for image fetching (in seconds)
DEFAULT_TIMEOUT = 30.0

# Maximum image size in bytes (10 MB)
MAX_IMAGE_SIZE = 10 * 1024 * 1024


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
        >>> base64_data, media_type = await fetch_image_as_base64(
        ...     "https://example.com/image.jpg"
        ... )
        >>> print(media_type)
        image/jpeg
    """
    if not HTTPX_AVAILABLE:
        raise ImageFetchError(
            "httpx is required for fetching images from URLs. " "Install it with: pip install httpx"
        )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Check content length if available
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > max_size:
                raise ImageFetchError(
                    f"Image size ({int(content_length)} bytes) exceeds "
                    f"maximum allowed size ({max_size} bytes)"
                )

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
                # Default to image/jpeg if content-type doesn't indicate an image
                media_type = "image/jpeg"

            # Encode to base64
            base64_data = base64.b64encode(content).decode("ascii")

            return base64_data, media_type

    except httpx.HTTPStatusError as e:
        raise ImageFetchError(
            f"HTTP error fetching image from {url}: {e.response.status_code}"
        ) from e
    except httpx.TimeoutException as e:
        raise ImageFetchError(f"Timeout fetching image from {url}") from e
    except httpx.RequestError as e:
        raise ImageFetchError(f"Error fetching image from {url}: {e}") from e


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
