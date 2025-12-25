"""Cross-provider vision compatibility tests.

This test suite verifies that the same ImageContent and multimodal messages
work consistently across all providers (OpenAI, Anthropic, Google, Ollama).
"""

import pytest
from unittest.mock import AsyncMock, patch

from casual_llm.messages import (
    UserMessage,
    AssistantMessage,
    SystemMessage,
    TextContent,
    ImageContent,
)
from casual_llm.message_converters.openai import convert_messages_to_openai
from casual_llm.message_converters.anthropic import convert_messages_to_anthropic
from casual_llm.message_converters.google import convert_messages_to_google
from casual_llm.message_converters.ollama import convert_messages_to_ollama


# Small 1x1 pixel PNG in base64 for testing
TEST_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


class TestMultimodalMessageCreation:
    """Tests for creating multimodal messages that work across all providers."""

    def test_url_image_content(self):
        """Test ImageContent with URL source."""
        image = ImageContent(
            type="image",
            source="https://example.com/image.jpg",
            media_type="image/jpeg",
        )

        assert image.type == "image"
        assert image.source == "https://example.com/image.jpg"
        assert image.media_type == "image/jpeg"

    def test_base64_image_content(self):
        """Test ImageContent with base64 dict source."""
        image = ImageContent(
            type="image",
            source={"type": "base64", "data": TEST_IMAGE_BASE64},
            media_type="image/png",
        )

        assert image.type == "image"
        assert image.source["type"] == "base64"
        assert image.source["data"] == TEST_IMAGE_BASE64
        assert image.media_type == "image/png"

    def test_data_uri_image_content(self):
        """Test ImageContent with data URI source."""
        data_uri = f"data:image/png;base64,{TEST_IMAGE_BASE64}"
        image = ImageContent(
            type="image",
            source=data_uri,
            media_type="image/png",
        )

        assert image.type == "image"
        assert image.source == data_uri

    def test_user_message_with_multimodal_content(self):
        """Test UserMessage with mixed text and image content."""
        msg = UserMessage(
            content=[
                TextContent(type="text", text="What is in this image?"),
                ImageContent(
                    type="image",
                    source="https://example.com/photo.jpg",
                    media_type="image/jpeg",
                ),
            ]
        )

        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextContent)
        assert isinstance(msg.content[1], ImageContent)

    def test_user_message_backward_compatibility(self):
        """Test that UserMessage still accepts plain string content."""
        msg = UserMessage(content="Plain text message")

        assert msg.role == "user"
        assert msg.content == "Plain text message"


class TestCrossProviderImageFormatConversion:
    """Tests that the same ImageContent produces correct provider-specific formats."""

    @pytest.fixture
    def url_image(self):
        """Common URL image for testing across providers."""
        return ImageContent(
            type="image",
            source="https://example.com/test-image.jpg",
            media_type="image/jpeg",
        )

    @pytest.fixture
    def base64_image(self):
        """Common base64 image for testing across providers."""
        return ImageContent(
            type="image",
            source={"type": "base64", "data": TEST_IMAGE_BASE64},
            media_type="image/png",
        )

    @pytest.fixture
    def data_uri_image(self):
        """Common data URI image for testing across providers."""
        return ImageContent(
            type="image",
            source=f"data:image/png;base64,{TEST_IMAGE_BASE64}",
            media_type="image/png",
        )

    def test_openai_format_with_url(self, url_image):
        """Test OpenAI converter produces correct format for URL images."""
        messages = [
            UserMessage(
                content=[
                    TextContent(type="text", text="Describe this:"),
                    url_image,
                ]
            )
        ]

        result = convert_messages_to_openai(messages)

        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "https://example.com/test-image.jpg"

    def test_openai_format_with_base64(self, base64_image):
        """Test OpenAI converter produces data URI for base64 images."""
        messages = [
            UserMessage(
                content=[
                    TextContent(type="text", text="Describe this:"),
                    base64_image,
                ]
            )
        ]

        result = convert_messages_to_openai(messages)

        content = result[0]["content"]
        assert content[1]["type"] == "image_url"
        # OpenAI expects data URI format
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert TEST_IMAGE_BASE64 in content[1]["image_url"]["url"]

    @pytest.mark.asyncio
    async def test_anthropic_format_with_base64(self, base64_image):
        """Test Anthropic converter produces correct format for base64 images."""
        messages = [
            UserMessage(
                content=[
                    TextContent(type="text", text="Describe this:"),
                    base64_image,
                ]
            )
        ]

        result, _ = await convert_messages_to_anthropic(messages)

        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[1]["type"] == "image"
        # Anthropic expects source dict with raw base64 (no data URI prefix)
        assert content[1]["source"]["type"] == "base64"
        assert content[1]["source"]["data"] == TEST_IMAGE_BASE64
        assert content[1]["source"]["media_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_anthropic_format_with_url(self, url_image):
        """Test Anthropic converter fetches URL and converts to base64."""
        messages = [
            UserMessage(
                content=[
                    TextContent(type="text", text="Describe this:"),
                    url_image,
                ]
            )
        ]

        with patch(
            "casual_llm.message_converters.anthropic.fetch_image_as_base64",
            new_callable=AsyncMock,
            return_value=("fetchedbase64data", "image/jpeg"),
        ):
            result, _ = await convert_messages_to_anthropic(messages)

        content = result[0]["content"]
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"
        assert content[1]["source"]["data"] == "fetchedbase64data"

    @pytest.mark.asyncio
    async def test_google_format_with_base64(self, base64_image):
        """Test Google converter produces parts format for base64 images."""
        messages = [
            UserMessage(
                content=[
                    TextContent(type="text", text="Describe this:"),
                    base64_image,
                ]
            )
        ]

        result, _ = await convert_messages_to_google(messages)

        parts = result[0]["parts"]
        assert isinstance(parts, list)
        assert parts[0]["text"] == "Describe this:"
        # Google expects inline_data format
        assert "inline_data" in parts[1]
        assert parts[1]["inline_data"]["mime_type"] == "image/png"
        assert parts[1]["inline_data"]["data"] == TEST_IMAGE_BASE64

    @pytest.mark.asyncio
    async def test_google_format_with_url(self, url_image):
        """Test Google converter fetches URL and converts to inline_data."""
        messages = [
            UserMessage(
                content=[
                    TextContent(type="text", text="Describe this:"),
                    url_image,
                ]
            )
        ]

        with patch(
            "casual_llm.message_converters.google.fetch_image_as_base64",
            new_callable=AsyncMock,
            return_value=("googlefetcheddata", "image/jpeg"),
        ):
            result, _ = await convert_messages_to_google(messages)

        parts = result[0]["parts"]
        assert parts[1]["inline_data"]["data"] == "googlefetcheddata"

    @pytest.mark.asyncio
    async def test_ollama_format_with_base64(self, base64_image):
        """Test Ollama converter produces images array for base64 images."""
        messages = [
            UserMessage(
                content=[
                    TextContent(type="text", text="Describe this:"),
                    base64_image,
                ]
            )
        ]

        result = await convert_messages_to_ollama(messages)

        assert result[0]["content"] == "Describe this:"
        # Ollama expects images array with raw base64 strings
        assert "images" in result[0]
        assert result[0]["images"] == [TEST_IMAGE_BASE64]

    @pytest.mark.asyncio
    async def test_ollama_format_with_url(self, url_image):
        """Test Ollama converter fetches URL and adds to images array."""
        messages = [
            UserMessage(
                content=[
                    TextContent(type="text", text="Describe this:"),
                    url_image,
                ]
            )
        ]

        with patch(
            "casual_llm.message_converters.ollama.fetch_image_as_base64",
            new_callable=AsyncMock,
            return_value=("ollamafetcheddata", "image/jpeg"),
        ):
            result = await convert_messages_to_ollama(messages)

        assert result[0]["images"] == ["ollamafetcheddata"]


class TestCrossProviderConversationCompatibility:
    """Tests that the same conversation works across all providers."""

    @pytest.fixture
    def vision_conversation(self):
        """A multi-turn conversation with vision content."""
        return [
            SystemMessage(content="You are a helpful assistant that describes images."),
            UserMessage(
                content=[
                    TextContent(type="text", text="What animal is in this image?"),
                    ImageContent(
                        type="image",
                        source={"type": "base64", "data": TEST_IMAGE_BASE64},
                        media_type="image/jpeg",
                    ),
                ]
            ),
            AssistantMessage(content="This appears to be a cat."),
            UserMessage(content="What color is it?"),
        ]

    def test_openai_handles_vision_conversation(self, vision_conversation):
        """Test OpenAI converter handles full vision conversation."""
        result = convert_messages_to_openai(vision_conversation)

        # OpenAI includes system message in the array
        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert isinstance(result[1]["content"], list)  # Multimodal content
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"
        assert result[3]["content"] == "What color is it?"  # Plain text

    @pytest.mark.asyncio
    async def test_anthropic_handles_vision_conversation(self, vision_conversation):
        """Test Anthropic converter handles full vision conversation."""
        result, system_prompt = await convert_messages_to_anthropic(vision_conversation)

        # Anthropic extracts system message separately
        assert system_prompt == "You are a helpful assistant that describes images."
        assert len(result) == 3  # No system message in array
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)  # Multimodal content
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"

    @pytest.mark.asyncio
    async def test_google_handles_vision_conversation(self, vision_conversation):
        """Test Google converter handles full vision conversation."""
        result, system_instruction = await convert_messages_to_google(vision_conversation)

        # Google extracts system message as system_instruction
        assert system_instruction == "You are a helpful assistant that describes images."
        assert len(result) == 3  # No system message in array
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "model"  # Google uses "model" instead of "assistant"
        assert result[2]["role"] == "user"

    @pytest.mark.asyncio
    async def test_ollama_handles_vision_conversation(self, vision_conversation):
        """Test Ollama converter handles full vision conversation."""
        result = await convert_messages_to_ollama(vision_conversation)

        # Ollama keeps system message in array
        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert "images" in result[1]  # Has images array
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"


class TestDataURIPrefixHandling:
    """Tests that data URI prefix is correctly added/stripped per provider."""

    @pytest.fixture
    def data_uri_source(self):
        """Image with data URI source."""
        return ImageContent(
            type="image",
            source=f"data:image/png;base64,{TEST_IMAGE_BASE64}",
            media_type="image/png",
        )

    def test_openai_preserves_data_uri(self):
        """OpenAI should use data URI format for base64 images."""
        messages = [
            UserMessage(
                content=[
                    ImageContent(
                        type="image",
                        source={"type": "base64", "data": TEST_IMAGE_BASE64},
                        media_type="image/png",
                    ),
                ]
            )
        ]

        result = convert_messages_to_openai(messages)

        url = result[0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_anthropic_strips_data_uri_prefix(self, data_uri_source):
        """Anthropic should strip data URI prefix from source."""
        messages = [UserMessage(content=[data_uri_source])]

        result, _ = await convert_messages_to_anthropic(messages)

        # Anthropic expects raw base64, not data URI
        data = result[0]["content"][0]["source"]["data"]
        assert not data.startswith("data:")
        assert data == TEST_IMAGE_BASE64

    @pytest.mark.asyncio
    async def test_google_strips_data_uri_prefix(self, data_uri_source):
        """Google should strip data URI prefix from inline_data."""
        messages = [UserMessage(content=[data_uri_source])]

        result, _ = await convert_messages_to_google(messages)

        # Google expects raw base64, not data URI
        data = result[0]["parts"][0]["inline_data"]["data"]
        assert not data.startswith("data:")
        assert data == TEST_IMAGE_BASE64

    @pytest.mark.asyncio
    async def test_ollama_strips_data_uri_prefix(self, data_uri_source):
        """Ollama should strip data URI prefix from images array."""
        messages = [UserMessage(content=[data_uri_source])]

        result = await convert_messages_to_ollama(messages)

        # Ollama expects raw base64 in images array
        image_data = result[0]["images"][0]
        assert not image_data.startswith("data:")
        assert image_data == TEST_IMAGE_BASE64


class TestMultipleImagesSupport:
    """Tests that multiple images in a single message work across providers."""

    @pytest.fixture
    def multi_image_message(self):
        """Message with multiple images."""
        return UserMessage(
            content=[
                TextContent(type="text", text="Compare these two images:"),
                ImageContent(
                    type="image",
                    source={"type": "base64", "data": "image1base64"},
                    media_type="image/jpeg",
                ),
                ImageContent(
                    type="image",
                    source={"type": "base64", "data": "image2base64"},
                    media_type="image/jpeg",
                ),
            ]
        )

    def test_openai_multiple_images(self, multi_image_message):
        """Test OpenAI handles multiple images in one message."""
        result = convert_messages_to_openai([multi_image_message])

        content = result[0]["content"]
        assert len(content) == 3
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_anthropic_multiple_images(self, multi_image_message):
        """Test Anthropic handles multiple images in one message."""
        result, _ = await convert_messages_to_anthropic([multi_image_message])

        content = result[0]["content"]
        assert len(content) == 3
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"
        assert content[2]["type"] == "image"

    @pytest.mark.asyncio
    async def test_google_multiple_images(self, multi_image_message):
        """Test Google handles multiple images in one message."""
        result, _ = await convert_messages_to_google([multi_image_message])

        parts = result[0]["parts"]
        assert len(parts) == 3
        assert "text" in parts[0]
        assert "inline_data" in parts[1]
        assert "inline_data" in parts[2]

    @pytest.mark.asyncio
    async def test_ollama_multiple_images(self, multi_image_message):
        """Test Ollama handles multiple images in one message."""
        result = await convert_messages_to_ollama([multi_image_message])

        assert result[0]["content"] == "Compare these two images:"
        assert len(result[0]["images"]) == 2
        assert result[0]["images"][0] == "image1base64"
        assert result[0]["images"][1] == "image2base64"


class TestEmptyAndEdgeCases:
    """Tests for edge cases and empty content handling."""

    def test_empty_messages_list(self):
        """Test handling of empty messages list."""
        assert convert_messages_to_openai([]) == []

    @pytest.mark.asyncio
    async def test_anthropic_empty_messages(self):
        """Test Anthropic handles empty messages."""
        result, system = await convert_messages_to_anthropic([])
        assert result == []
        assert system is None

    @pytest.mark.asyncio
    async def test_google_empty_messages(self):
        """Test Google handles empty messages."""
        result, system = await convert_messages_to_google([])
        assert result == []
        assert system is None

    @pytest.mark.asyncio
    async def test_ollama_empty_messages(self):
        """Test Ollama handles empty messages."""
        result = await convert_messages_to_ollama([])
        assert result == []

    def test_text_only_message_in_array_format(self):
        """Test message with only TextContent (no images)."""
        messages = [
            UserMessage(
                content=[
                    TextContent(type="text", text="Just text, no images"),
                ]
            )
        ]

        result = convert_messages_to_openai(messages)

        content = result[0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Just text, no images"

    @pytest.mark.asyncio
    async def test_none_content_handling_anthropic(self):
        """Test Anthropic handles None content gracefully."""
        messages = [UserMessage(content=None)]

        result, _ = await convert_messages_to_anthropic(messages)

        # Anthropic returns empty string for None content
        assert result[0]["content"] == ""

    @pytest.mark.asyncio
    async def test_none_content_handling_google(self):
        """Test Google handles None content gracefully."""
        messages = [UserMessage(content=None)]

        result, _ = await convert_messages_to_google(messages)

        # Google returns empty parts list for None content
        assert result[0]["parts"] == []

    @pytest.mark.asyncio
    async def test_none_content_handling_ollama(self):
        """Test Ollama handles None content gracefully."""
        messages = [UserMessage(content=None)]

        result = await convert_messages_to_ollama(messages)

        assert result[0]["content"] == ""
        assert "images" not in result[0]


class TestMediaTypeVariations:
    """Tests for different image media types across providers."""

    @pytest.mark.parametrize(
        "media_type",
        [
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
        ],
    )
    def test_openai_media_types(self, media_type):
        """Test OpenAI handles various media types."""
        messages = [
            UserMessage(
                content=[
                    ImageContent(
                        type="image",
                        source={"type": "base64", "data": "testdata"},
                        media_type=media_type,
                    ),
                ]
            )
        ]

        result = convert_messages_to_openai(messages)

        url = result[0]["content"][0]["image_url"]["url"]
        assert f"data:{media_type};base64," in url

    @pytest.mark.parametrize(
        "media_type",
        [
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
        ],
    )
    @pytest.mark.asyncio
    async def test_anthropic_media_types(self, media_type):
        """Test Anthropic handles various media types."""
        messages = [
            UserMessage(
                content=[
                    ImageContent(
                        type="image",
                        source={"type": "base64", "data": "testdata"},
                        media_type=media_type,
                    ),
                ]
            )
        ]

        result, _ = await convert_messages_to_anthropic(messages)

        assert result[0]["content"][0]["source"]["media_type"] == media_type

    @pytest.mark.parametrize(
        "media_type",
        [
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
        ],
    )
    @pytest.mark.asyncio
    async def test_google_media_types(self, media_type):
        """Test Google handles various media types."""
        messages = [
            UserMessage(
                content=[
                    ImageContent(
                        type="image",
                        source={"type": "base64", "data": "testdata"},
                        media_type=media_type,
                    ),
                ]
            )
        ]

        result, _ = await convert_messages_to_google(messages)

        assert result[0]["parts"][0]["inline_data"]["mime_type"] == media_type


class TestDefaultMediaTypeInference:
    """Tests for default media type behavior."""

    def test_image_content_default_media_type(self):
        """Test ImageContent defaults to image/jpeg."""
        image = ImageContent(
            type="image",
            source="https://example.com/unknown-type",
        )

        assert image.media_type == "image/jpeg"

    @pytest.mark.asyncio
    async def test_url_fetch_overrides_default(self):
        """Test that fetched media type is used for URL images."""
        messages = [
            UserMessage(
                content=[
                    ImageContent(
                        type="image",
                        source="https://example.com/image.png",
                        # Uses default media_type="image/jpeg"
                    ),
                ]
            )
        ]

        with patch(
            "casual_llm.message_converters.anthropic.fetch_image_as_base64",
            new_callable=AsyncMock,
            return_value=("pngdata", "image/png"),
        ):
            result, _ = await convert_messages_to_anthropic(messages)

        # Should use fetched media type, not default
        assert result[0]["content"][0]["source"]["media_type"] == "image/png"
