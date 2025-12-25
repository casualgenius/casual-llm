"""Tests for Google vision support with Gemini models."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from casual_llm.messages import (
    UserMessage,
    AssistantMessage,
    TextContent,
    ImageContent,
)
from casual_llm.message_converters.google import (
    _convert_image_to_google,
    _convert_user_content_to_google_parts,
    convert_messages_to_google,
)


# Try to import Google provider - may not be available
try:
    from casual_llm.providers import GoogleProvider
    import google.generativeai  # noqa: F401 - Check if google module is installed

    GOOGLE_AVAILABLE = GoogleProvider is not None
except ImportError:
    GOOGLE_AVAILABLE = False


class TestImageContentConversion:
    """Tests for _convert_image_to_google function."""

    @pytest.mark.asyncio
    async def test_url_image_conversion(self):
        """Test URL image is fetched and converted to inline_data format."""
        image = ImageContent(
            type="image",
            source="https://example.com/image.jpg",
            media_type="image/jpeg",
        )

        with patch(
            "casual_llm.message_converters.google.fetch_image_as_base64",
            new=AsyncMock(return_value=("base64imagedata", "image/jpeg")),
        ):
            result = await _convert_image_to_google(image)

        assert "inline_data" in result
        assert result["inline_data"]["mime_type"] == "image/jpeg"
        assert result["inline_data"]["data"] == "base64imagedata"

    @pytest.mark.asyncio
    async def test_base64_image_conversion(self):
        """Test base64 image is converted to inline_data format."""
        image = ImageContent(
            type="image",
            source={"type": "base64", "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="},
            media_type="image/png",
        )

        result = await _convert_image_to_google(image)

        assert "inline_data" in result
        assert result["inline_data"]["mime_type"] == "image/png"
        assert result["inline_data"]["data"] == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    @pytest.mark.asyncio
    async def test_base64_image_jpeg_media_type(self):
        """Test base64 image with jpeg media type."""
        image = ImageContent(
            type="image",
            source={"type": "base64", "data": "base64encodeddata"},
            media_type="image/jpeg",
        )

        result = await _convert_image_to_google(image)

        assert result["inline_data"]["mime_type"] == "image/jpeg"
        assert result["inline_data"]["data"] == "base64encodeddata"

    @pytest.mark.asyncio
    async def test_base64_image_webp_media_type(self):
        """Test base64 image with webp media type."""
        image = ImageContent(
            type="image",
            source={"type": "base64", "data": "webpdata"},
            media_type="image/webp",
        )

        result = await _convert_image_to_google(image)

        assert result["inline_data"]["mime_type"] == "image/webp"
        assert result["inline_data"]["data"] == "webpdata"

    @pytest.mark.asyncio
    async def test_data_uri_image_conversion(self):
        """Test data URI image is converted to inline_data format."""
        image = ImageContent(
            type="image",
            source="data:image/png;base64,abc123data",
            media_type="image/png",
        )

        result = await _convert_image_to_google(image)

        assert "inline_data" in result
        assert result["inline_data"]["mime_type"] == "image/png"
        assert result["inline_data"]["data"] == "abc123data"

    @pytest.mark.asyncio
    async def test_url_uses_fetched_media_type_when_default(self):
        """Test URL image uses fetched media type when default is used."""
        image = ImageContent(
            type="image",
            source="https://example.com/image.webp",
            # Default media_type is image/jpeg
        )

        with patch(
            "casual_llm.message_converters.google.fetch_image_as_base64",
            new=AsyncMock(return_value=("webpdata", "image/webp")),
        ):
            result = await _convert_image_to_google(image)

        # Should use fetched media type since default was used
        assert result["inline_data"]["mime_type"] == "image/webp"


class TestUserContentConversion:
    """Tests for _convert_user_content_to_google_parts function."""

    @pytest.mark.asyncio
    async def test_string_content_to_parts(self):
        """Test that simple string content is wrapped in parts format."""
        result = await _convert_user_content_to_google_parts("Hello, world!")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == {"text": "Hello, world!"}

    @pytest.mark.asyncio
    async def test_none_content_returns_empty_list(self):
        """Test that None content returns empty parts list."""
        result = await _convert_user_content_to_google_parts(None)

        assert result == []

    @pytest.mark.asyncio
    async def test_text_only_multimodal_content(self):
        """Test multimodal content with text only."""
        content = [
            TextContent(type="text", text="What is in this image?"),
        ]

        result = await _convert_user_content_to_google_parts(content)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == {"text": "What is in this image?"}

    @pytest.mark.asyncio
    async def test_image_only_multimodal_content(self):
        """Test multimodal content with image only."""
        content = [
            ImageContent(
                type="image",
                source={"type": "base64", "data": "catimagedata"},
                media_type="image/jpeg",
            ),
        ]

        result = await _convert_user_content_to_google_parts(content)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "inline_data" in result[0]
        assert result[0]["inline_data"]["data"] == "catimagedata"

    @pytest.mark.asyncio
    async def test_mixed_text_and_image_content(self):
        """Test multimodal content with text and image."""
        content = [
            TextContent(type="text", text="Describe this image:"),
            ImageContent(
                type="image",
                source={"type": "base64", "data": "photodata"},
                media_type="image/png",
            ),
        ]

        result = await _convert_user_content_to_google_parts(content)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"text": "Describe this image:"}
        assert "inline_data" in result[1]
        assert result[1]["inline_data"]["mime_type"] == "image/png"
        assert result[1]["inline_data"]["data"] == "photodata"

    @pytest.mark.asyncio
    async def test_multiple_images_content(self):
        """Test multimodal content with multiple images."""
        content = [
            TextContent(type="text", text="Compare these two images:"),
            ImageContent(
                type="image",
                source={"type": "base64", "data": "image1data"},
                media_type="image/jpeg",
            ),
            ImageContent(
                type="image",
                source={"type": "base64", "data": "image2data"},
                media_type="image/jpeg",
            ),
        ]

        result = await _convert_user_content_to_google_parts(content)

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == {"text": "Compare these two images:"}
        assert result[1]["inline_data"]["data"] == "image1data"
        assert result[2]["inline_data"]["data"] == "image2data"


class TestMessageConversionWithVision:
    """Tests for convert_messages_to_google with vision content."""

    @pytest.mark.asyncio
    async def test_user_message_with_text_content(self):
        """Test converting user message with simple text."""
        messages = [UserMessage(content="Hello!")]

        result, system_instruction = await convert_messages_to_google(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"] == [{"text": "Hello!"}]
        assert system_instruction is None

    @pytest.mark.asyncio
    async def test_user_message_with_multimodal_content(self):
        """Test converting user message with multimodal content."""
        messages = [
            UserMessage(
                content=[
                    TextContent(type="text", text="What's in this image?"),
                    ImageContent(
                        type="image",
                        source={"type": "base64", "data": "imagedata"},
                        media_type="image/jpeg",
                    ),
                ]
            )
        ]

        result, system_instruction = await convert_messages_to_google(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["parts"], list)
        assert len(result[0]["parts"]) == 2
        assert result[0]["parts"][0] == {"text": "What's in this image?"}
        assert "inline_data" in result[0]["parts"][1]

    @pytest.mark.asyncio
    async def test_mixed_messages_with_vision(self):
        """Test converting conversation with vision and text messages."""
        messages = [
            UserMessage(
                content=[
                    TextContent(type="text", text="What is this?"),
                    ImageContent(
                        type="image",
                        source={"type": "base64", "data": "catdata"},
                    ),
                ]
            ),
            AssistantMessage(content="This is a photo of a cat."),
            UserMessage(content="What color is the cat?"),
        ]

        result, system_instruction = await convert_messages_to_google(messages)

        assert len(result) == 3
        # First message - multimodal user message
        assert result[0]["role"] == "user"
        assert len(result[0]["parts"]) == 2
        assert result[0]["parts"][0] == {"text": "What is this?"}
        assert "inline_data" in result[0]["parts"][1]
        # Second message - assistant (model) response
        assert result[1]["role"] == "model"
        assert result[1]["parts"] == [{"text": "This is a photo of a cat."}]
        # Third message - simple text follow-up
        assert result[2]["role"] == "user"
        assert result[2]["parts"] == [{"text": "What color is the cat?"}]

    @pytest.mark.asyncio
    async def test_url_image_in_message_is_fetched(self):
        """Test that URL images in messages are fetched and converted."""
        messages = [
            UserMessage(
                content=[
                    TextContent(type="text", text="What is this?"),
                    ImageContent(
                        type="image",
                        source="https://example.com/cat.jpg",
                        media_type="image/jpeg",
                    ),
                ]
            ),
        ]

        with patch(
            "casual_llm.message_converters.google.fetch_image_as_base64",
            new=AsyncMock(return_value=("fetchedbase64data", "image/jpeg")),
        ) as mock_fetch:
            result, _ = await convert_messages_to_google(messages)

        # Verify fetch was called
        mock_fetch.assert_called_once_with("https://example.com/cat.jpg")

        # Verify result uses fetched data
        assert result[0]["parts"][1]["inline_data"]["data"] == "fetchedbase64data"


@pytest.mark.skipif(not GOOGLE_AVAILABLE, reason="Google provider not installed")
class TestGoogleProviderVision:
    """Tests for GoogleProvider with vision content using Gemini models."""

    @pytest.fixture
    def provider(self):
        """Create a GoogleProvider instance for testing."""
        if not GOOGLE_AVAILABLE:
            pytest.skip("Google provider not installed")

        # Import google.generativeai only when tests are running (not skipped)
        import google.generativeai as genai

        with patch.object(genai, "configure"):
            return GoogleProvider(
                model="gemini-1.5-flash",
                api_key="test-api-key",
                temperature=0.7,
            )

    @pytest.mark.asyncio
    async def test_chat_with_base64_image(self, provider):
        """Test chat with base64 encoded image in user message."""
        # Mock the response
        mock_part = MagicMock()
        mock_part.text = "I see a cat in the image."
        mock_part.function_call = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch.object(provider, "_create_model", return_value=mock_model):
            messages = [
                UserMessage(
                    content=[
                        TextContent(type="text", text="What is in this image?"),
                        ImageContent(
                            type="image",
                            source={"type": "base64", "data": "base64catdata"},
                            media_type="image/jpeg",
                        ),
                    ]
                )
            ]

            result = await provider.chat(messages)

            assert isinstance(result, AssistantMessage)
            assert result.content == "I see a cat in the image."

            # Verify the message format passed to Google
            call_args = mock_model.generate_content_async.call_args
            google_messages = call_args[0][0]
            assert len(google_messages) == 1
            assert google_messages[0]["role"] == "user"
            assert len(google_messages[0]["parts"]) == 2
            assert google_messages[0]["parts"][0] == {"text": "What is in this image?"}
            assert "inline_data" in google_messages[0]["parts"][1]
            assert google_messages[0]["parts"][1]["inline_data"]["data"] == "base64catdata"

    @pytest.mark.asyncio
    async def test_chat_with_url_image(self, provider):
        """Test chat with URL image in user message (fetched to base64)."""
        # Mock the response
        mock_part = MagicMock()
        mock_part.text = "This is a small red dot."
        mock_part.function_call = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with (
            patch.object(provider, "_create_model", return_value=mock_model),
            patch(
                "casual_llm.message_converters.google.fetch_image_as_base64",
                new=AsyncMock(return_value=("fetchedimagedata", "image/png")),
            ),
        ):
            messages = [
                UserMessage(
                    content=[
                        TextContent(type="text", text="Describe this image."),
                        ImageContent(
                            type="image",
                            source="https://example.com/image.png",
                            media_type="image/png",
                        ),
                    ]
                )
            ]

            result = await provider.chat(messages)

            assert isinstance(result, AssistantMessage)
            assert result.content == "This is a small red dot."

            # Verify fetched data was used
            call_args = mock_model.generate_content_async.call_args
            google_messages = call_args[0][0]
            image_part = google_messages[0]["parts"][1]
            assert image_part["inline_data"]["data"] == "fetchedimagedata"

    @pytest.mark.asyncio
    async def test_chat_with_multiple_images(self, provider):
        """Test chat with multiple images in a single message."""
        # Mock the response
        mock_part = MagicMock()
        mock_part.text = "The first image shows a cat, the second shows a dog."
        mock_part.function_call = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch.object(provider, "_create_model", return_value=mock_model):
            messages = [
                UserMessage(
                    content=[
                        TextContent(type="text", text="Compare these two images:"),
                        ImageContent(
                            type="image",
                            source={"type": "base64", "data": "catdata"},
                        ),
                        ImageContent(
                            type="image",
                            source={"type": "base64", "data": "dogdata"},
                        ),
                    ]
                )
            ]

            result = await provider.chat(messages)

            assert isinstance(result, AssistantMessage)

            # Verify all images were included
            call_args = mock_model.generate_content_async.call_args
            google_messages = call_args[0][0]
            parts = google_messages[0]["parts"]
            assert len(parts) == 3
            assert parts[0] == {"text": "Compare these two images:"}
            assert "inline_data" in parts[1]
            assert "inline_data" in parts[2]
            assert parts[1]["inline_data"]["data"] == "catdata"
            assert parts[2]["inline_data"]["data"] == "dogdata"

    @pytest.mark.asyncio
    async def test_chat_vision_conversation(self, provider):
        """Test multi-turn conversation with vision."""
        # Mock the response
        mock_part = MagicMock()
        mock_part.text = "The cat appears to be orange."
        mock_part.function_call = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch.object(provider, "_create_model", return_value=mock_model):
            messages = [
                UserMessage(
                    content=[
                        TextContent(type="text", text="What is this?"),
                        ImageContent(
                            type="image",
                            source={"type": "base64", "data": "catimagedata"},
                        ),
                    ]
                ),
                AssistantMessage(content="This is a photo of a cat."),
                UserMessage(content="What color is the cat?"),
            ]

            result = await provider.chat(messages)

            assert isinstance(result, AssistantMessage)
            assert result.content == "The cat appears to be orange."

            # Verify all messages were converted
            call_args = mock_model.generate_content_async.call_args
            google_messages = call_args[0][0]
            assert len(google_messages) == 3
            # First message has image
            assert google_messages[0]["role"] == "user"
            assert len(google_messages[0]["parts"]) == 2
            # Second message is model text
            assert google_messages[1]["role"] == "model"
            assert google_messages[1]["parts"] == [{"text": "This is a photo of a cat."}]
            # Third message is user text
            assert google_messages[2]["role"] == "user"
            assert google_messages[2]["parts"] == [{"text": "What color is the cat?"}]

    @pytest.mark.asyncio
    async def test_stream_with_vision(self, provider):
        """Test streaming with vision content."""

        async def mock_stream():
            """Mock async generator that yields stream chunks."""
            chunks = [
                MagicMock(
                    candidates=[
                        MagicMock(
                            content=MagicMock(
                                parts=[MagicMock(text="I see")]
                            )
                        )
                    ]
                ),
                MagicMock(
                    candidates=[
                        MagicMock(
                            content=MagicMock(
                                parts=[MagicMock(text=" a cat")]
                            )
                        )
                    ]
                ),
                MagicMock(
                    candidates=[
                        MagicMock(
                            content=MagicMock(
                                parts=[MagicMock(text=".")]
                            )
                        )
                    ]
                ),
            ]
            for chunk in chunks:
                yield chunk

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_stream())

        with patch.object(provider, "_create_model", return_value=mock_model):
            messages = [
                UserMessage(
                    content=[
                        TextContent(type="text", text="What is in this image?"),
                        ImageContent(
                            type="image",
                            source={"type": "base64", "data": "catdata"},
                        ),
                    ]
                )
            ]

            collected_chunks = []
            async for chunk in provider.stream(messages):
                collected_chunks.append(chunk)

            # Verify we got chunks
            assert len(collected_chunks) == 3
            assert collected_chunks[0].content == "I see"
            assert collected_chunks[1].content == " a cat"
            assert collected_chunks[2].content == "."

            # Verify stream=True and vision content was passed
            call_kwargs = mock_model.generate_content_async.call_args.kwargs
            assert call_kwargs["stream"] is True
