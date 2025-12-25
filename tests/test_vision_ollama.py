"""Tests for Ollama vision support with llava model."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from casual_llm.messages import (
    UserMessage,
    AssistantMessage,
    TextContent,
    ImageContent,
)
from casual_llm.message_converters.ollama import (
    _convert_image_to_ollama,
    _convert_user_content_to_ollama,
    convert_messages_to_ollama,
)


# Try to import Ollama provider - may not be available
try:
    from casual_llm.providers import OllamaProvider

    OLLAMA_AVAILABLE = OllamaProvider is not None
except ImportError:
    OLLAMA_AVAILABLE = False


class TestImageContentConversion:
    """Tests for _convert_image_to_ollama function."""

    @pytest.mark.asyncio
    async def test_url_image_conversion(self):
        """Test URL image is fetched and converted to raw base64 string."""
        image = ImageContent(
            type="image",
            source="https://example.com/image.jpg",
            media_type="image/jpeg",
        )

        with patch(
            "casual_llm.message_converters.ollama.fetch_image_as_base64",
            new_callable=AsyncMock,
            return_value=("base64encodeddata", "image/jpeg"),
        ):
            result = await _convert_image_to_ollama(image)

        # Ollama expects raw base64 string
        assert result == "base64encodeddata"

    @pytest.mark.asyncio
    async def test_base64_dict_image_conversion(self):
        """Test base64 dict image is converted to raw base64 string."""
        image = ImageContent(
            type="image",
            source={
                "type": "base64",
                "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            },
            media_type="image/png",
        )

        result = await _convert_image_to_ollama(image)

        # Should return raw base64 without prefix
        assert (
            result
            == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

    @pytest.mark.asyncio
    async def test_data_uri_image_conversion(self):
        """Test data URI is converted to raw base64 with prefix stripped."""
        image = ImageContent(
            type="image",
            source="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            media_type="image/png",
        )

        result = await _convert_image_to_ollama(image)

        # Data URI prefix should be stripped
        assert (
            result
            == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

    @pytest.mark.asyncio
    async def test_base64_dict_with_data_uri_prefix(self):
        """Test base64 dict with data URI prefix is stripped."""
        image = ImageContent(
            type="image",
            source={"type": "base64", "data": "data:image/jpeg;base64,rawbase64data"},
            media_type="image/jpeg",
        )

        result = await _convert_image_to_ollama(image)

        # Data URI prefix in the data field should be stripped
        assert result == "rawbase64data"


class TestUserContentConversion:
    """Tests for _convert_user_content_to_ollama function."""

    @pytest.mark.asyncio
    async def test_string_content_passthrough(self):
        """Test that simple string content passes through unchanged."""
        text, images = await _convert_user_content_to_ollama("Hello, world!")

        assert text == "Hello, world!"
        assert images == []

    @pytest.mark.asyncio
    async def test_none_content_returns_empty(self):
        """Test that None content returns empty string and list."""
        text, images = await _convert_user_content_to_ollama(None)

        assert text == ""
        assert images == []

    @pytest.mark.asyncio
    async def test_text_only_multimodal_content(self):
        """Test multimodal content with text only."""
        content = [
            TextContent(type="text", text="What is in this image?"),
        ]

        text, images = await _convert_user_content_to_ollama(content)

        assert text == "What is in this image?"
        assert images == []

    @pytest.mark.asyncio
    async def test_image_only_multimodal_content(self):
        """Test multimodal content with image only."""
        content = [
            ImageContent(
                type="image",
                source={"type": "base64", "data": "base64data"},
                media_type="image/jpeg",
            ),
        ]

        text, images = await _convert_user_content_to_ollama(content)

        assert text == ""
        assert len(images) == 1
        assert images[0] == "base64data"

    @pytest.mark.asyncio
    async def test_mixed_text_and_image_content(self):
        """Test multimodal content with text and image."""
        content = [
            TextContent(type="text", text="Describe this image:"),
            ImageContent(
                type="image",
                source={"type": "base64", "data": "pngdata"},
                media_type="image/png",
            ),
        ]

        text, images = await _convert_user_content_to_ollama(content)

        assert text == "Describe this image:"
        assert len(images) == 1
        assert images[0] == "pngdata"

    @pytest.mark.asyncio
    async def test_multiple_text_parts_joined(self):
        """Test that multiple text parts are joined with newlines."""
        content = [
            TextContent(type="text", text="First paragraph."),
            TextContent(type="text", text="Second paragraph."),
        ]

        text, images = await _convert_user_content_to_ollama(content)

        assert text == "First paragraph.\nSecond paragraph."
        assert images == []

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

        text, images = await _convert_user_content_to_ollama(content)

        assert text == "Compare these two images:"
        assert len(images) == 2
        assert images[0] == "image1data"
        assert images[1] == "image2data"

    @pytest.mark.asyncio
    async def test_url_image_is_fetched(self):
        """Test that URL images are fetched and converted to base64."""
        content = [
            ImageContent(
                type="image",
                source="https://example.com/cat.jpg",
                media_type="image/jpeg",
            ),
        ]

        with patch(
            "casual_llm.message_converters.ollama.fetch_image_as_base64",
            new_callable=AsyncMock,
            return_value=("fetchedbase64", "image/jpeg"),
        ) as mock_fetch:
            text, images = await _convert_user_content_to_ollama(content)

            mock_fetch.assert_called_once_with("https://example.com/cat.jpg")
            assert images[0] == "fetchedbase64"


class TestMessageConversionWithVision:
    """Tests for convert_messages_to_ollama with vision content."""

    @pytest.mark.asyncio
    async def test_user_message_with_text_content(self):
        """Test converting user message with simple text."""
        messages = [UserMessage(content="Hello!")]

        result = await convert_messages_to_ollama(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello!"
        assert "images" not in result[0]

    @pytest.mark.asyncio
    async def test_user_message_with_multimodal_content(self):
        """Test converting user message with multimodal content."""
        messages = [
            UserMessage(
                content=[
                    TextContent(type="text", text="What's in this image?"),
                    ImageContent(
                        type="image",
                        source={"type": "base64", "data": "base64data"},
                        media_type="image/jpeg",
                    ),
                ]
            )
        ]

        result = await convert_messages_to_ollama(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "What's in this image?"
        assert "images" in result[0]
        assert len(result[0]["images"]) == 1
        assert result[0]["images"][0] == "base64data"

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
                        media_type="image/jpeg",
                    ),
                ]
            ),
            AssistantMessage(content="This is a photo of a cat."),
            UserMessage(content="What color is the cat?"),
        ]

        result = await convert_messages_to_ollama(messages)

        assert len(result) == 3
        # First message - multimodal
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "What is this?"
        assert result[0]["images"] == ["catdata"]
        # Second message - assistant response
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "This is a photo of a cat."
        # Third message - simple text follow-up
        assert result[2]["role"] == "user"
        assert result[2]["content"] == "What color is the cat?"
        assert "images" not in result[2]

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
            "casual_llm.message_converters.ollama.fetch_image_as_base64",
            new_callable=AsyncMock,
            return_value=("fetchedbase64data", "image/jpeg"),
        ) as mock_fetch:
            result = await convert_messages_to_ollama(messages)

        # Verify fetch was called
        mock_fetch.assert_called_once_with("https://example.com/cat.jpg")

        # Verify result uses fetched data
        assert result[0]["images"][0] == "fetchedbase64data"


@pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="Ollama provider not installed")
class TestOllamaProviderVision:
    """Tests for OllamaProvider with vision content using llava model."""

    @pytest.fixture
    def provider(self):
        """Create an OllamaProvider instance for testing with llava model."""
        return OllamaProvider(
            model="llava",
            host="http://localhost:11434",
            temperature=0.7,
        )

    @pytest.mark.asyncio
    async def test_chat_with_base64_image(self, provider):
        """Test chat with base64 encoded image in user message."""
        mock_message = MagicMock()
        mock_message.content = "I see a cat in the image."
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.prompt_eval_count = 100
        mock_response.eval_count = 20

        mock_chat = AsyncMock(return_value=mock_response)

        with patch.object(provider.client, "chat", new=mock_chat):
            messages = [
                UserMessage(
                    content=[
                        TextContent(type="text", text="What is in this image?"),
                        ImageContent(
                            type="image",
                            source={"type": "base64", "data": "base64imagedata"},
                            media_type="image/jpeg",
                        ),
                    ]
                )
            ]

            result = await provider.chat(messages)

            assert isinstance(result, AssistantMessage)
            assert result.content == "I see a cat in the image."

            # Verify the message format passed to Ollama
            call_kwargs = mock_chat.call_args.kwargs
            chat_messages = call_kwargs["messages"]
            assert len(chat_messages) == 1
            assert chat_messages[0]["role"] == "user"
            assert chat_messages[0]["content"] == "What is in this image?"
            assert "images" in chat_messages[0]
            assert len(chat_messages[0]["images"]) == 1
            assert chat_messages[0]["images"][0] == "base64imagedata"

    @pytest.mark.asyncio
    async def test_chat_with_url_image(self, provider):
        """Test chat with URL image in user message (fetched and converted)."""
        mock_message = MagicMock()
        mock_message.content = "This is a small red dot."
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.prompt_eval_count = 100
        mock_response.eval_count = 20

        mock_chat = AsyncMock(return_value=mock_response)

        with (
            patch.object(provider.client, "chat", new=mock_chat),
            patch(
                "casual_llm.message_converters.ollama.fetch_image_as_base64",
                new_callable=AsyncMock,
                return_value=("fetchedimagedata", "image/png"),
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

            # Verify base64 data was used
            call_kwargs = mock_chat.call_args.kwargs
            chat_messages = call_kwargs["messages"]
            assert chat_messages[0]["images"][0] == "fetchedimagedata"

    @pytest.mark.asyncio
    async def test_chat_with_multiple_images(self, provider):
        """Test chat with multiple images in a single message."""
        mock_message = MagicMock()
        mock_message.content = "The first image shows a cat, the second shows a dog."
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.prompt_eval_count = 200
        mock_response.eval_count = 30

        mock_chat = AsyncMock(return_value=mock_response)

        with patch.object(provider.client, "chat", new=mock_chat):
            messages = [
                UserMessage(
                    content=[
                        TextContent(type="text", text="Compare these two images:"),
                        ImageContent(
                            type="image",
                            source={"type": "base64", "data": "catimagedata"},
                            media_type="image/jpeg",
                        ),
                        ImageContent(
                            type="image",
                            source={"type": "base64", "data": "dogimagedata"},
                            media_type="image/jpeg",
                        ),
                    ]
                )
            ]

            result = await provider.chat(messages)

            assert isinstance(result, AssistantMessage)

            # Verify all images were included
            call_kwargs = mock_chat.call_args.kwargs
            chat_messages = call_kwargs["messages"]
            assert chat_messages[0]["content"] == "Compare these two images:"
            assert len(chat_messages[0]["images"]) == 2
            assert chat_messages[0]["images"][0] == "catimagedata"
            assert chat_messages[0]["images"][1] == "dogimagedata"

    @pytest.mark.asyncio
    async def test_chat_vision_conversation(self, provider):
        """Test multi-turn conversation with vision."""
        mock_message = MagicMock()
        mock_message.content = "The cat appears to be orange."
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.prompt_eval_count = 150
        mock_response.eval_count = 25

        mock_chat = AsyncMock(return_value=mock_response)

        with patch.object(provider.client, "chat", new=mock_chat):
            messages = [
                UserMessage(
                    content=[
                        TextContent(type="text", text="What is this?"),
                        ImageContent(
                            type="image",
                            source={"type": "base64", "data": "catphoto"},
                            media_type="image/jpeg",
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
            call_kwargs = mock_chat.call_args.kwargs
            chat_messages = call_kwargs["messages"]
            assert len(chat_messages) == 3
            # First message has image
            assert chat_messages[0]["content"] == "What is this?"
            assert chat_messages[0]["images"] == ["catphoto"]
            # Second message is assistant text
            assert chat_messages[1]["content"] == "This is a photo of a cat."
            # Third message is user text (no images)
            assert chat_messages[2]["content"] == "What color is the cat?"
            assert "images" not in chat_messages[2]

    @pytest.mark.asyncio
    async def test_stream_with_vision(self, provider):
        """Test streaming with vision content."""

        async def mock_stream():
            """Mock async generator that yields stream chunks."""
            chunks = [
                MagicMock(
                    message=MagicMock(content="I see"),
                    done=False,
                ),
                MagicMock(
                    message=MagicMock(content=" a cat"),
                    done=False,
                ),
                MagicMock(
                    message=MagicMock(content="."),
                    done=True,
                ),
            ]
            for chunk in chunks:
                yield chunk

        mock_chat = AsyncMock(return_value=mock_stream())

        with patch.object(provider.client, "chat", new=mock_chat):
            messages = [
                UserMessage(
                    content=[
                        TextContent(type="text", text="What is in this image?"),
                        ImageContent(
                            type="image",
                            source={"type": "base64", "data": "catimage"},
                            media_type="image/jpeg",
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
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["stream"] is True
            chat_messages = call_kwargs["messages"]
            assert "images" in chat_messages[0]
            assert chat_messages[0]["images"][0] == "catimage"

    @pytest.mark.asyncio
    async def test_chat_with_data_uri_image(self, provider):
        """Test chat with data URI image (prefix stripped)."""
        mock_message = MagicMock()
        mock_message.content = "I see a colorful pattern."
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.prompt_eval_count = 100
        mock_response.eval_count = 20

        mock_chat = AsyncMock(return_value=mock_response)

        with patch.object(provider.client, "chat", new=mock_chat):
            messages = [
                UserMessage(
                    content=[
                        TextContent(type="text", text="What do you see?"),
                        ImageContent(
                            type="image",
                            source="data:image/png;base64,rawbase64data",
                            media_type="image/png",
                        ),
                    ]
                )
            ]

            result = await provider.chat(messages)

            assert isinstance(result, AssistantMessage)

            # Verify data URI prefix was stripped
            call_kwargs = mock_chat.call_args.kwargs
            chat_messages = call_kwargs["messages"]
            assert chat_messages[0]["images"][0] == "rawbase64data"
