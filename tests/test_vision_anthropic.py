"""Tests for Anthropic vision support with Claude models."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from casual_llm.messages import (
    UserMessage,
    AssistantMessage,
    TextContent,
    ImageContent,
)
from casual_llm.message_converters.anthropic import (
    _convert_image_to_anthropic,
    _convert_user_content_to_anthropic,
    convert_messages_to_anthropic,
)


# Try to import Anthropic provider - may not be available
try:
    import anthropic  # noqa: F401 - Check if anthropic package is installed

    from casual_llm.providers import AnthropicProvider

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    AnthropicProvider = None  # type: ignore


class TestImageContentConversion:
    """Tests for _convert_image_to_anthropic function."""

    @pytest.mark.asyncio
    async def test_url_image_conversion(self):
        """Test URL image is fetched and converted to Anthropic base64 format."""
        image = ImageContent(
            type="image",
            source="https://example.com/image.jpg",
            media_type="image/jpeg",
        )

        with patch(
            "casual_llm.message_converters.anthropic.fetch_image_as_base64",
            new_callable=AsyncMock,
            return_value=("base64encodeddata", "image/jpeg"),
        ):
            result = await _convert_image_to_anthropic(image)

        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/jpeg"
        assert result["source"]["data"] == "base64encodeddata"

    @pytest.mark.asyncio
    async def test_url_image_uses_fetched_media_type(self):
        """Test that URL images use the fetched media type when not explicitly set."""
        image = ImageContent(
            type="image",
            source="https://example.com/image.png",
            # Uses default media_type="image/jpeg"
        )

        with patch(
            "casual_llm.message_converters.anthropic.fetch_image_as_base64",
            new_callable=AsyncMock,
            return_value=("pngdata", "image/png"),
        ):
            result = await _convert_image_to_anthropic(image)

        # Should use the fetched media type
        assert result["source"]["media_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_base64_image_conversion(self):
        """Test base64 image is converted to Anthropic format."""
        image = ImageContent(
            type="image",
            source={"type": "base64", "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="},
            media_type="image/png",
        )

        result = await _convert_image_to_anthropic(image)

        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/png"
        assert result["source"]["data"] == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    @pytest.mark.asyncio
    async def test_base64_image_jpeg_media_type(self):
        """Test base64 image with jpeg media type."""
        image = ImageContent(
            type="image",
            source={"type": "base64", "data": "base64encodeddata"},
            media_type="image/jpeg",
        )

        result = await _convert_image_to_anthropic(image)

        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/jpeg"
        assert result["source"]["data"] == "base64encodeddata"

    @pytest.mark.asyncio
    async def test_base64_image_webp_media_type(self):
        """Test base64 image with webp media type."""
        image = ImageContent(
            type="image",
            source={"type": "base64", "data": "webpdata"},
            media_type="image/webp",
        )

        result = await _convert_image_to_anthropic(image)

        assert result["source"]["media_type"] == "image/webp"
        assert result["source"]["data"] == "webpdata"

    @pytest.mark.asyncio
    async def test_data_uri_image_conversion(self):
        """Test data URI is converted to Anthropic format with prefix stripped."""
        image = ImageContent(
            type="image",
            source="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            media_type="image/png",
        )

        result = await _convert_image_to_anthropic(image)

        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/png"
        # Data URI prefix should be stripped
        assert result["source"]["data"] == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    @pytest.mark.asyncio
    async def test_base64_dict_with_data_uri_prefix(self):
        """Test base64 dict with data URI prefix is stripped."""
        image = ImageContent(
            type="image",
            source={"type": "base64", "data": "data:image/jpeg;base64,rawbase64data"},
            media_type="image/jpeg",
        )

        result = await _convert_image_to_anthropic(image)

        # Data URI prefix in the data field should be stripped
        assert result["source"]["data"] == "rawbase64data"


class TestUserContentConversion:
    """Tests for _convert_user_content_to_anthropic function."""

    @pytest.mark.asyncio
    async def test_string_content_passthrough(self):
        """Test that simple string content passes through unchanged."""
        result = await _convert_user_content_to_anthropic("Hello, world!")

        assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_none_content_returns_empty_string(self):
        """Test that None content returns empty string for Anthropic."""
        result = await _convert_user_content_to_anthropic(None)

        assert result == ""

    @pytest.mark.asyncio
    async def test_text_only_multimodal_content(self):
        """Test multimodal content with text only."""
        content = [
            TextContent(type="text", text="What is in this image?"),
        ]

        result = await _convert_user_content_to_anthropic(content)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "What is in this image?"

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

        result = await _convert_user_content_to_anthropic(content)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["data"] == "base64data"

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

        result = await _convert_user_content_to_anthropic(content)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Describe this image:"
        assert result[1]["type"] == "image"
        assert result[1]["source"]["data"] == "pngdata"

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

        result = await _convert_user_content_to_anthropic(content)

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image"
        assert result[2]["type"] == "image"
        assert result[1]["source"]["data"] == "image1data"
        assert result[2]["source"]["data"] == "image2data"

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
            "casual_llm.message_converters.anthropic.fetch_image_as_base64",
            new_callable=AsyncMock,
            return_value=("fetchedbase64", "image/jpeg"),
        ) as mock_fetch:
            result = await _convert_user_content_to_anthropic(content)

            mock_fetch.assert_called_once_with("https://example.com/cat.jpg")
            assert result[0]["source"]["data"] == "fetchedbase64"


class TestMessageConversionWithVision:
    """Tests for convert_messages_to_anthropic with vision content."""

    @pytest.mark.asyncio
    async def test_user_message_with_text_content(self):
        """Test converting user message with simple text."""
        messages = [UserMessage(content="Hello!")]

        result, system_prompt = await convert_messages_to_anthropic(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello!"
        assert system_prompt is None

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

        result, system_prompt = await convert_messages_to_anthropic(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image"

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

        result, system_prompt = await convert_messages_to_anthropic(messages)

        assert len(result) == 3
        # First message - multimodal
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        # Second message - assistant response
        assert result[1]["role"] == "assistant"
        # Third message - simple text follow-up
        assert result[2]["role"] == "user"
        assert result[2]["content"] == "What color is the cat?"


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic provider not installed")
class TestAnthropicProviderVision:
    """Tests for AnthropicProvider with vision content using Claude models."""

    @pytest.fixture
    def provider(self):
        """Create an AnthropicProvider instance for testing."""
        return AnthropicProvider(
            model="claude-3-5-sonnet-20241022",
            api_key="sk-test-key",
            temperature=0.7,
        )

    @pytest.mark.asyncio
    async def test_chat_with_base64_image(self, provider):
        """Test chat with base64 encoded image in user message."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="I see a cat in the image.")]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=20)

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(provider.client.messages, "create", new=mock_create):
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

            # Verify the message format passed to Anthropic
            call_kwargs = mock_create.call_args.kwargs
            chat_messages = call_kwargs["messages"]
            assert len(chat_messages) == 1
            assert chat_messages[0]["role"] == "user"
            assert isinstance(chat_messages[0]["content"], list)
            assert len(chat_messages[0]["content"]) == 2
            assert chat_messages[0]["content"][0]["type"] == "text"
            assert chat_messages[0]["content"][1]["type"] == "image"
            assert chat_messages[0]["content"][1]["source"]["type"] == "base64"
            assert chat_messages[0]["content"][1]["source"]["data"] == "base64imagedata"

    @pytest.mark.asyncio
    async def test_chat_with_url_image(self, provider):
        """Test chat with URL image in user message (fetched and converted)."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="This is a small red dot.")]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=20)

        mock_create = AsyncMock(return_value=mock_response)

        with (
            patch.object(provider.client.messages, "create", new=mock_create),
            patch(
                "casual_llm.message_converters.anthropic.fetch_image_as_base64",
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

            # Verify base64 format in the message
            call_kwargs = mock_create.call_args.kwargs
            chat_messages = call_kwargs["messages"]
            image_content = chat_messages[0]["content"][1]
            assert image_content["type"] == "image"
            assert image_content["source"]["type"] == "base64"
            assert image_content["source"]["data"] == "fetchedimagedata"

    @pytest.mark.asyncio
    async def test_chat_with_multiple_images(self, provider):
        """Test chat with multiple images in a single message."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="The first image shows a cat, the second shows a dog.")]
        mock_response.usage = MagicMock(input_tokens=200, output_tokens=30)

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(provider.client.messages, "create", new=mock_create):
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
            call_kwargs = mock_create.call_args.kwargs
            chat_messages = call_kwargs["messages"]
            content = chat_messages[0]["content"]
            assert len(content) == 3
            assert content[0]["type"] == "text"
            assert content[1]["type"] == "image"
            assert content[2]["type"] == "image"

    @pytest.mark.asyncio
    async def test_chat_vision_conversation(self, provider):
        """Test multi-turn conversation with vision."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="The cat appears to be orange.")]
        mock_response.usage = MagicMock(input_tokens=150, output_tokens=25)

        mock_create = AsyncMock(return_value=mock_response)

        with patch.object(provider.client.messages, "create", new=mock_create):
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
            call_kwargs = mock_create.call_args.kwargs
            chat_messages = call_kwargs["messages"]
            assert len(chat_messages) == 3
            # First message has image
            assert isinstance(chat_messages[0]["content"], list)
            # Third message is user text
            assert chat_messages[2]["content"] == "What color is the cat?"

    @pytest.mark.asyncio
    async def test_stream_with_vision(self, provider):
        """Test streaming with vision content."""

        async def mock_stream_context():
            """Mock async context manager for stream."""

            class MockStreamContext:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

                @property
                def text_stream(self):
                    async def generate():
                        yield "I see"
                        yield " a cat"
                        yield "."

                    return generate()

            return MockStreamContext()

        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        async def mock_text_stream():
            yield "I see"
            yield " a cat"
            yield "."

        mock_stream.text_stream = mock_text_stream()

        with patch.object(provider.client.messages, "stream", return_value=mock_stream):
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
