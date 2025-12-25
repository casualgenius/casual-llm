"""Tests for OpenAI vision support with gpt-4o model."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from casual_llm.messages import (
    UserMessage,
    AssistantMessage,
    TextContent,
    ImageContent,
)
from casual_llm.message_converters.openai import (
    _convert_image_to_openai,
    _convert_user_content_to_openai,
    convert_messages_to_openai,
)


# Try to import OpenAI provider - may not be available
try:
    from casual_llm.providers import OpenAIProvider

    OPENAI_AVAILABLE = OpenAIProvider is not None
except ImportError:
    OPENAI_AVAILABLE = False


class TestImageContentConversion:
    """Tests for _convert_image_to_openai function."""

    def test_url_image_conversion(self):
        """Test URL image is converted to OpenAI image_url format."""
        image = ImageContent(
            type="image",
            source="https://example.com/image.jpg",
            media_type="image/jpeg",
        )

        result = _convert_image_to_openai(image)

        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == "https://example.com/image.jpg"

    def test_base64_image_conversion(self):
        """Test base64 image is converted to data URI format."""
        image = ImageContent(
            type="image",
            source={"type": "base64", "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="},
            media_type="image/png",
        )

        result = _convert_image_to_openai(image)

        assert result["type"] == "image_url"
        expected_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        assert result["image_url"]["url"] == expected_url

    def test_base64_image_jpeg_media_type(self):
        """Test base64 image with jpeg media type."""
        image = ImageContent(
            type="image",
            source={"type": "base64", "data": "base64encodeddata"},
            media_type="image/jpeg",
        )

        result = _convert_image_to_openai(image)

        assert result["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert result["image_url"]["url"].endswith("base64encodeddata")

    def test_base64_image_webp_media_type(self):
        """Test base64 image with webp media type."""
        image = ImageContent(
            type="image",
            source={"type": "base64", "data": "webpdata"},
            media_type="image/webp",
        )

        result = _convert_image_to_openai(image)

        assert "data:image/webp;base64," in result["image_url"]["url"]


class TestUserContentConversion:
    """Tests for _convert_user_content_to_openai function."""

    def test_string_content_passthrough(self):
        """Test that simple string content passes through unchanged."""
        result = _convert_user_content_to_openai("Hello, world!")

        assert result == "Hello, world!"

    def test_none_content_passthrough(self):
        """Test that None content passes through unchanged."""
        result = _convert_user_content_to_openai(None)

        assert result is None

    def test_text_only_multimodal_content(self):
        """Test multimodal content with text only."""
        content = [
            TextContent(type="text", text="What is in this image?"),
        ]

        result = _convert_user_content_to_openai(content)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "What is in this image?"

    def test_image_only_multimodal_content(self):
        """Test multimodal content with image only."""
        content = [
            ImageContent(
                type="image",
                source="https://example.com/cat.jpg",
                media_type="image/jpeg",
            ),
        ]

        result = _convert_user_content_to_openai(content)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"]["url"] == "https://example.com/cat.jpg"

    def test_mixed_text_and_image_content(self):
        """Test multimodal content with text and image."""
        content = [
            TextContent(type="text", text="Describe this image:"),
            ImageContent(
                type="image",
                source="https://example.com/photo.png",
                media_type="image/png",
            ),
        ]

        result = _convert_user_content_to_openai(content)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Describe this image:"
        assert result[1]["type"] == "image_url"
        assert result[1]["image_url"]["url"] == "https://example.com/photo.png"

    def test_multiple_images_content(self):
        """Test multimodal content with multiple images."""
        content = [
            TextContent(type="text", text="Compare these two images:"),
            ImageContent(
                type="image",
                source="https://example.com/image1.jpg",
                media_type="image/jpeg",
            ),
            ImageContent(
                type="image",
                source="https://example.com/image2.jpg",
                media_type="image/jpeg",
            ),
        ]

        result = _convert_user_content_to_openai(content)

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"
        assert result[2]["type"] == "image_url"
        assert result[1]["image_url"]["url"] == "https://example.com/image1.jpg"
        assert result[2]["image_url"]["url"] == "https://example.com/image2.jpg"


class TestMessageConversionWithVision:
    """Tests for convert_messages_to_openai with vision content."""

    def test_user_message_with_text_content(self):
        """Test converting user message with simple text."""
        messages = [UserMessage(content="Hello!")]

        result = convert_messages_to_openai(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello!"

    def test_user_message_with_multimodal_content(self):
        """Test converting user message with multimodal content."""
        messages = [
            UserMessage(
                content=[
                    TextContent(type="text", text="What's in this image?"),
                    ImageContent(
                        type="image",
                        source="https://example.com/image.jpg",
                        media_type="image/jpeg",
                    ),
                ]
            )
        ]

        result = convert_messages_to_openai(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image_url"

    def test_mixed_messages_with_vision(self):
        """Test converting conversation with vision and text messages."""
        messages = [
            UserMessage(
                content=[
                    TextContent(type="text", text="What is this?"),
                    ImageContent(
                        type="image",
                        source="https://example.com/cat.jpg",
                    ),
                ]
            ),
            AssistantMessage(content="This is a photo of a cat."),
            UserMessage(content="What color is the cat?"),
        ]

        result = convert_messages_to_openai(messages)

        assert len(result) == 3
        # First message - multimodal
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        # Second message - assistant response
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "This is a photo of a cat."
        # Third message - simple text follow-up
        assert result[2]["role"] == "user"
        assert result[2]["content"] == "What color is the cat?"


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI provider not installed")
class TestOpenAIProviderVision:
    """Tests for OpenAIProvider with vision content using gpt-4o model."""

    @pytest.fixture
    def provider(self):
        """Create an OpenAIProvider instance for testing with gpt-4o model."""
        return OpenAIProvider(
            model="gpt-4o",
            api_key="sk-test-key",
            temperature=0.7,
        )

    @pytest.mark.asyncio
    async def test_chat_with_url_image(self, provider):
        """Test chat with URL image in user message."""
        mock_completion = MagicMock()
        mock_message = MagicMock(content="I see a cat in the image.")
        del mock_message.tool_calls
        mock_completion.choices = [MagicMock(message=mock_message)]
        mock_completion.usage = None

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(provider.client.chat.completions, "create", new=mock_create):
            messages = [
                UserMessage(
                    content=[
                        TextContent(type="text", text="What is in this image?"),
                        ImageContent(
                            type="image",
                            source="https://example.com/cat.jpg",
                            media_type="image/jpeg",
                        ),
                    ]
                )
            ]

            result = await provider.chat(messages)

            assert isinstance(result, AssistantMessage)
            assert result.content == "I see a cat in the image."

            # Verify the message format passed to OpenAI
            call_kwargs = mock_create.call_args.kwargs
            chat_messages = call_kwargs["messages"]
            assert len(chat_messages) == 1
            assert chat_messages[0]["role"] == "user"
            assert isinstance(chat_messages[0]["content"], list)
            assert len(chat_messages[0]["content"]) == 2
            assert chat_messages[0]["content"][0]["type"] == "text"
            assert chat_messages[0]["content"][1]["type"] == "image_url"
            assert (
                chat_messages[0]["content"][1]["image_url"]["url"]
                == "https://example.com/cat.jpg"
            )

    @pytest.mark.asyncio
    async def test_chat_with_base64_image(self, provider):
        """Test chat with base64 encoded image in user message."""
        mock_completion = MagicMock()
        mock_message = MagicMock(content="This is a small red dot.")
        del mock_message.tool_calls
        mock_completion.choices = [MagicMock(message=mock_message)]
        mock_completion.usage = None

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(provider.client.chat.completions, "create", new=mock_create):
            messages = [
                UserMessage(
                    content=[
                        TextContent(type="text", text="Describe this image."),
                        ImageContent(
                            type="image",
                            source={"type": "base64", "data": "base64encodeddata"},
                            media_type="image/png",
                        ),
                    ]
                )
            ]

            result = await provider.chat(messages)

            assert isinstance(result, AssistantMessage)
            assert result.content == "This is a small red dot."

            # Verify base64 data URI format
            call_kwargs = mock_create.call_args.kwargs
            chat_messages = call_kwargs["messages"]
            image_content = chat_messages[0]["content"][1]
            assert image_content["type"] == "image_url"
            assert image_content["image_url"]["url"] == "data:image/png;base64,base64encodeddata"

    @pytest.mark.asyncio
    async def test_chat_with_multiple_images(self, provider):
        """Test chat with multiple images in a single message."""
        mock_completion = MagicMock()
        mock_message = MagicMock(content="The first image shows a cat, the second shows a dog.")
        del mock_message.tool_calls
        mock_completion.choices = [MagicMock(message=mock_message)]
        mock_completion.usage = None

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(provider.client.chat.completions, "create", new=mock_create):
            messages = [
                UserMessage(
                    content=[
                        TextContent(type="text", text="Compare these two images:"),
                        ImageContent(
                            type="image",
                            source="https://example.com/cat.jpg",
                        ),
                        ImageContent(
                            type="image",
                            source="https://example.com/dog.jpg",
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
            assert content[1]["type"] == "image_url"
            assert content[2]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_chat_vision_conversation(self, provider):
        """Test multi-turn conversation with vision."""
        mock_completion = MagicMock()
        mock_message = MagicMock(content="The cat appears to be orange.")
        del mock_message.tool_calls
        mock_completion.choices = [MagicMock(message=mock_message)]
        mock_completion.usage = None

        mock_create = AsyncMock(return_value=mock_completion)

        with patch.object(provider.client.chat.completions, "create", new=mock_create):
            messages = [
                UserMessage(
                    content=[
                        TextContent(type="text", text="What is this?"),
                        ImageContent(
                            type="image",
                            source="https://example.com/cat.jpg",
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
            # Second message is assistant text
            assert chat_messages[1]["content"] == "This is a photo of a cat."
            # Third message is user text
            assert chat_messages[2]["content"] == "What color is the cat?"

    @pytest.mark.asyncio
    async def test_stream_with_vision(self, provider):
        """Test streaming with vision content."""

        async def mock_stream():
            """Mock async generator that yields stream chunks."""
            chunks = [
                MagicMock(
                    choices=[MagicMock(delta=MagicMock(content="I see"), finish_reason=None)]
                ),
                MagicMock(
                    choices=[MagicMock(delta=MagicMock(content=" a cat"), finish_reason=None)]
                ),
                MagicMock(
                    choices=[MagicMock(delta=MagicMock(content="."), finish_reason="stop")]
                ),
            ]
            for chunk in chunks:
                yield chunk

        mock_create = AsyncMock(return_value=mock_stream())

        with patch.object(provider.client.chat.completions, "create", new=mock_create):
            messages = [
                UserMessage(
                    content=[
                        TextContent(type="text", text="What is in this image?"),
                        ImageContent(
                            type="image",
                            source="https://example.com/cat.jpg",
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
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["stream"] is True
            chat_messages = call_kwargs["messages"]
            assert isinstance(chat_messages[0]["content"], list)
            assert chat_messages[0]["content"][1]["type"] == "image_url"
