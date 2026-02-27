"""
Tests for Anthropic message converter edge cases.

Covers TST-004: Anthropic converter gaps including multi-image messages,
mixed text+image content, system message extraction, and tool call parsing.
"""

import json
from unittest.mock import MagicMock


from casual_llm.messages import (
    AssistantMessage,
    AssistantToolCall,
    AssistantToolCallFunction,
    ImageContent,
    SystemMessage,
    TextContent,
    ToolResultMessage,
    UserMessage,
)
from casual_llm.message_converters.anthropic import (
    _convert_image_to_anthropic,
    _convert_user_content_to_anthropic,
    convert_messages_to_anthropic,
    convert_tool_calls_from_anthropic,
)


class TestConvertImageToAnthropic:
    """Tests for _convert_image_to_anthropic."""

    def test_url_image(self):
        """URL-based image converts to Anthropic URL format."""
        img = ImageContent(source="https://example.com/image.jpg", media_type="image/jpeg")
        result = _convert_image_to_anthropic(img)

        assert result["type"] == "image"
        assert result["source"]["type"] == "url"
        assert result["source"]["url"] == "https://example.com/image.jpg"

    def test_base64_image(self):
        """Base64 dict image converts to Anthropic base64 format."""
        img = ImageContent(
            source={"type": "base64", "data": "abc123"},
            media_type="image/png",
        )
        result = _convert_image_to_anthropic(img)

        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/png"
        assert result["source"]["data"] == "abc123"

    def test_base64_image_default_media_type(self):
        """Base64 image uses default media type when not explicitly set."""
        img = ImageContent(
            source={"type": "base64", "data": "xyz"},
        )
        result = _convert_image_to_anthropic(img)

        assert result["source"]["media_type"] == "image/jpeg"  # default

    def test_base64_image_missing_data(self):
        """Base64 image with missing data key returns empty string."""
        img = ImageContent(
            source={"type": "base64"},
            media_type="image/png",
        )
        result = _convert_image_to_anthropic(img)

        assert result["source"]["data"] == ""


class TestConvertUserContentToAnthropic:
    """Tests for _convert_user_content_to_anthropic."""

    def test_string_content(self):
        """Simple string content converts to text block."""
        result = _convert_user_content_to_anthropic("Hello")
        assert len(result) == 1
        assert result[0] == {"type": "text", "text": "Hello"}

    def test_none_content(self):
        """None content converts to empty text block."""
        result = _convert_user_content_to_anthropic(None)
        assert len(result) == 1
        assert result[0] == {"type": "text", "text": ""}

    def test_multimodal_text_only(self):
        """List with only text content."""
        content = [TextContent(text="Hello"), TextContent(text=" world")]
        result = _convert_user_content_to_anthropic(content)

        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Hello"}
        assert result[1] == {"type": "text", "text": " world"}

    def test_multimodal_image_only(self):
        """List with only image content."""
        content = [ImageContent(source="https://example.com/img.jpg")]
        result = _convert_user_content_to_anthropic(content)

        assert len(result) == 1
        assert result[0]["type"] == "image"

    def test_multimodal_mixed_text_and_images(self):
        """List with mixed text and image content."""
        content = [
            TextContent(text="What's in this image?"),
            ImageContent(source="https://example.com/photo.jpg"),
            TextContent(text="And this one?"),
            ImageContent(source="https://example.com/another.jpg"),
        ]
        result = _convert_user_content_to_anthropic(content)

        assert len(result) == 4
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image"
        assert result[2]["type"] == "text"
        assert result[3]["type"] == "image"

    def test_multimodal_multiple_images(self):
        """Multiple images in a single message."""
        content = [
            TextContent(text="Compare these:"),
            ImageContent(source="https://example.com/a.jpg"),
            ImageContent(source="https://example.com/b.jpg"),
            ImageContent(source="https://example.com/c.jpg"),
        ]
        result = _convert_user_content_to_anthropic(content)

        assert len(result) == 4
        image_blocks = [b for b in result if b["type"] == "image"]
        assert len(image_blocks) == 3


class TestConvertMessagesToAnthropic:
    """Tests for convert_messages_to_anthropic."""

    def test_empty_messages(self):
        """Empty list returns empty list."""
        assert convert_messages_to_anthropic([]) == []

    def test_system_messages_excluded(self):
        """System messages are not included in the output."""
        messages = [
            SystemMessage(content="You are helpful"),
            UserMessage(content="Hello"),
        ]
        result = convert_messages_to_anthropic(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_user_message_simple(self):
        """Simple user message converts correctly."""
        messages = [UserMessage(content="Hello")]
        result = convert_messages_to_anthropic(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == [{"type": "text", "text": "Hello"}]

    def test_assistant_message_text_only(self):
        """Assistant message with text only."""
        messages = [AssistantMessage(content="Hi there!")]
        result = convert_messages_to_anthropic(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == [{"type": "text", "text": "Hi there!"}]

    def test_assistant_message_with_tool_calls(self):
        """Assistant message with tool calls."""
        messages = [
            AssistantMessage(
                content="Let me check.",
                tool_calls=[
                    AssistantToolCall(
                        id="tc_1",
                        function=AssistantToolCallFunction(
                            name="get_weather",
                            arguments='{"city": "NYC"}',
                        ),
                    )
                ],
            )
        ]
        result = convert_messages_to_anthropic(messages)

        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 2  # text + tool_use
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "tool_use"
        assert content[1]["id"] == "tc_1"
        assert content[1]["name"] == "get_weather"
        assert content[1]["input"] == {"city": "NYC"}

    def test_assistant_message_multiple_tool_calls(self):
        """Assistant message with multiple tool calls."""
        messages = [
            AssistantMessage(
                content="I'll check both.",
                tool_calls=[
                    AssistantToolCall(
                        id="tc_1",
                        function=AssistantToolCallFunction(
                            name="get_weather",
                            arguments='{"city": "NYC"}',
                        ),
                    ),
                    AssistantToolCall(
                        id="tc_2",
                        function=AssistantToolCallFunction(
                            name="get_time",
                            arguments='{"timezone": "EST"}',
                        ),
                    ),
                ],
            )
        ]
        result = convert_messages_to_anthropic(messages)

        content = result[0]["content"]
        assert len(content) == 3  # text + 2 tool_use blocks
        tool_uses = [b for b in content if b["type"] == "tool_use"]
        assert len(tool_uses) == 2
        assert tool_uses[0]["name"] == "get_weather"
        assert tool_uses[1]["name"] == "get_time"

    def test_tool_call_with_invalid_json_arguments(self):
        """Tool call with malformed JSON arguments produces empty dict input."""
        messages = [
            AssistantMessage(
                content="",
                tool_calls=[
                    AssistantToolCall(
                        id="tc_1",
                        function=AssistantToolCallFunction(
                            name="test",
                            arguments="not valid json",
                        ),
                    )
                ],
            )
        ]
        result = convert_messages_to_anthropic(messages)

        tool_use = result[0]["content"][0]  # No text content since content=""
        assert tool_use["type"] == "tool_use"
        assert tool_use["input"] == {}

    def test_tool_result_message(self):
        """Tool result converts to user message with tool_result content."""
        messages = [
            ToolResultMessage(
                name="get_weather",
                tool_call_id="tc_1",
                content='{"temperature": 72}',
            )
        ]
        result = convert_messages_to_anthropic(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "tc_1"
        assert content[0]["content"] == '{"temperature": 72}'

    def test_multi_turn_conversation(self):
        """Full multi-turn conversation with all message types."""
        messages = [
            SystemMessage(content="Be helpful"),
            UserMessage(content="Hello"),
            AssistantMessage(content="Hi!"),
            UserMessage(content="What's the weather?"),
            AssistantMessage(
                content="Let me check.",
                tool_calls=[
                    AssistantToolCall(
                        id="tc_1",
                        function=AssistantToolCallFunction(
                            name="weather",
                            arguments='{"city": "NYC"}',
                        ),
                    )
                ],
            ),
            ToolResultMessage(
                name="weather",
                tool_call_id="tc_1",
                content="72F",
            ),
            AssistantMessage(content="It's 72F in NYC."),
        ]
        result = convert_messages_to_anthropic(messages)

        # System excluded, so 6 messages
        assert len(result) == 6
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"
        assert result[3]["role"] == "assistant"
        assert result[4]["role"] == "user"  # tool result
        assert result[5]["role"] == "assistant"

    def test_user_message_with_vision(self):
        """User message with image content."""
        messages = [
            UserMessage(
                content=[
                    TextContent(text="What's in this image?"),
                    ImageContent(source="https://example.com/photo.jpg"),
                ]
            )
        ]
        result = convert_messages_to_anthropic(messages)

        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"


class TestConvertToolCallsFromAnthropic:
    """Tests for convert_tool_calls_from_anthropic."""

    def test_single_tool_call(self):
        """Single tool call converts correctly."""
        mock_block = MagicMock()
        mock_block.id = "tc_123"
        mock_block.name = "get_weather"
        mock_block.input = {"city": "NYC"}

        result = convert_tool_calls_from_anthropic([mock_block])

        assert len(result) == 1
        assert result[0].id == "tc_123"
        assert result[0].type == "function"
        assert result[0].function.name == "get_weather"
        assert json.loads(result[0].function.arguments) == {"city": "NYC"}

    def test_multiple_tool_calls(self):
        """Multiple tool calls convert correctly."""
        block1 = MagicMock()
        block1.id = "tc_1"
        block1.name = "tool_a"
        block1.input = {"x": 1}

        block2 = MagicMock()
        block2.id = "tc_2"
        block2.name = "tool_b"
        block2.input = {"y": 2}

        result = convert_tool_calls_from_anthropic([block1, block2])

        assert len(result) == 2
        assert result[0].function.name == "tool_a"
        assert result[1].function.name == "tool_b"

    def test_tool_call_with_empty_input(self):
        """Tool call with empty/None input serializes to '{}'."""
        mock_block = MagicMock()
        mock_block.id = "tc_1"
        mock_block.name = "no_args"
        mock_block.input = None

        result = convert_tool_calls_from_anthropic([mock_block])

        assert result[0].function.arguments == "{}"

    def test_tool_call_with_complex_input(self):
        """Tool call with nested input serializes correctly."""
        mock_block = MagicMock()
        mock_block.id = "tc_1"
        mock_block.name = "complex"
        mock_block.input = {"nested": {"key": "value"}, "list": [1, 2, 3]}

        result = convert_tool_calls_from_anthropic([mock_block])

        parsed = json.loads(result[0].function.arguments)
        assert parsed["nested"]["key"] == "value"
        assert parsed["list"] == [1, 2, 3]

    def test_empty_tool_calls(self):
        """Empty list returns empty list."""
        assert convert_tool_calls_from_anthropic([]) == []
