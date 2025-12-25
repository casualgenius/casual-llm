"""Tests for backward compatibility - text-only messages still work.

This test suite verifies that after adding multimodal/vision support,
existing text-only message functionality continues to work correctly
across all message types and all providers.
"""

import pytest

from casual_llm import (
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    AssistantToolCall,
    AssistantToolCallFunction,
    ChatMessage,
)
from casual_llm.message_converters.openai import convert_messages_to_openai
from casual_llm.message_converters.ollama import convert_messages_to_ollama


# ==============================================================================
# Message Model Backward Compatibility Tests
# ==============================================================================


class TestUserMessageBackwardCompatibility:
    """Test that UserMessage still works with string content."""

    def test_user_message_string_content(self):
        """Test UserMessage with simple string content."""
        msg = UserMessage(content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert isinstance(msg.content, str)

    def test_user_message_empty_string(self):
        """Test UserMessage with empty string content."""
        msg = UserMessage(content="")
        assert msg.role == "user"
        assert msg.content == ""

    def test_user_message_none_content(self):
        """Test UserMessage with None content."""
        msg = UserMessage(content=None)
        assert msg.role == "user"
        assert msg.content is None

    def test_user_message_multiline_string(self):
        """Test UserMessage with multiline string content."""
        content = """This is a multiline
message with several
lines of text."""
        msg = UserMessage(content=content)
        assert msg.role == "user"
        assert msg.content == content

    def test_user_message_unicode_content(self):
        """Test UserMessage with unicode characters."""
        content = "Hello, 世界! 🌍 Привет мир!"
        msg = UserMessage(content=content)
        assert msg.role == "user"
        assert msg.content == content

    def test_user_message_serialization(self):
        """Test UserMessage serialization with string content."""
        msg = UserMessage(content="Test message")
        data = msg.model_dump()
        assert data == {"role": "user", "content": "Test message"}

        # Test round-trip
        restored = UserMessage(**data)
        assert restored.content == "Test message"


class TestOtherMessagesBackwardCompatibility:
    """Test that all other message types still work."""

    def test_assistant_message_text_only(self):
        """Test AssistantMessage with text content."""
        msg = AssistantMessage(content="I am here to help!")
        assert msg.role == "assistant"
        assert msg.content == "I am here to help!"
        assert msg.tool_calls is None

    def test_assistant_message_with_tool_calls(self):
        """Test AssistantMessage with tool calls still works."""
        tool_call = AssistantToolCall(
            id="call_123",
            function=AssistantToolCallFunction(
                name="get_weather",
                arguments='{"city": "Paris"}',
            ),
        )
        msg = AssistantMessage(
            content="Let me check the weather.",
            tool_calls=[tool_call],
        )
        assert msg.role == "assistant"
        assert msg.content == "Let me check the weather."
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "get_weather"

    def test_system_message(self):
        """Test SystemMessage still works."""
        msg = SystemMessage(content="You are a helpful assistant.")
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."

    def test_tool_result_message(self):
        """Test ToolResultMessage still works."""
        msg = ToolResultMessage(
            name="get_weather",
            tool_call_id="call_123",
            content='{"temp": 20, "unit": "celsius"}',
        )
        assert msg.role == "tool"
        assert msg.name == "get_weather"
        assert msg.tool_call_id == "call_123"
        assert msg.content == '{"temp": 20, "unit": "celsius"}'


class TestChatMessageTypeAlias:
    """Test that ChatMessage type alias works with all message types."""

    def test_chat_message_accepts_all_types(self):
        """Test ChatMessage accepts all message types."""
        messages: list[ChatMessage] = [
            SystemMessage(content="Be helpful"),
            UserMessage(content="Hello"),
            AssistantMessage(content="Hi there!"),
            ToolResultMessage(name="tool", tool_call_id="id", content="result"),
        ]
        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"
        assert messages[3].role == "tool"


# ==============================================================================
# OpenAI Converter Backward Compatibility Tests
# ==============================================================================


class TestOpenAIConverterBackwardCompatibility:
    """Test that OpenAI converter still handles text-only messages."""

    def test_user_message_string_content(self):
        """Test converting UserMessage with string content."""
        msg = UserMessage(content="Hello, OpenAI!")
        result = convert_messages_to_openai([msg])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello, OpenAI!"

    def test_all_message_types(self):
        """Test converting all message types."""
        messages = [
            SystemMessage(content="You are helpful."),
            UserMessage(content="What is 2+2?"),
            AssistantMessage(content="2+2 equals 4."),
        ]
        result = convert_messages_to_openai(messages)

        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1] == {"role": "user", "content": "What is 2+2?"}
        assert result[2] == {"role": "assistant", "content": "2+2 equals 4."}

    def test_assistant_with_tool_calls(self):
        """Test converting assistant message with tool calls."""
        tool_call = AssistantToolCall(
            id="call_abc123",
            function=AssistantToolCallFunction(
                name="get_time",
                arguments="{}",
            ),
        )
        messages = [
            UserMessage(content="What time is it?"),
            AssistantMessage(content="Let me check.", tool_calls=[tool_call]),
            ToolResultMessage(name="get_time", tool_call_id="call_abc123", content="10:30 AM"),
        ]
        result = convert_messages_to_openai(messages)

        assert len(result) == 3
        assert result[0]["content"] == "What time is it?"
        assert result[1]["tool_calls"][0]["function"]["name"] == "get_time"
        assert result[2]["role"] == "tool"

    def test_multi_turn_conversation(self):
        """Test converting multi-turn text conversation."""
        messages = [
            SystemMessage(content="You are a math tutor."),
            UserMessage(content="What is 10 divided by 2?"),
            AssistantMessage(content="10 divided by 2 equals 5."),
            UserMessage(content="And what is 5 times 3?"),
            AssistantMessage(content="5 times 3 equals 15."),
        ]
        result = convert_messages_to_openai(messages)

        assert len(result) == 5
        assert result[0]["content"] == "You are a math tutor."
        assert result[1]["content"] == "What is 10 divided by 2?"
        assert result[2]["content"] == "10 divided by 2 equals 5."
        assert result[3]["content"] == "And what is 5 times 3?"
        assert result[4]["content"] == "5 times 3 equals 15."

    def test_empty_messages(self):
        """Test converting empty message list."""
        result = convert_messages_to_openai([])
        assert result == []

    def test_none_user_content(self):
        """Test converting UserMessage with None content."""
        msg = UserMessage(content=None)
        result = convert_messages_to_openai([msg])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] is None


# ==============================================================================
# Ollama Converter Backward Compatibility Tests
# ==============================================================================


class TestOllamaConverterBackwardCompatibility:
    """Test that Ollama converter still handles text-only messages."""

    @pytest.mark.asyncio
    async def test_user_message_string_content(self):
        """Test converting UserMessage with string content."""
        msg = UserMessage(content="Hello, Ollama!")
        result = await convert_messages_to_ollama([msg])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello, Ollama!"
        assert "images" not in result[0]  # No images key for text-only

    @pytest.mark.asyncio
    async def test_all_message_types(self):
        """Test converting all message types."""
        messages = [
            SystemMessage(content="You are a local AI."),
            UserMessage(content="What can you do?"),
            AssistantMessage(content="I can help with many tasks!"),
        ]
        result = await convert_messages_to_ollama(messages)

        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "You are a local AI."}
        assert result[1] == {"role": "user", "content": "What can you do?"}
        assert result[2] == {"role": "assistant", "content": "I can help with many tasks!"}

    @pytest.mark.asyncio
    async def test_assistant_with_tool_calls(self):
        """Test converting assistant message with tool calls."""
        tool_call = AssistantToolCall(
            id="call_xyz789",
            function=AssistantToolCallFunction(
                name="calculate",
                arguments='{"x": 5, "y": 10}',
            ),
        )
        messages = [
            UserMessage(content="Add 5 and 10"),
            AssistantMessage(content="Let me calculate.", tool_calls=[tool_call]),
        ]
        result = await convert_messages_to_ollama(messages)

        assert len(result) == 2
        assert result[0]["content"] == "Add 5 and 10"
        # Ollama expects arguments as dict, not JSON string
        assert result[1]["tool_calls"][0]["function"]["arguments"] == {"x": 5, "y": 10}

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Test converting multi-turn text conversation."""
        messages = [
            SystemMessage(content="You are a coding assistant."),
            UserMessage(content="How do I print in Python?"),
            AssistantMessage(content="Use print('Hello')"),
            UserMessage(content="And in JavaScript?"),
            AssistantMessage(content="Use console.log('Hello')"),
        ]
        result = await convert_messages_to_ollama(messages)

        assert len(result) == 5
        assert result[0]["content"] == "You are a coding assistant."
        assert result[1]["content"] == "How do I print in Python?"
        assert result[2]["content"] == "Use print('Hello')"
        assert result[3]["content"] == "And in JavaScript?"
        assert result[4]["content"] == "Use console.log('Hello')"

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Test converting empty message list."""
        result = await convert_messages_to_ollama([])
        assert result == []

    @pytest.mark.asyncio
    async def test_none_user_content(self):
        """Test converting UserMessage with None content."""
        msg = UserMessage(content=None)
        result = await convert_messages_to_ollama([msg])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == ""  # None converted to empty string
        assert "images" not in result[0]


# ==============================================================================
# Cross-Provider Consistency Tests
# ==============================================================================


class TestCrossProviderTextConsistency:
    """Test that text-only messages work consistently across all providers."""

    @pytest.mark.asyncio
    async def test_simple_user_message_all_providers(self):
        """Test that same user message works with all providers."""
        msg = UserMessage(content="Hello, World!")

        # OpenAI
        openai_result = convert_messages_to_openai([msg])
        assert openai_result[0]["content"] == "Hello, World!"

        # Ollama
        ollama_result = await convert_messages_to_ollama([msg])
        assert ollama_result[0]["content"] == "Hello, World!"

    @pytest.mark.asyncio
    async def test_conversation_all_providers(self):
        """Test that same conversation works with all providers."""
        messages = [
            SystemMessage(content="You are helpful."),
            UserMessage(content="Hi"),
            AssistantMessage(content="Hello!"),
        ]

        # OpenAI
        openai_result = convert_messages_to_openai(messages)
        assert len(openai_result) == 3

        # Ollama
        ollama_result = await convert_messages_to_ollama(messages)
        assert len(ollama_result) == 3


# ==============================================================================
# Regression Tests for Known Edge Cases
# ==============================================================================


class TestEdgeCaseBackwardCompatibility:
    """Test edge cases that should still work after multimodal changes."""

    def test_user_message_with_special_characters(self):
        """Test UserMessage with special characters in content."""
        content = 'Hello! <script>alert("xss")</script> & "quotes" \'single\''
        msg = UserMessage(content=content)
        assert msg.content == content

        result = convert_messages_to_openai([msg])
        assert result[0]["content"] == content

    def test_user_message_with_json_like_content(self):
        """Test UserMessage with JSON-like string content."""
        content = '{"name": "test", "value": [1, 2, 3]}'
        msg = UserMessage(content=content)
        assert msg.content == content
        assert isinstance(msg.content, str)

        result = convert_messages_to_openai([msg])
        assert result[0]["content"] == content

    def test_user_message_with_code_content(self):
        """Test UserMessage with code in content."""
        content = """```python
def hello():
    print("Hello, World!")
```"""
        msg = UserMessage(content=content)
        assert msg.content == content

    def test_long_content(self):
        """Test UserMessage with very long content."""
        content = "A" * 10000
        msg = UserMessage(content=content)
        assert msg.content == content
        assert len(msg.content) == 10000

    @pytest.mark.asyncio
    async def test_tool_call_flow_backward_compatible(self):
        """Test complete tool call flow still works."""
        tool_call = AssistantToolCall(
            id="call_test",
            function=AssistantToolCallFunction(
                name="get_weather",
                arguments='{"city": "London"}',
            ),
        )

        messages = [
            UserMessage(content="What's the weather in London?"),
            AssistantMessage(content=None, tool_calls=[tool_call]),
            ToolResultMessage(
                name="get_weather",
                tool_call_id="call_test",
                content='{"temp": 15, "condition": "cloudy"}',
            ),
            AssistantMessage(content="The weather in London is 15C and cloudy."),
        ]

        # Should work with all providers
        openai_result = convert_messages_to_openai(messages)
        assert len(openai_result) == 4
        assert openai_result[1]["tool_calls"] is not None

        ollama_result = await convert_messages_to_ollama(messages)
        assert len(ollama_result) == 4
