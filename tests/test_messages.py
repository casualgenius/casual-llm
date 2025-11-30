"""Tests for message models."""

from casual_llm import (
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    AssistantToolCall,
    AssistantToolCallFunction,
    ChatMessage,
)


def test_user_message():
    """Test UserMessage creation and validation."""
    msg = UserMessage(content="Hello!")
    assert msg.role == "user"
    assert msg.content == "Hello!"


def test_user_message_none_content():
    """Test UserMessage with None content."""
    msg = UserMessage(content=None)
    assert msg.role == "user"
    assert msg.content is None


def test_assistant_message():
    """Test AssistantMessage without tool calls."""
    msg = AssistantMessage(content="Hi there!")
    assert msg.role == "assistant"
    assert msg.content == "Hi there!"
    assert msg.tool_calls is None


def test_assistant_message_with_tool_calls():
    """Test AssistantMessage with tool calls."""
    tool_call = AssistantToolCall(
        id="call_123",
        function=AssistantToolCallFunction(name="get_weather", arguments='{"city": "Paris"}'),
    )

    msg = AssistantMessage(content="Let me check the weather.", tool_calls=[tool_call])
    assert msg.role == "assistant"
    assert msg.content == "Let me check the weather."
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "call_123"
    assert msg.tool_calls[0].function.name == "get_weather"


def test_system_message():
    """Test SystemMessage creation."""
    msg = SystemMessage(content="You are a helpful assistant.")
    assert msg.role == "system"
    assert msg.content == "You are a helpful assistant."


def test_tool_result_message():
    """Test ToolResultMessage creation."""
    msg = ToolResultMessage(name="get_weather", tool_call_id="call_123", content='{"temp": 20}')
    assert msg.role == "tool"
    assert msg.name == "get_weather"
    assert msg.tool_call_id == "call_123"
    assert msg.content == '{"temp": 20}'


def test_chat_message_type_alias():
    """Test that ChatMessage accepts all message types."""
    messages: list[ChatMessage] = [
        SystemMessage(content="System"),
        UserMessage(content="User"),
        AssistantMessage(content="Assistant"),
        ToolResultMessage(name="tool", tool_call_id="id", content="result"),
    ]

    assert len(messages) == 4
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert messages[2].role == "assistant"
    assert messages[3].role == "tool"


def test_message_serialization():
    """Test message model serialization."""
    msg = UserMessage(content="Test")
    data = msg.model_dump()

    assert data == {"role": "user", "content": "Test"}

    # Test round-trip
    restored = UserMessage(**data)
    assert restored.content == "Test"
