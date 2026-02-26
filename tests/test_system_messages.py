"""
Tests for multiple system message handling.

Covers merge_system_messages utility, extract_system_messages Anthropic converter,
system_message_handling resolution chain, and per-provider behavior.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from casual_llm.config import ChatOptions
from casual_llm.messages import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from casual_llm.model import Model
from casual_llm.message_converters.utils import merge_system_messages
from casual_llm.message_converters.anthropic import extract_system_messages


# ---------------------------------------------------------------------------
# merge_system_messages
# ---------------------------------------------------------------------------


class TestMergeSystemMessages:
    """Tests for merge_system_messages utility."""

    def test_multiple_system_messages_merged(self):
        """Multiple system messages are joined with double newline."""
        messages = [
            SystemMessage(content="You are helpful"),
            UserMessage(content="Hello"),
            SystemMessage(content="Be concise"),
        ]
        result = merge_system_messages(messages)

        assert len(result) == 2
        assert result[0].role == "system"
        assert result[0].content == "You are helpful\n\nBe concise"
        assert result[1].role == "user"

    def test_single_system_message_unchanged(self):
        """Single system message returns original list."""
        messages = [
            SystemMessage(content="Only one"),
            UserMessage(content="Hello"),
        ]
        result = merge_system_messages(messages)

        assert result is messages  # identity check — same list object

    def test_no_system_messages_unchanged(self):
        """No system messages returns original list."""
        messages = [
            UserMessage(content="Hello"),
            AssistantMessage(content="Hi"),
        ]
        result = merge_system_messages(messages)

        assert result is messages

    def test_empty_system_messages_filtered(self):
        """Empty/whitespace-only system messages are excluded."""
        messages = [
            SystemMessage(content="Keep this"),
            SystemMessage(content="   "),
            SystemMessage(content=""),
            UserMessage(content="Hello"),
        ]
        result = merge_system_messages(messages)

        # Only one non-empty system message so list returned unchanged
        assert result is messages

    def test_non_system_order_preserved(self):
        """Non-system messages maintain their original order."""
        messages = [
            SystemMessage(content="System A"),
            UserMessage(content="First"),
            SystemMessage(content="System B"),
            AssistantMessage(content="Second"),
            UserMessage(content="Third"),
        ]
        result = merge_system_messages(messages)

        assert result[0].role == "system"
        assert result[1].role == "user"
        assert result[1].content == "First"
        assert result[2].role == "assistant"
        assert result[2].content == "Second"
        assert result[3].role == "user"
        assert result[3].content == "Third"

    def test_mixed_positions(self):
        """System messages interspersed with other types are gathered."""
        messages = [
            UserMessage(content="before"),
            SystemMessage(content="middle system"),
            AssistantMessage(content="response"),
            SystemMessage(content="late system"),
        ]
        result = merge_system_messages(messages)

        assert len(result) == 3  # 1 merged system + 2 non-system
        assert result[0].content == "middle system\n\nlate system"
        assert result[1].content == "before"
        assert result[2].content == "response"

    def test_three_system_messages(self):
        """Three system messages produce correct merged content."""
        messages = [
            SystemMessage(content="A"),
            SystemMessage(content="B"),
            SystemMessage(content="C"),
            UserMessage(content="Hello"),
        ]
        result = merge_system_messages(messages)

        assert result[0].content == "A\n\nB\n\nC"
        assert result[1].content == "Hello"

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped from each part."""
        messages = [
            SystemMessage(content="  padded  "),
            SystemMessage(content="\nnewlines\n"),
            UserMessage(content="Hello"),
        ]
        result = merge_system_messages(messages)

        assert result[0].content == "padded\n\nnewlines"

    def test_all_empty_system_messages(self):
        """All-empty system messages means 0 non-empty — unchanged."""
        messages = [
            SystemMessage(content=""),
            SystemMessage(content="  "),
            UserMessage(content="Hello"),
        ]
        result = merge_system_messages(messages)

        assert result is messages


# ---------------------------------------------------------------------------
# extract_system_messages (Anthropic content blocks)
# ---------------------------------------------------------------------------


class TestExtractSystemMessages:
    """Tests for extract_system_messages."""

    def test_single_system_message(self):
        """Single system message returns one content block."""
        messages = [
            SystemMessage(content="You are helpful"),
            UserMessage(content="Hello"),
        ]
        result = extract_system_messages(messages)

        assert result == [{"type": "text", "text": "You are helpful"}]

    def test_multiple_system_messages(self):
        """Multiple system messages return multiple content blocks in order."""
        messages = [
            SystemMessage(content="First"),
            UserMessage(content="Hello"),
            SystemMessage(content="Second"),
            SystemMessage(content="Third"),
        ]
        result = extract_system_messages(messages)

        assert len(result) == 3
        assert result[0] == {"type": "text", "text": "First"}
        assert result[1] == {"type": "text", "text": "Second"}
        assert result[2] == {"type": "text", "text": "Third"}

    def test_no_system_messages_returns_none(self):
        """No system messages returns None."""
        messages = [UserMessage(content="Hello")]
        result = extract_system_messages(messages)

        assert result is None

    def test_empty_content_filtered(self):
        """Empty/whitespace system messages are filtered out."""
        messages = [
            SystemMessage(content="Keep"),
            SystemMessage(content=""),
            SystemMessage(content="   "),
        ]
        result = extract_system_messages(messages)

        assert result == [{"type": "text", "text": "Keep"}]

    def test_all_empty_returns_none(self):
        """All empty system messages returns None."""
        messages = [
            SystemMessage(content=""),
            SystemMessage(content="  "),
        ]
        result = extract_system_messages(messages)

        assert result is None

    def test_whitespace_stripped(self):
        """Content whitespace is stripped."""
        messages = [SystemMessage(content="  hello world  ")]
        result = extract_system_messages(messages)

        assert result == [{"type": "text", "text": "hello world"}]

    def test_empty_messages_list(self):
        """Empty message list returns None."""
        assert extract_system_messages([]) is None


# ---------------------------------------------------------------------------
# System message handling resolution chain
# ---------------------------------------------------------------------------


class TestSystemMessageHandlingResolution:
    """Test the resolution chain: per-call > model > client > default."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client with system_message_handling attr."""
        client = MagicMock()
        client.system_message_handling = None
        client._chat = AsyncMock(return_value=(AssistantMessage(content="ok"), None))
        client._stream = AsyncMock()
        return client

    def test_per_call_wins_over_model(self, mock_client):
        """Per-call ChatOptions.system_message_handling wins."""
        model = Model(mock_client, name="test", system_message_handling="passthrough")
        options = ChatOptions(system_message_handling="merge")

        resolved = model._resolve_system_message_handling(options)
        assert resolved == "merge"

    def test_model_wins_over_client(self, mock_client):
        """Model-level wins when per-call is None."""
        mock_client.system_message_handling = "passthrough"
        model = Model(mock_client, name="test", system_message_handling="merge")
        options = ChatOptions()

        resolved = model._resolve_system_message_handling(options)
        assert resolved == "merge"

    def test_client_wins_over_default(self, mock_client):
        """Client-level wins when model and per-call are both None."""
        mock_client.system_message_handling = "merge"
        model = Model(mock_client, name="test")
        options = ChatOptions()

        resolved = model._resolve_system_message_handling(options)
        assert resolved == "merge"

    def test_none_at_all_levels(self, mock_client):
        """All levels None → returns None (provider decides default)."""
        model = Model(mock_client, name="test")
        options = ChatOptions()

        resolved = model._resolve_system_message_handling(options)
        assert resolved is None

    def test_client_without_attr(self):
        """Client without system_message_handling attr returns None."""
        client = MagicMock(spec=[])  # no attributes
        model = Model(client, name="test")
        options = ChatOptions()

        resolved = model._resolve_system_message_handling(options)
        assert resolved is None

    @pytest.mark.asyncio
    async def test_resolved_injected_into_chat(self, mock_client):
        """Resolved handling is injected into options passed to client._chat."""
        model = Model(mock_client, name="test", system_message_handling="merge")
        await model.chat([UserMessage(content="Hi")])

        call_options = mock_client._chat.call_args[1]["options"]
        assert call_options.system_message_handling == "merge"


# ---------------------------------------------------------------------------
# OpenAI provider merge behavior
# ---------------------------------------------------------------------------


class TestOpenAISystemMessageMerge:
    """Test OpenAI client applies merge when configured."""

    @pytest.fixture
    def client(self):
        from casual_llm.providers.openai import OpenAIClient

        return OpenAIClient(api_key="test-key")

    def test_default_passthrough(self, client):
        """Default behavior passes through multiple system messages."""
        messages = [
            SystemMessage(content="First"),
            UserMessage(content="Hello"),
            SystemMessage(content="Second"),
        ]
        kwargs = client._build_request_kwargs("gpt-4", messages, ChatOptions())

        system_msgs = [m for m in kwargs["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 2

    def test_merge_via_options(self, client):
        """ChatOptions merge combines system messages."""
        messages = [
            SystemMessage(content="First"),
            UserMessage(content="Hello"),
            SystemMessage(content="Second"),
        ]
        kwargs = client._build_request_kwargs(
            "gpt-4", messages, ChatOptions(system_message_handling="merge")
        )

        system_msgs = [m for m in kwargs["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "First" in system_msgs[0]["content"]
        assert "Second" in system_msgs[0]["content"]

    def test_merge_via_client_init(self):
        """Client-level merge setting is applied."""
        from casual_llm.providers.openai import OpenAIClient

        client = OpenAIClient(api_key="test-key", system_message_handling="merge")
        messages = [
            SystemMessage(content="A"),
            SystemMessage(content="B"),
            UserMessage(content="Hello"),
        ]
        kwargs = client._build_request_kwargs("gpt-4", messages, ChatOptions())

        system_msgs = [m for m in kwargs["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1

    def test_passthrough_explicit(self, client):
        """Explicit passthrough keeps all system messages."""
        messages = [
            SystemMessage(content="A"),
            SystemMessage(content="B"),
            UserMessage(content="Hello"),
        ]
        kwargs = client._build_request_kwargs(
            "gpt-4", messages, ChatOptions(system_message_handling="passthrough")
        )

        system_msgs = [m for m in kwargs["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 2


# ---------------------------------------------------------------------------
# Ollama provider merge behavior
# ---------------------------------------------------------------------------


class TestOllamaSystemMessageMerge:
    """Test Ollama client applies merge when configured."""

    @pytest.fixture
    def client(self):
        from casual_llm.providers.ollama import OllamaClient

        return OllamaClient()

    @pytest.mark.asyncio
    async def test_default_passthrough(self, client):
        """Default passes through multiple system messages."""
        messages = [
            SystemMessage(content="First"),
            UserMessage(content="Hello"),
            SystemMessage(content="Second"),
        ]
        kwargs = await client._build_request_kwargs("llama3", messages, ChatOptions())

        system_msgs = [m for m in kwargs["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 2

    @pytest.mark.asyncio
    async def test_merge_via_options(self, client):
        """ChatOptions merge combines system messages."""
        messages = [
            SystemMessage(content="First"),
            UserMessage(content="Hello"),
            SystemMessage(content="Second"),
        ]
        kwargs = await client._build_request_kwargs(
            "llama3", messages, ChatOptions(system_message_handling="merge")
        )

        system_msgs = [m for m in kwargs["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "First" in system_msgs[0]["content"]
        assert "Second" in system_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_merge_via_client_init(self):
        """Client-level merge setting is applied."""
        from casual_llm.providers.ollama import OllamaClient

        client = OllamaClient(system_message_handling="merge")
        messages = [
            SystemMessage(content="A"),
            SystemMessage(content="B"),
            UserMessage(content="Hello"),
        ]
        kwargs = await client._build_request_kwargs("llama3", messages, ChatOptions())

        system_msgs = [m for m in kwargs["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1


# ---------------------------------------------------------------------------
# Anthropic provider content blocks
# ---------------------------------------------------------------------------


try:
    from casual_llm.providers.anthropic import AnthropicClient

    ANTHROPIC_AVAILABLE = AnthropicClient is not None
except ImportError:
    ANTHROPIC_AVAILABLE = False


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic client not installed")
class TestAnthropicSystemMessageBlocks:
    """Test that Anthropic provider passes all system messages as content blocks."""

    @pytest.fixture
    def client(self):
        return AnthropicClient(api_key="test-key")

    def test_single_system_as_blocks(self, client):
        """Single system message produces content block list."""
        messages = [
            SystemMessage(content="You are helpful"),
            UserMessage(content="Hello"),
        ]
        kwargs = client._build_request_kwargs("claude-3-haiku-20240307", messages, ChatOptions())

        assert isinstance(kwargs["system"], list)
        assert len(kwargs["system"]) == 1
        assert kwargs["system"][0] == {"type": "text", "text": "You are helpful"}

    def test_multiple_systems_as_blocks(self, client):
        """Multiple system messages all appear as content blocks."""
        messages = [
            SystemMessage(content="You are helpful"),
            UserMessage(content="Hello"),
            SystemMessage(content="Be concise"),
            SystemMessage(content="Use markdown"),
        ]
        kwargs = client._build_request_kwargs("claude-3-haiku-20240307", messages, ChatOptions())

        system_blocks = kwargs["system"]
        assert len(system_blocks) == 3
        assert system_blocks[0]["text"] == "You are helpful"
        assert system_blocks[1]["text"] == "Be concise"
        assert system_blocks[2]["text"] == "Use markdown"

    def test_no_system_message(self, client):
        """No system messages means no system key in kwargs."""
        messages = [UserMessage(content="Hello")]
        kwargs = client._build_request_kwargs("claude-3-haiku-20240307", messages, ChatOptions())

        assert "system" not in kwargs

    def test_json_format_appended_as_block(self, client):
        """JSON format instruction appended as additional content block."""
        messages = [
            SystemMessage(content="You are helpful"),
            UserMessage(content="Give JSON"),
        ]
        kwargs = client._build_request_kwargs(
            "claude-3-haiku-20240307", messages, ChatOptions(response_format="json")
        )

        system_blocks = kwargs["system"]
        assert len(system_blocks) == 2
        assert system_blocks[0]["text"] == "You are helpful"
        assert "JSON" in system_blocks[1]["text"]

    def test_json_format_no_system_message(self, client):
        """JSON format creates system blocks even with no system messages."""
        messages = [UserMessage(content="Give JSON")]
        kwargs = client._build_request_kwargs(
            "claude-3-haiku-20240307", messages, ChatOptions(response_format="json")
        )

        system_blocks = kwargs["system"]
        assert len(system_blocks) == 1
        assert "JSON" in system_blocks[0]["text"]

    def test_pydantic_format_appended_as_block(self, client):
        """Pydantic schema instruction appended as additional content block."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str

        messages = [
            SystemMessage(content="You are helpful"),
            UserMessage(content="Give me data"),
        ]
        kwargs = client._build_request_kwargs(
            "claude-3-haiku-20240307",
            messages,
            ChatOptions(response_format=MyModel),
        )

        system_blocks = kwargs["system"]
        assert len(system_blocks) == 2
        assert system_blocks[0]["text"] == "You are helpful"
        assert "schema" in system_blocks[1]["text"].lower()

    def test_ignores_system_message_handling_option(self, client):
        """Anthropic always uses content blocks regardless of option."""
        messages = [
            SystemMessage(content="First"),
            SystemMessage(content="Second"),
            UserMessage(content="Hello"),
        ]
        kwargs = client._build_request_kwargs(
            "claude-3-haiku-20240307",
            messages,
            ChatOptions(system_message_handling="merge"),
        )

        # Still content blocks, not merged
        system_blocks = kwargs["system"]
        assert len(system_blocks) == 2
        assert system_blocks[0]["text"] == "First"
        assert system_blocks[1]["text"] == "Second"
