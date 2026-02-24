"""
Tests for ChatOptions merge behavior in Model._merge_options.

Covers TST-009: Verifies precedence rules (call-level > model-level > defaults).
"""

import pytest

from casual_llm.config import ChatOptions
from casual_llm.model import Model
from casual_llm.providers import OllamaClient
from casual_llm.tools import Tool


class TestMergeOptions:
    """Test Model._merge_options precedence and behavior."""

    @pytest.fixture
    def client(self):
        return OllamaClient(host="http://localhost:11434")

    def test_no_defaults_no_per_call(self, client):
        """No default options and no per-call options returns bare ChatOptions."""
        model = Model(client=client, name="test")
        merged = model._merge_options(None)

        assert merged.temperature is None
        assert merged.max_tokens is None
        assert merged.response_format == "text"
        assert merged.extra == {}

    def test_defaults_only(self, client):
        """Model defaults used when no per-call options provided."""
        defaults = ChatOptions(temperature=0.7, max_tokens=500)
        model = Model(client=client, name="test", default_options=defaults)

        merged = model._merge_options(None)

        assert merged.temperature == 0.7
        assert merged.max_tokens == 500

    def test_per_call_only(self, client):
        """Per-call options used directly when no model defaults."""
        model = Model(client=client, name="test")
        per_call = ChatOptions(temperature=0.1, max_tokens=100)

        merged = model._merge_options(per_call)

        assert merged.temperature == 0.1
        assert merged.max_tokens == 100

    def test_per_call_overrides_defaults(self, client):
        """Per-call non-None values override model defaults."""
        defaults = ChatOptions(temperature=0.7, max_tokens=500, top_p=0.9)
        model = Model(client=client, name="test", default_options=defaults)

        per_call = ChatOptions(temperature=0.1)
        merged = model._merge_options(per_call)

        # Overridden
        assert merged.temperature == 0.1
        # Inherited from defaults (per-call is None)
        assert merged.max_tokens == 500
        assert merged.top_p == 0.9

    def test_per_call_none_does_not_override(self, client):
        """Per-call None values fall back to model defaults."""
        defaults = ChatOptions(temperature=0.7, seed=42)
        model = Model(client=client, name="test", default_options=defaults)

        per_call = ChatOptions(temperature=None, seed=None, max_tokens=100)
        merged = model._merge_options(per_call)

        # None per-call -> use defaults
        assert merged.temperature == 0.7
        assert merged.seed == 42
        # Explicit per-call value
        assert merged.max_tokens == 100

    def test_extra_dicts_are_merged(self, client):
        """Extra dicts from defaults and per-call are merged."""
        defaults = ChatOptions(extra={"keep_alive": "10m", "shared_key": "default"})
        model = Model(client=client, name="test", default_options=defaults)

        per_call = ChatOptions(extra={"logprobs": True, "shared_key": "override"})
        merged = model._merge_options(per_call)

        # Both keys present
        assert merged.extra["keep_alive"] == "10m"
        assert merged.extra["logprobs"] is True
        # Per-call wins on conflict
        assert merged.extra["shared_key"] == "override"

    def test_response_format_override(self, client):
        """response_format can be overridden per-call."""
        defaults = ChatOptions(response_format="json")
        model = Model(client=client, name="test", default_options=defaults)

        per_call = ChatOptions(response_format="text")
        merged = model._merge_options(per_call)

        # Per-call "text" is not None, so it overrides
        assert merged.response_format == "text"

    def test_tools_override(self, client):
        """Per-call tools override model-level tools."""
        tool_a = Tool(name="tool_a", description="Tool A")
        tool_b = Tool(name="tool_b", description="Tool B")

        defaults = ChatOptions(tools=[tool_a])
        model = Model(client=client, name="test", default_options=defaults)

        per_call = ChatOptions(tools=[tool_b])
        merged = model._merge_options(per_call)

        assert len(merged.tools) == 1
        assert merged.tools[0].name == "tool_b"

    def test_tools_none_inherits_defaults(self, client):
        """Per-call tools=None falls back to model default tools."""
        tool_a = Tool(name="tool_a", description="Tool A")

        defaults = ChatOptions(tools=[tool_a])
        model = Model(client=client, name="test", default_options=defaults)

        per_call = ChatOptions(temperature=0.5)  # tools=None
        merged = model._merge_options(per_call)

        assert merged.tools is not None
        assert len(merged.tools) == 1
        assert merged.tools[0].name == "tool_a"
        assert merged.temperature == 0.5
