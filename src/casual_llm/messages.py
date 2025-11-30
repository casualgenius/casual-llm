"""
OpenAI-compatible message models for LLM conversations.

These models follow the OpenAI chat completion API format and can be used
with any provider that implements the LLMProvider protocol.
"""

from typing import Literal, TypeAlias

from pydantic import BaseModel


class AssistantToolCallFunction(BaseModel):
    """Function call within an assistant tool call."""

    name: str
    arguments: str


class AssistantToolCall(BaseModel):
    """Tool call made by the assistant."""

    id: str
    type: Literal["function"] = "function"
    function: AssistantToolCallFunction


class AssistantMessage(BaseModel):
    """Message from the AI assistant."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[AssistantToolCall] | None = None


class SystemMessage(BaseModel):
    """System prompt message that sets the assistant's behavior."""

    role: Literal["system"] = "system"
    content: str


class ToolResultMessage(BaseModel):
    """Result from a tool/function call execution."""

    role: Literal["tool"] = "tool"
    name: str
    tool_call_id: str
    content: str


class UserMessage(BaseModel):
    """Message from the user."""

    role: Literal["user"] = "user"
    content: str | None


ChatMessage: TypeAlias = AssistantMessage | SystemMessage | ToolResultMessage | UserMessage
"""Type alias for any chat message type (user, assistant, system, or tool result)."""
