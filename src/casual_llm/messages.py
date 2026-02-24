"""
OpenAI-compatible message models for LLM conversations.

These models follow the OpenAI chat completion API format and can be used
with any provider that implements the LLMProvider protocol.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, field_validator

from casual_llm.usage import Usage


class TextContent(BaseModel):
    """Text content block for multimodal messages."""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content block for multimodal messages.

    Supports both URL strings and base64-encoded image data.

    Examples:
        # URL source
        ImageContent(type="image", source="https://example.com/image.jpg")

        # Base64 source (dict format)
        ImageContent(
            type="image",
            source={"type": "base64", "data": "...base64..."},
            media_type="image/png"
        )
    """

    type: Literal["image"] = "image"
    source: str | dict[str, str]
    """URL string or dict with {type: "base64", data: "..."} format."""
    media_type: str = "image/jpeg"
    """MIME type of the image (e.g., image/jpeg, image/png, image/gif, image/webp)."""

    @field_validator("media_type")
    @classmethod
    def _validate_media_type(cls, v: str) -> str:
        if not v.startswith("image/"):
            raise ValueError(
                f"media_type must be an image MIME type (got {v!r}). "
                f"Expected format: image/jpeg, image/png, image/gif, image/webp, etc."
            )
        return v


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
    """Message from the user.

    Supports both simple text content and multimodal content (text + images).

    Examples:
        # Simple text content
        UserMessage(content="Hello, world!")

        # Multimodal content
        UserMessage(content=[
            TextContent(type="text", text="What's in this image?"),
            ImageContent(type="image", source="https://example.com/image.jpg")
        ])
    """

    role: Literal["user"] = "user"
    content: str | list[TextContent | ImageContent] | None


class StreamChunk(BaseModel):
    """A chunk of streamed response content from an LLM provider.

    The optional ``usage`` field is populated on the final chunk by providers
    that report token usage during streaming.
    """

    content: str
    finish_reason: str | None = None
    usage: Usage | None = None


ChatMessage: TypeAlias = AssistantMessage | SystemMessage | ToolResultMessage | UserMessage
"""Type alias for any chat message type (user, assistant, system, or tool result)."""
