"""
Base protocol for LLM providers.

Provides a unified interface for different LLM backends (OpenAI, Ollama, etc.)
using standard OpenAI-compatible message formats.
"""

from typing import Protocol, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from casual_llm.messages import ChatMessage, AssistantMessage
    from casual_llm.tools import Tool


class LLMProvider(Protocol):
    """
    Protocol for LLM providers.

    Uses OpenAI-compatible ChatMessage format for all interactions.
    Supports both structured (JSON) and unstructured (text) responses.

    This is a Protocol (PEP 544), meaning any class that implements
    the chat() method with this signature is compatible - no
    inheritance required.

    Examples:
        >>> from casual_llm import LLMProvider, ChatMessage, UserMessage
        >>>
        >>> # Any provider implementing this protocol works
        >>> async def get_response(provider: LLMProvider, prompt: str) -> str:
        ...     messages = [UserMessage(content=prompt)]
        ...     return await provider.chat(messages)
    """

    async def chat(
        self,
        messages: list["ChatMessage"],
        response_format: Literal["json", "text"] = "text",
        max_tokens: int | None = None,
        tools: list["Tool"] | None = None,
    ) -> "str | AssistantMessage":
        """
        Generate a chat response from the LLM.

        Args:
            messages: List of ChatMessage (UserMessage, AssistantMessage, SystemMessage, etc.)
            response_format: Expected response format ("json" or "text")
            max_tokens: Maximum tokens to generate (optional)
            tools: List of tools available for the LLM to call (optional)

        Returns:
            If tools are provided: AssistantMessage (may contain tool_calls)
            If no tools: String response content

        Raises:
            Provider-specific exceptions (httpx.HTTPError, openai.OpenAIError, etc.)
        """
        ...
