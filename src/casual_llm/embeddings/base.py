"""
Base protocol for embedding providers.

Provides a unified interface for different embedding backends (OpenAI, Ollama, etc.)
for generating text embeddings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from casual_llm.usage import Usage

if TYPE_CHECKING:
    from casual_llm.embeddings.models import EmbeddingResponse


class EmbeddingProvider(Protocol):
    """
    Protocol for embedding providers.

    Provides a consistent interface for generating text embeddings
    from various providers (OpenAI, Ollama, etc.).

    This is a Protocol (PEP 544), meaning any class that implements
    the embed() method with this signature is compatible - no
    inheritance required.

    Examples:
        >>> from casual_llm import EmbeddingProvider, EmbeddingResponse
        >>>
        >>> # Any provider implementing this protocol works
        >>> async def get_embeddings(provider: EmbeddingProvider, texts: list[str]) -> EmbeddingResponse:
        ...     return await provider.embed(texts)
    """

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed. Can contain one or more texts
                for batch embedding.
            model: Optional model override. Uses the provider's default model
                if not specified.

        Returns:
            EmbeddingResponse containing:
                - data: List of EmbeddingData with embedding vectors and indices
                - model: The model used for generating embeddings
                - dimensions: Dimension of the embedding vectors
                - usage: Optional token usage statistics

        Raises:
            Provider-specific exceptions (httpx.HTTPError, openai.OpenAIError, etc.)

        Examples:
            >>> from casual_llm import OpenAIEmbeddingProvider
            >>>
            >>> provider = OpenAIEmbeddingProvider(model="text-embedding-3-small")
            >>>
            >>> # Single text embedding
            >>> response = await provider.embed(["Hello, world!"])
            >>> vector = response.data[0].embedding
            >>> print(f"Dimensions: {response.dimensions}")
            >>>
            >>> # Batch embedding
            >>> texts = ["First document", "Second document", "Third document"]
            >>> response = await provider.embed(texts)
            >>> for item in response.data:
            ...     print(f"Text {item.index}: {len(item.embedding)} dimensions")
        """
        ...

    def get_usage(self) -> Usage | None:
        """
        Get token usage statistics from the last embed() call.

        Returns:
            Usage object with prompt_tokens, completion_tokens, and total_tokens,
            or None if no calls have been made yet.

        Examples:
            >>> provider = OpenAIEmbeddingProvider(model="text-embedding-3-small")
            >>> await provider.embed(["Hello, world!"])
            >>> usage = provider.get_usage()
            >>> if usage:
            ...     print(f"Used {usage.total_tokens} tokens")
        """
        ...


__all__ = ["EmbeddingProvider"]
