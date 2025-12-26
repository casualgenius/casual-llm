"""
OpenAI embedding provider (compatible with OpenAI API and compatible services).
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from casual_llm.embeddings.models import EmbeddingData, EmbeddingResponse
from casual_llm.usage import Usage

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider:
    """
    OpenAI-compatible embedding provider.

    Generates text embeddings using OpenAI's embeddings API. Works with
    OpenAI API and compatible services (OpenRouter, Azure OpenAI, etc.).

    Examples:
        >>> from casual_llm import OpenAIEmbeddingProvider
        >>>
        >>> # Initialize with default settings
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
        >>>
        >>> # Check usage
        >>> usage = provider.get_usage()
        >>> if usage:
        ...     print(f"Used {usage.total_tokens} tokens")
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        dimensions: int | None = None,
        timeout: float = 60.0,
    ):
        """
        Initialize OpenAI embedding provider.

        Args:
            model: Model name (e.g., "text-embedding-3-small", "text-embedding-3-large",
                "text-embedding-ada-002")
            api_key: API key (optional, can use OPENAI_API_KEY env var)
            base_url: Base URL for API (e.g., "https://openrouter.ai/api/v1")
            organization: OpenAI organization ID (optional)
            dimensions: Output dimensions for embedding (optional, only supported by
                text-embedding-3-small and text-embedding-3-large for dimension reduction)
            timeout: HTTP request timeout in seconds
        """
        client_kwargs: dict[str, Any] = {"timeout": timeout}

        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization

        self.client = AsyncOpenAI(**client_kwargs)
        self.model = model
        self.dimensions = dimensions

        # Usage tracking
        self._last_usage: Usage | None = None

        logger.info(
            f"OpenAIEmbeddingProvider initialized: model={model}, "
            f"base_url={base_url or 'default'}"
        )

    def get_usage(self) -> Usage | None:
        """
        Get token usage statistics from the last embed() call.

        Returns:
            Usage object with token counts, or None if no calls have been made

        Examples:
            >>> provider = OpenAIEmbeddingProvider(model="text-embedding-3-small")
            >>> await provider.embed(["Hello, world!"])
            >>> usage = provider.get_usage()
            >>> if usage:
            ...     print(f"Used {usage.total_tokens} tokens")
        """
        return self._last_usage

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for a list of texts using OpenAI API.

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
                - usage: Token usage statistics

        Raises:
            openai.OpenAIError: If request fails

        Examples:
            >>> provider = OpenAIEmbeddingProvider(model="text-embedding-3-small")
            >>>
            >>> # Single text embedding
            >>> response = await provider.embed(["Hello, world!"])
            >>> vector = response.data[0].embedding
            >>>
            >>> # Batch embedding
            >>> texts = ["First", "Second", "Third"]
            >>> response = await provider.embed(texts)
            >>> print(f"Got {len(response.data)} embeddings")
        """
        # Handle empty input
        if not texts:
            logger.debug("Empty input list, returning empty response")
            return EmbeddingResponse(
                data=[],
                model=model or self.model,
                dimensions=self.dimensions or 0,
                usage=None,
            )

        # Use provided model or fall back to instance model
        effective_model = model or self.model

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": effective_model,
            "input": texts,
        }

        # Add dimensions if specified (for dimension reduction)
        if self.dimensions is not None:
            request_kwargs["dimensions"] = self.dimensions

        logger.debug(f"Generating embeddings for {len(texts)} texts with model {effective_model}")
        response = await self.client.embeddings.create(**request_kwargs)

        # Extract usage statistics
        if response.usage:
            # OpenAI embeddings API only returns prompt_tokens (no completion tokens)
            self._last_usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=0,
            )
            logger.debug(f"Usage: {response.usage.prompt_tokens} prompt tokens")

        # Convert response data to EmbeddingData objects
        embedding_data = [
            EmbeddingData(
                embedding=item.embedding,
                index=item.index,
            )
            for item in response.data
        ]

        # Get dimensions from first embedding
        actual_dimensions = len(response.data[0].embedding) if response.data else 0

        logger.debug(
            f"Generated {len(embedding_data)} embeddings with {actual_dimensions} dimensions"
        )

        return EmbeddingResponse(
            data=embedding_data,
            model=response.model,
            dimensions=actual_dimensions,
            usage=self._last_usage,
        )


__all__ = ["OpenAIEmbeddingProvider"]
