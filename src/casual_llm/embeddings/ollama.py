"""
Ollama embedding provider using the official ollama library.
"""

from __future__ import annotations

import logging
from typing import Any

from ollama import AsyncClient

from casual_llm.embeddings.models import EmbeddingData, EmbeddingResponse
from casual_llm.usage import Usage

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider:
    """
    Ollama embedding provider.

    Generates text embeddings using Ollama's local embedding models.
    Uses the official ollama Python library for communication.

    Examples:
        >>> from casual_llm import OllamaEmbeddingProvider
        >>>
        >>> # Initialize with default settings
        >>> provider = OllamaEmbeddingProvider(model="nomic-embed-text")
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
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
    ):
        """
        Initialize Ollama embedding provider.

        Args:
            model: Model name (e.g., "nomic-embed-text", "mxbai-embed-large",
                "all-minilm")
            host: Ollama server URL (e.g., "http://localhost:11434")
            timeout: HTTP request timeout in seconds
        """
        self.model = model
        self.host = host.rstrip("/")  # Remove trailing slashes
        self.timeout = timeout

        # Create async client
        self.client = AsyncClient(host=self.host, timeout=timeout)

        # Usage tracking
        self._last_usage: Usage | None = None

        logger.info(f"OllamaEmbeddingProvider initialized: model={model}, host={host}")

    def get_usage(self) -> Usage | None:
        """
        Get token usage statistics from the last embed() call.

        Returns:
            Usage object with token counts, or None if no calls have been made

        Examples:
            >>> provider = OllamaEmbeddingProvider(model="nomic-embed-text")
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
        Generate embeddings for a list of texts using Ollama.

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
                - usage: Token usage statistics (if available)

        Raises:
            ollama.ResponseError: If the request could not be fulfilled
            ollama.RequestError: If the request was invalid
            httpx.ConnectError: If Ollama server is not running

        Examples:
            >>> provider = OllamaEmbeddingProvider(model="nomic-embed-text")
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
                dimensions=0,
                usage=None,
            )

        # Use provided model or fall back to instance model
        effective_model = model or self.model

        logger.debug(f"Generating embeddings for {len(texts)} texts with model {effective_model}")

        # Ollama's embed API can handle multiple texts in a single call
        # using the 'input' parameter (list of strings)
        request_kwargs: dict[str, Any] = {
            "model": effective_model,
            "input": texts,
        }

        response = await self.client.embed(**request_kwargs)

        # Extract embeddings from response
        # Response has 'embeddings' field which is a list of embedding vectors
        embeddings = response.get("embeddings", [])

        # Build EmbeddingData objects
        embedding_data = [
            EmbeddingData(
                embedding=embedding,
                index=idx,
            )
            for idx, embedding in enumerate(embeddings)
        ]

        # Get dimensions from first embedding
        actual_dimensions = len(embeddings[0]) if embeddings else 0

        # Extract usage statistics if available
        # Ollama returns prompt_eval_count for token usage
        prompt_tokens = response.get("prompt_eval_count", 0)
        if prompt_tokens > 0:
            self._last_usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,  # Embeddings don't have completion tokens
            )
            logger.debug(f"Usage: {prompt_tokens} prompt tokens")
        else:
            # Reset usage if not available
            self._last_usage = None

        logger.debug(
            f"Generated {len(embedding_data)} embeddings with {actual_dimensions} dimensions"
        )

        return EmbeddingResponse(
            data=embedding_data,
            model=effective_model,
            dimensions=actual_dimensions,
            usage=self._last_usage,
        )


__all__ = ["OllamaEmbeddingProvider"]
