"""
Embedding response models for embedding API calls.

Provides standardized response structures for embedding providers.
"""

from pydantic import BaseModel, Field

from casual_llm.usage import Usage


class EmbeddingData(BaseModel):
    """
    Single embedding result.

    Attributes:
        embedding: The embedding vector as a list of floats
        index: Index of this embedding in the input list
    """

    embedding: list[float] = Field(..., description="The embedding vector")
    index: int = Field(..., ge=0, description="Index of this embedding in the input list")


class EmbeddingResponse(BaseModel):
    """
    Response from an embedding provider.

    Attributes:
        data: List of embedding results
        model: Model used for generating embeddings
        dimensions: Dimension of embedding vectors
        usage: Optional token usage statistics
    """

    data: list[EmbeddingData] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model used for embeddings")
    dimensions: int = Field(..., ge=0, description="Dimension of embedding vectors")
    usage: Usage | None = Field(None, description="Token usage statistics")


__all__ = ["EmbeddingData", "EmbeddingResponse"]
