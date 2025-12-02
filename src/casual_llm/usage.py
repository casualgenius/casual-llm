"""
Usage statistics models for LLM API calls.

Tracks token usage for monitoring costs and rate limits.
"""

from pydantic import BaseModel, Field, computed_field


class Usage(BaseModel):
    """
    Token usage statistics for an LLM API call.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt
        completion_tokens: Number of tokens in the generated response
        total_tokens: Total tokens used (prompt + completion) - computed automatically
    """

    prompt_tokens: int = Field(..., ge=0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., ge=0, description="Number of tokens in the completion")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        """Total tokens used (prompt + completion) - computed automatically."""
        return self.prompt_tokens + self.completion_tokens


__all__ = ["Usage"]
