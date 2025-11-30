"""Utility functions for LLM providers."""

import json
import re
from typing import Any


def extract_json_from_markdown(text: str) -> Any:
    """
    Extract JSON from markdown code blocks or raw text.

    Handles cases where LLMs wrap JSON in ```json...``` blocks.

    Args:
        text: Raw response text from LLM

    Returns:
        Parsed JSON object

    Raises:
        json.JSONDecodeError: If text is not valid JSON
    """
    # Try to find JSON in markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Otherwise try to parse as raw JSON
    return json.loads(text)
