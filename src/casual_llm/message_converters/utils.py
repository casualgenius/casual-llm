"""
Shared utilities for message converters.
"""

from __future__ import annotations

from casual_llm.messages import ChatMessage, SystemMessage


def merge_system_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """
    Merge all system messages into a single SystemMessage at the front.

    Collects all SystemMessage contents, strips whitespace, filters empties,
    and joins with double newlines. Non-system messages preserve their
    original order.

    Returns the original list unchanged if there are 0 or 1 non-empty
    system messages.

    Args:
        messages: List of ChatMessage objects

    Returns:
        New list with at most one SystemMessage at the front, followed by
        all non-system messages in their original order.
    """
    system_parts: list[str] = []
    non_system: list[ChatMessage] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            stripped = msg.content.strip()
            if stripped:
                system_parts.append(stripped)
        else:
            non_system.append(msg)

    if len(system_parts) <= 1:
        return messages

    merged_content = "\n\n".join(system_parts)
    return [SystemMessage(content=merged_content), *non_system]
