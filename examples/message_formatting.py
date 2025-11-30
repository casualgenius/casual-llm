"""
Example showing different message types and formatting.

This example demonstrates how to use all message types available in casual-llm.
"""

from casual_llm import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
    AssistantToolCall,
    AssistantToolCallFunction,
    ChatMessage,
)


def main():
    # System message - sets the assistant's behavior
    system_msg = SystemMessage(
        content="You are a helpful weather assistant with access to weather tools."
    )

    # User message - from the human
    user_msg = UserMessage(
        content="What's the weather like in Paris?"
    )

    # Assistant message with tool call
    assistant_msg_with_tool = AssistantMessage(
        content="Let me check the weather in Paris for you.",
        tool_calls=[
            AssistantToolCall(
                id="call_abc123",
                type="function",
                function=AssistantToolCallFunction(
                    name="get_weather",
                    arguments='{"city": "Paris", "units": "celsius"}'
                )
            )
        ]
    )

    # Tool result message - result from tool execution
    tool_result = ToolResultMessage(
        role="tool",
        name="get_weather",
        tool_call_id="call_abc123",
        content='{"temperature": 18, "condition": "partly cloudy", "humidity": 65}'
    )

    # Assistant message with final answer
    assistant_final = AssistantMessage(
        content="The weather in Paris is currently 18°C and partly cloudy with 65% humidity."
    )

    # Combine into a conversation
    conversation: list[ChatMessage] = [
        system_msg,
        user_msg,
        assistant_msg_with_tool,
        tool_result,
        assistant_final,
    ]

    # Display the conversation
    print("=== Example Conversation ===\n")
    for i, msg in enumerate(conversation, 1):
        print(f"{i}. {msg.role.upper()}: {msg.content if hasattr(msg, 'content') else 'N/A'}")

        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                print(f"   Tool Call: {tool_call.function.name}({tool_call.function.arguments})")

        if isinstance(msg, ToolResultMessage):
            print(f"   Tool Result: {msg.content}")

        print()

    # Serialize messages (useful for storage or API calls)
    print("\n=== Serialized Messages ===\n")
    for msg in conversation:
        print(msg.model_dump_json(indent=2))
        print()


if __name__ == "__main__":
    main()
