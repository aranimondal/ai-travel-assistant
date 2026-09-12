"""Conversation memory: preferences and follow-up handling."""

from __future__ import annotations

from travel_assistant.conversation import Conversation


def test_preferences_are_extracted_and_normalised() -> None:
    conversation = Conversation()
    conversation.add_user(
        "I am travelling with my kids for three days and my budget is INR 80,000."
    )

    assert conversation.preferences["travelling_with"] == "kids"
    assert conversation.preferences["trip_length"] == "3 days"
    assert conversation.preferences["budget"] == "INR 80,000"


def test_preferences_survive_beyond_the_history_window() -> None:
    conversation = Conversation(max_turns=1)
    conversation.add_user("We are vegetarian and interested in cultural sights.")
    conversation.add_assistant("Noted.")
    for i in range(4):
        conversation.add_user(f"Question {i}")
        conversation.add_assistant(f"Answer {i}")

    assert "vegetarian" in conversation.preferences["constraints"]
    assert "cultural" in conversation.preferences["interests"]
    assert len(conversation.history()) <= 2


def test_short_follow_up_is_expanded_with_the_previous_question() -> None:
    conversation = Conversation()
    conversation.add_user("Suggest outdoor attractions in Singapore.")
    conversation.add_assistant("...")
    conversation.add_user("What about indoors?")

    expanded = conversation.contextualise("What about indoors?")
    assert expanded.startswith("Suggest outdoor attractions in Singapore.")
    assert expanded.endswith("What about indoors?")


def test_self_contained_question_is_not_expanded() -> None:
    conversation = Conversation()
    conversation.add_user("What is the weather in Singapore?")
    conversation.add_assistant("...")
    conversation.add_user("Convert INR 50,000 to SGD.")

    # Prefixing an unrelated previous question would pollute retrieval.
    assert conversation.contextualise("Convert INR 50,000 to SGD.") == "Convert INR 50,000 to SGD."


def test_preferences_block_is_empty_without_preferences() -> None:
    conversation = Conversation()
    conversation.add_user("Hello")
    assert conversation.preferences_block() == ""


def test_reset_clears_history_and_preferences() -> None:
    conversation = Conversation()
    conversation.add_user("Family trip for 3 days with a budget of USD 900.")
    conversation.reset()
    assert not conversation.messages
    assert not conversation.preferences
