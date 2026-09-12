"""Intent routing: the assistant must pick the right sources for each question."""

from __future__ import annotations

import pytest

from travel_assistant.router import classify, currency_codes, parse_amount


@pytest.mark.parametrize(
    "question",
    [
        "What are the must-visit attractions in Singapore?",
        "Which neighbourhoods are suitable for cultural experiences?",
        "How can a tourist travel around Singapore?",
        "Suggest activities for a family with children.",
        "Create a three-day sightseeing itinerary.",
        "What indoor attractions can I visit?",
    ],
)
def test_destination_questions_use_only_the_knowledge_base(question: str) -> None:
    intent = classify(question)
    assert intent.needs_kb
    assert not intent.uses_mcp, "MCP must not answer questions the knowledge base covers"


@pytest.mark.parametrize(
    ("question", "days"),
    [
        ("What is the weather in Singapore?", 3),
        ("What is the forecast for the next three days?", 3),
        ("Is rain expected during my trip?", 3),
        ("Should I plan indoor or outdoor activities tomorrow?", 2),
        ("What is the forecast for the next 5 days?", 5),
    ],
)
def test_weather_questions_select_the_weather_tool(question: str, days: int) -> None:
    intent = classify(question)
    assert intent.needs_weather
    assert intent.forecast_days == days
    assert not intent.needs_currency


@pytest.mark.parametrize(
    ("question", "amount", "source", "target"),
    [
        ("Convert INR 50,000 to SGD.", 50_000, "INR", "SGD"),
        ("How much is 200 SGD in INR?", 200, "SGD", "INR"),
        ("Convert my travel budget from USD 1,500 to Singapore dollars", 1_500, "USD", "SGD"),
        ("I have a budget of 60k INR", 60_000, "INR", "SGD"),
    ],
)
def test_currency_questions_extract_tool_arguments(
    question: str, amount: float, source: str, target: str
) -> None:
    intent = classify(question)
    assert intent.needs_currency
    assert intent.currency.amount == amount
    assert intent.currency.from_currency == source
    assert intent.currency.to_currency == target


def test_combined_question_uses_knowledge_base_and_weather_tool() -> None:
    intent = classify(
        "Create a three-day Singapore itinerary for next week and adjust it according to the "
        "weather forecast."
    )
    assert intent.needs_kb
    assert intent.needs_weather
    assert not intent.needs_currency
    assert intent.label == "knowledge base + weather tool"


def test_combined_question_uses_knowledge_base_and_currency_tool() -> None:
    intent = classify(
        "I have a budget of INR 60,000. Convert it to SGD and suggest a 3-day itinerary."
    )
    assert intent.needs_kb
    assert intent.needs_currency
    assert intent.currency.amount == 60_000


def test_day_counts_do_not_trigger_a_currency_conversion() -> None:
    intent = classify("Create a three-day sightseeing itinerary with 2 museums.")
    assert not intent.needs_currency


def test_unclassified_question_falls_back_to_the_knowledge_base() -> None:
    # The knowledge base is allowed to answer "not enough information";
    # silently answering from model memory is not.
    assert classify("Tell me something interesting.").needs_kb


@pytest.mark.parametrize(
    ("text", "expected"),
    [("60,000", 60_000), ("60k", 60_000), ("1.5 lakh", 150_000), ("200", 200), ("none", None)],
)
def test_parse_amount(text: str, expected: float | None) -> None:
    assert parse_amount(text) == expected


def test_currency_codes_preserve_order_of_appearance() -> None:
    assert currency_codes("convert 100 usd to sgd")[:2] == ["USD", "SGD"]
