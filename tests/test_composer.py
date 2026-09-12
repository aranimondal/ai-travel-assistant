"""Grounding rules of the answer composer and the prompt context builders."""

from __future__ import annotations

from travel_assistant.composer import compose
from travel_assistant.context import NO_CONTEXT, format_kb_context, format_tool_context
from travel_assistant.conversation import Conversation
from travel_assistant.kb.retriever import RetrievedChunk
from travel_assistant.mcp_client import ToolCall

CHUNK = RetrievedChunk(
    text="Gardens by the Bay has two cooled indoor conservatories.",
    title="Wikivoyage: Singapore/Marina Bay",
    source_url="https://en.wikivoyage.org/wiki/Singapore/Marina_Bay",
    section="See > Gardens and parks",
    score=0.61,
)

WEATHER = ToolCall(
    tool="get_weather_forecast",
    arguments={"location": "Singapore", "days": 2},
    ok=True,
    result={
        "location": {"name": "Singapore"},
        "current": {
            "conditions": "Overcast",
            "temperature_c": 27.2,
            "feels_like_c": 33.3,
            "relative_humidity_pct": 89,
            "wind_speed_kmh": 4.7,
        },
        "daily": [
            {
                "date": "2026-01-05",
                "conditions": "Heavy rain",
                "temp_min_c": 25.0,
                "temp_max_c": 29.0,
                "rain_probability_pct": 90,
                "outdoor_suitability": "indoor",
            },
            {
                "date": "2026-01-06",
                "conditions": "Partly cloudy",
                "temp_min_c": 26.0,
                "temp_max_c": 32.0,
                "rain_probability_pct": 15,
                "outdoor_suitability": "outdoor",
            },
        ],
        "source": "Open-Meteo Forecast API",
    },
)

FAILED = ToolCall(
    tool="convert_currency",
    arguments={"amount": 100, "from_currency": "XXX", "to_currency": "SGD"},
    ok=False,
    error="The exchange-rate service does not publish a XXX->SGD rate.",
)


def test_answer_states_when_the_knowledge_base_has_nothing() -> None:
    answer = compose(
        "What are the 2030 ticket prices?",
        chunks=[],
        tool_calls=[],
        conversation=Conversation(),
        needs_kb=True,
    )
    assert "does not contain enough information" in answer
    assert "$" not in answer, "no invented prices"


def test_answer_labels_live_data_and_lists_both_source_kinds() -> None:
    answer = compose(
        "Plan two days around the weather.",
        chunks=[CHUNK],
        tool_calls=[WEATHER],
        conversation=Conversation(),
        needs_kb=True,
    )

    assert "[Live: MCP tool `get_weather_forecast`]" in answer
    assert "Gardens by the Bay" in answer
    assert "Wikivoyage: Singapore/Marina Bay" in answer
    assert "https://en.wikivoyage.org/wiki/Singapore/Marina_Bay" in answer
    assert "Assistant suggestions" in answer, "LLM/heuristic reasoning must be labelled"
    assert "rain likely" in answer and "good conditions" in answer


def test_failed_tool_is_reported_without_substituting_values() -> None:
    answer = compose(
        "Convert XXX 100 to SGD.",
        chunks=[],
        tool_calls=[FAILED],
        conversation=Conversation(),
        needs_kb=False,
    )
    assert "could not be used" in answer
    assert "No estimated values" in answer
    assert "unavailable" in answer.lower()


def test_remembered_preferences_are_restated_for_recommendations() -> None:
    conversation = Conversation()
    conversation.add_user("Family trip for 3 days with a budget of INR 60,000.")

    answer = compose(
        "Suggest activities.",
        chunks=[CHUNK],
        tool_calls=[],
        conversation=conversation,
        needs_kb=True,
    )
    assert "Preferences carried over" in answer
    assert "3 days" in answer


def test_specific_figures_absent_from_passages_are_flagged() -> None:
    answer = compose(
        "What are the 2026 ticket prices for the Grand Prix paddock club?",
        chunks=[
            RetrievedChunk(
                text="The Singapore Grand Prix is an on-street race held in Marina Bay.",
                title="Wikivoyage: Singapore travel guide",
                source_url="https://en.wikivoyage.org/wiki/Singapore",
                section="Do > Races",
                score=0.51,
            )
        ],
        tool_calls=[],
        conversation=Conversation(),
        needs_kb=True,
    )
    assert "do not contain the specific figures" in answer


def test_no_caveat_when_passages_do_contain_figures() -> None:
    answer = compose(
        "How much does a dorm bed cost?",
        chunks=[
            RetrievedChunk(
                text="Footprints Hostel, 25A Perak Rd. Dorms from $22 per person.",
                title="Wikivoyage: Singapore/Little India",
                source_url="https://en.wikivoyage.org/wiki/Singapore/Little_India",
                section="Sleep > Budget",
                score=0.44,
            )
        ],
        tool_calls=[],
        conversation=Conversation(),
        needs_kb=True,
    )
    assert "do not contain the specific figures" not in answer

def test_answer_prompt_renders_all_context_blocks() -> None:
    from langchain_core.messages import HumanMessage

    from travel_assistant.prompts import ANSWER_PROMPT

    messages = ANSWER_PROMPT.format_messages(
        destination="Singapore",
        preferences_block="TRAVELLER PREFERENCES\n- budget: INR 60,000\n\n",
        kb_context=format_kb_context([CHUNK]),
        tool_context=format_tool_context([WEATHER]),
        question="Plan three days around the weather.",
        history=[HumanMessage(content="earlier turn")],
    )

    system, *rest = messages
    assert "Singapore" in system.content
    # The system prompt states the rules; the evidence itself belongs in the user turn.
    assert "Gardens by the Bay" not in system.content
    assert len(rest) == 2, "history message plus the current question"

    user = rest[-1].content
    assert "TRAVELLER PREFERENCES" in user
    assert "KNOWLEDGE BASE CONTEXT" in user
    assert "LIVE TOOL RESULTS" in user
    assert user.rstrip().endswith("Plan three days around the weather.")


def test_kb_context_is_numbered_and_cited() -> None:
    context = format_kb_context([CHUNK])
    assert context.startswith("[1] Wikivoyage: Singapore/Marina Bay")
    assert "section: See > Gardens and parks" in context
    assert "url: https://en.wikivoyage.org/wiki/Singapore/Marina_Bay" in context


def test_empty_kb_context_is_explicit() -> None:
    assert format_kb_context([]) == NO_CONTEXT


def test_tool_context_marks_failures_and_forbids_estimates() -> None:
    context = format_tool_context([FAILED])
    assert "FAILED" in context
    assert "Do not substitute estimated values" in context


def test_tool_context_includes_suitability_for_each_day() -> None:
    context = format_tool_context([WEATHER])
    assert "suitability: indoor" in context
    assert "suitability: outdoor" in context
    assert "Open-Meteo" in context
