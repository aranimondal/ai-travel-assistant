"""End-to-end tests for the four required scenarios.

These exercise the real FAISS index and the real MCP server subprocess, so they
need a built knowledge base (``python scripts/build_kb.py``) and outbound
network access. Run them with::

    pytest -m "index and network"
"""

from __future__ import annotations

import pytest

from travel_assistant.assistant import TravelAssistant
from travel_assistant.config import Settings, get_settings

pytestmark = [pytest.mark.index, pytest.mark.network]


@pytest.fixture(scope="module")
def assistant(settings: Settings) -> TravelAssistant:
    if not (settings.faiss_dir / "index.faiss").exists():
        pytest.skip("no FAISS index; run python scripts/build_kb.py first")
    return TravelAssistant(settings)


@pytest.fixture(autouse=True)
def _fresh_conversation(assistant: TravelAssistant):
    assistant.reset()
    yield
    assistant.reset()


def test_mcp_server_exposes_both_required_tools() -> None:
    from travel_assistant.mcp_client import McpToolbox

    status = McpToolbox(get_settings()).status
    if not status["connected"]:
        pytest.skip(f"MCP server unavailable: {status['error']}")
    assert {tool["name"] for tool in status["tools"]} == {
        "get_weather_forecast",
        "convert_currency",
    }


def test_rag_answer_is_grounded_and_cited(assistant: TravelAssistant) -> None:
    answer = assistant.ask("What are the must-visit attractions in Singapore?")

    assert answer.chunks, "retrieval returned nothing"
    assert not answer.tool_calls, "a knowledge-base question must not call MCP tools"
    assert answer.citations
    assert all(citation["title"] for citation in answer.citations)
    assert any(citation["url"].startswith("https://") for citation in answer.citations)


def test_weather_question_uses_the_mcp_tool(assistant: TravelAssistant) -> None:
    answer = assistant.ask("What is the weather forecast in Singapore for the next three days?")

    weather = [call for call in answer.tool_calls if call.tool == "get_weather_forecast"]
    assert len(weather) == 1
    if not weather[0].ok:
        pytest.skip(f"forecast service unavailable: {weather[0].error}")
    assert weather[0].arguments["days"] == 3
    assert len(weather[0].result["daily"]) == 3
    assert "get_weather_forecast" in answer.text


def test_currency_question_uses_the_mcp_tool(assistant: TravelAssistant) -> None:
    answer = assistant.ask("Convert INR 50,000 to SGD.")

    calls = [call for call in answer.tool_calls if call.tool == "convert_currency"]
    assert len(calls) == 1
    if not calls[0].ok:
        pytest.skip(f"exchange-rate service unavailable: {calls[0].error}")
    result = calls[0].result
    assert result["from_currency"] == "INR"
    assert result["to_currency"] == "SGD"
    assert result["converted_amount"] == pytest.approx(50_000 * result["rate"], rel=1e-3)


def test_combined_answer_uses_both_sources(assistant: TravelAssistant) -> None:
    answer = assistant.ask(
        "Create a three-day Singapore itinerary for next week and adjust it according to the "
        "weather forecast."
    )

    assert answer.chunks, "the itinerary must be grounded in the knowledge base"
    weather = [call for call in answer.tool_calls if call.tool == "get_weather_forecast"]
    assert weather, "the forecast must come from the MCP tool"
    assert answer.intent.label == "knowledge base + weather tool"
    if weather[0].ok:
        assert "get_weather_forecast" in answer.text
        assert answer.citations


def test_conversation_retains_preferences_across_turns(assistant: TravelAssistant) -> None:
    assistant.ask("I am travelling with my kids for three days and my budget is INR 80,000.")
    follow_up = assistant.ask("And how much is that budget in SGD?")

    calls = [call for call in follow_up.tool_calls if call.tool == "convert_currency"]
    assert calls, "the follow-up must reuse the remembered budget"
    assert calls[0].arguments["amount"] == 80_000
    assert calls[0].arguments["from_currency"] == "INR"
    assert calls[0].arguments["to_currency"] == "SGD"
    assert assistant.conversation.preferences["travelling_with"] == "kids"


def test_follow_up_question_is_resolved_against_history(assistant: TravelAssistant) -> None:
    assistant.ask("Suggest outdoor activities in Singapore.")
    follow_up = assistant.ask("What about indoors?")

    assert follow_up.chunks, "the follow-up must still retrieve relevant passages"
    assert any("expanded for retrieval" in note for note in follow_up.notes)


def test_out_of_scope_question_is_refused(assistant: TravelAssistant) -> None:
    answer = assistant.ask("How do I apply for a mortgage in Iceland?")

    assert not answer.chunks, "off-topic questions must not retrieve Singapore passages"
    assert "does not contain enough information" in answer.text


def test_empty_question_is_rejected(assistant: TravelAssistant) -> None:
    with pytest.raises(ValueError):
        assistant.ask("   ")
