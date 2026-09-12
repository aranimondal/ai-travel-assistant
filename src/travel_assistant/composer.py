"""Deterministic answer composer used when no LLM is configured.

It never writes destination facts of its own: every factual sentence is either
a quoted knowledge-base passage or a value returned by an MCP tool. Planning
logic (which day is indoor-friendly, how to sequence a day) is clearly labelled
as an assistant suggestion. This keeps the RAG + MCP pipeline demonstrable and
auditable without credentials.
"""

from __future__ import annotations

import re

from travel_assistant.conversation import Conversation
from travel_assistant.kb.retriever import RetrievedChunk
from travel_assistant.mcp_client import ToolCall

MAX_QUOTE_CHARS = 520

INDOOR_TERMS = ("indoor", "museum", "mall", "shopping", "aquarium", "gallery")
OUTDOOR_TERMS = ("garden", "park", "beach", "walk", "outdoor", "zoo", "river")

DAY_GUIDANCE = {
    "indoor": "rain likely — prioritise indoor options",
    "mixed": "showers possible — keep an indoor fallback",
    "outdoor": "good conditions for outdoor plans",
}

# Questions asking for hard specifics need a caveat when the retrieved passages
# are merely on-topic: travel wikis rarely carry current prices or exact hours.
SPECIFIC_ASK = re.compile(
    r"\b(price|prices|pricing|cost|costs|fee|fees|ticket|tickets|how much|"
    r"opening hours|timings|schedule|availability)\b",
    re.I,
)
NUMERIC_DETAIL = re.compile(r"(\$|S\$|SGD|USD|\d{1,2}[:.]\d{2}\s?(am|pm)?|\d+\s?(am|pm))", re.I)


def _excerpt(text: str, limit: int = MAX_QUOTE_CHARS) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    cut = collapsed[:limit].rsplit(". ", 1)[0]
    return (cut + ".") if len(cut) > limit * 0.5 else collapsed[:limit].rstrip() + "…"


def _mentions(chunk: RetrievedChunk, terms: tuple[str, ...]) -> bool:
    haystack = f"{chunk.section} {chunk.text}".lower()
    return any(term in haystack for term in terms)


def _kb_section(chunks: list[RetrievedChunk], question: str) -> list[str]:
    lines = ["## From the knowledge base"]
    shown = chunks[:4]
    for chunk in shown:
        label = f"**{chunk.title}**"
        if chunk.section:
            label += f" — {chunk.section}"
        lines.append(f"- {label} (relevance {chunk.score:.2f})")
        lines.append(f"  > {_excerpt(chunk.text)}")

    if SPECIFIC_ASK.search(question) and not any(
        NUMERIC_DETAIL.search(chunk.text) for chunk in shown
    ):
        lines.append(
            "- These passages are on the right topic but do not contain the specific figures "
            "asked for, so no price, fee or timing has been stated. Check the official operator "
            "for current values."
        )
    return lines


def _weather_section(call: ToolCall) -> list[str]:
    result = call.result or {}
    location = (result.get("location") or {}).get("name", "the destination")
    current = result.get("current") or {}
    lines = [f"## Current information [Live: MCP tool `{call.tool}`]"]
    if current.get("temperature_c") is not None:
        lines.append(
            f"- Now in {location}: {current.get('conditions')}, {current.get('temperature_c')}°C "
            f"(feels like {current.get('feels_like_c')}°C), humidity "
            f"{current.get('relative_humidity_pct')}%, wind {current.get('wind_speed_kmh')} km/h."
        )
    for day in result.get("daily", []):
        lines.append(
            f"- {day.get('date')}: {day.get('conditions')}, {day.get('temp_min_c')}–"
            f"{day.get('temp_max_c')}°C, rain probability {day.get('rain_probability_pct')}% "
            f"→ best for **{day.get('outdoor_suitability')}** activities."
        )
    return lines


def _currency_section(call: ToolCall) -> list[str]:
    result = call.result or {}
    return [
        f"## Currency conversion [Live: MCP tool `{call.tool}`]",
        f"- {result.get('amount'):,.2f} {result.get('from_currency')} = "
        f"**{result.get('converted_amount'):,.2f} {result.get('to_currency')}** "
        f"(reference rate {result.get('rate')}, published {result.get('rate_date')}).",
    ]


def _failed_tools_section(failed: list[ToolCall]) -> list[str]:
    return ["## Unavailable current information"] + [
        f"- MCP tool `{call.tool}` could not be used: {call.error} "
        "No estimated values have been substituted."
        for call in failed
    ]


def _sources_section(chunks: list[RetrievedChunk], tool_calls: list[ToolCall]) -> list[str]:
    lines = ["## Sources"]
    seen: set[str] = set()
    for chunk in chunks[:4]:
        key = f"{chunk.title}|{chunk.source_url}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- Knowledge base: {chunk.title}"
            + (f" — {chunk.source_url}" if chunk.source_url else "")
        )
    for call in tool_calls:
        state = "live result" if call.ok else "unavailable"
        source = (call.result or {}).get("source", "") if call.ok else ""
        lines.append(f"- MCP tool `{call.tool}` ({state})" + (f" — {source}" if source else ""))
    return lines


def _itinerary_suggestions(chunks: list[RetrievedChunk], weather: ToolCall | None) -> list[str]:
    """Weather-aware day plan assembled only from retrieved KB sections."""
    days = ((weather.result or {}).get("daily") or []) if weather and weather.ok else []
    if not days or not chunks:
        return []

    indoor = [c for c in chunks if _mentions(c, INDOOR_TERMS)]
    outdoor = [c for c in chunks if _mentions(c, OUTDOOR_TERMS)]

    lines = [
        "## Assistant suggestions (weather-aware day plan)",
        "_Sequencing is the assistant's recommendation. The places come from the "
        "knowledge-base passages above; the weather comes from the MCP tool._",
    ]
    used: set[str] = set()
    for index, day in enumerate(days, start=1):
        suitability = day.get("outdoor_suitability", "mixed")
        pool = (indoor if suitability == "indoor" else outdoor) or chunks
        # Prefer a section not already assigned to an earlier day.
        pick = next((c for c in pool if (c.section or c.title) not in used), pool[0])
        focus = pick.section or pick.title
        used.add(focus)
        lines.append(
            f"- **Day {index} ({day.get('date')})** — {DAY_GUIDANCE[suitability]}. "
            f"Suggested focus from the knowledge base: _{focus}_ [{pick.title}]."
        )
    return lines


def compose(
    question: str,
    *,
    chunks: list[RetrievedChunk],
    tool_calls: list[ToolCall],
    conversation: Conversation,
    needs_kb: bool,
) -> str:
    """Build a grounded answer from retrieved passages and tool results."""
    successful = [call for call in tool_calls if call.ok]
    failed = [call for call in tool_calls if not call.ok]
    weather = next((c for c in successful if c.tool == "get_weather_forecast"), None)
    currency = next((c for c in successful if c.tool == "convert_currency"), None)

    parts: list[str] = [
        "_Composed without a language model: the sections below quote retrieved "
        "knowledge-base passages and MCP tool output directly._"
    ]

    if conversation.preferences and needs_kb:
        # Only worth restating when the preferences actually shaped a recommendation.
        preferences = ", ".join(
            f"{key.replace('_', ' ')}: {value}"
            for key, value in sorted(conversation.preferences.items())
        )
        parts += ["", f"**Preferences carried over:** {preferences}"]

    if needs_kb:
        parts += [""]
        parts += (
            _kb_section(chunks, question)
            if chunks
            else [
                "## From the knowledge base",
                "- The knowledge base does not contain enough information to answer this "
                "reliably, so nothing has been invented to fill the gap. Try rephrasing, "
                "or add a source covering this topic and rebuild the index.",
            ]
        )

    if weather:
        parts += [""] + _weather_section(weather)
    if currency:
        parts += [""] + _currency_section(currency)
    if failed:
        parts += [""] + _failed_tools_section(failed)

    suggestions = _itinerary_suggestions(chunks, weather)
    if suggestions:
        parts += [""] + suggestions

    if chunks or tool_calls:
        parts += [""] + _sources_section(chunks, tool_calls)

    return "\n".join(parts).strip()
