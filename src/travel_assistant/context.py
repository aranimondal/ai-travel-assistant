"""Format retrieved chunks and MCP tool results into prompt context."""

from __future__ import annotations

from typing import Any

from travel_assistant.kb.retriever import RetrievedChunk
from travel_assistant.mcp_client import ToolCall

NO_CONTEXT = "(no relevant knowledge-base passages were retrieved for this question)"
NO_TOOLS = "(no MCP tool was called for this question)"


def format_kb_context(chunks: list[RetrievedChunk], *, max_chars: int = 7000) -> str:
    """Render chunks as numbered, cited passages within a character budget."""
    if not chunks:
        return NO_CONTEXT

    blocks: list[str] = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        header = f"[{index}] {chunk.title}"
        if chunk.section:
            header += f" — section: {chunk.section}"
        if chunk.source_url:
            header += f"\n    url: {chunk.source_url}"
        block = f"{header}\n{chunk.text.strip()}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _weather_lines(result: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    location = result.get("location", {})
    name = location.get("name", "the destination")
    current = result.get("current", {})
    if current.get("temperature_c") is not None:
        lines.append(
            f"  current in {name}: {current.get('conditions')}, "
            f"{current.get('temperature_c')}°C (feels like {current.get('feels_like_c')}°C), "
            f"humidity {current.get('relative_humidity_pct')}%, "
            f"wind {current.get('wind_speed_kmh')} km/h, observed {current.get('observed_at')}"
        )
    for day in result.get("daily", []):
        lines.append(
            f"  {day.get('date')}: {day.get('conditions')}, "
            f"{day.get('temp_min_c')}–{day.get('temp_max_c')}°C, "
            f"rain probability {day.get('rain_probability_pct')}%, "
            f"precipitation {day.get('precipitation_mm')} mm, "
            f"rain expected: {'yes' if day.get('rain_expected') else 'no'}, "
            f"suitability: {day.get('outdoor_suitability')}"
        )
    lines.append(f"  source: {result.get('source')} ({result.get('source_url')})")
    return lines


def _currency_lines(result: dict[str, Any]) -> list[str]:
    lines = [
        f"  {result.get('amount'):,.2f} {result.get('from_currency')} = "
        f"{result.get('converted_amount'):,.2f} {result.get('to_currency')} "
        f"(rate {result.get('rate')}, published {result.get('rate_date')})"
    ]
    if result.get("note"):
        lines.append(f"  note: {result['note']}")
    lines.append(f"  source: {result.get('source')} ({result.get('source_url')})")
    return lines


def format_tool_context(tool_calls: list[ToolCall]) -> str:
    """Render MCP results (and failures) as compact, labelled facts."""
    if not tool_calls:
        return NO_TOOLS

    blocks: list[str] = []
    for call in tool_calls:
        if not call.ok:
            blocks.append(
                f"{call.tool} (MCP): FAILED — {call.error}\n"
                "  Do not substitute estimated values for this tool's data."
            )
            continue

        result = call.result or {}
        blocks.append(
            f"{call.tool} (MCP), arguments {call.arguments}:\n"
            + "\n".join(
                _weather_lines(result)
                if call.tool == "get_weather_forecast"
                else _currency_lines(result)
            )
        )
    return "\n\n".join(blocks)
