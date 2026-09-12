"""MCP tool behaviour: result normalisation and failure handling.

The happy paths are covered by ``test_integration.py`` (marked ``network``);
these tests pin the error handling that must never fabricate an answer.
"""

from __future__ import annotations

from typing import Any

import pytest

from travel_assistant.config import Settings
from travel_assistant.mcp_client import McpToolbox, _coerce
from travel_mcp import currency


def test_coerce_unwraps_mcp_text_content_blocks() -> None:
    payload = [{"type": "text", "text": '{"ok": true, "rate": 0.0133}'}]
    assert _coerce(payload) == {"ok": True, "rate": 0.0133}


def test_coerce_joins_multiple_text_blocks() -> None:
    payload = [{"type": "text", "text": "line one"}, {"type": "text", "text": "line two"}]
    assert _coerce(payload) == "line one\nline two"


def test_coerce_passes_through_plain_text() -> None:
    assert _coerce("not json") == "not json"


def test_unknown_tool_is_reported_not_invented(monkeypatch: pytest.MonkeyPatch) -> None:
    toolbox = McpToolbox(Settings())
    monkeypatch.setattr(toolbox, "load", lambda: {})

    call = toolbox.call("get_weather_forecast", {"location": "Singapore"})
    assert not call.ok
    assert "not offered" in (call.error or "")
    assert call.result is None


def test_tool_error_payload_becomes_a_failed_call(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingTool:
        name = "convert_currency"
        description = "stub"

        async def ainvoke(self, _: dict[str, Any]) -> list[dict[str, str]]:
            return [{"type": "text", "text": '{"ok": false, "error": "upstream unavailable"}'}]

    toolbox = McpToolbox(Settings())
    monkeypatch.setattr(toolbox, "load", lambda: {"convert_currency": FailingTool()})

    call = toolbox.call(
        "convert_currency", {"amount": 1, "from_currency": "INR", "to_currency": "SGD"}
    )
    assert not call.ok
    assert call.error == "upstream unavailable"


def test_transport_failure_is_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenTool:
        name = "get_weather_forecast"
        description = "stub"

        async def ainvoke(self, _: dict[str, Any]) -> None:
            raise RuntimeError("stdio pipe closed")

    toolbox = McpToolbox(Settings())
    monkeypatch.setattr(toolbox, "load", lambda: {"get_weather_forecast": BrokenTool()})

    call = toolbox.call("get_weather_forecast", {"location": "Singapore", "days": 3})
    assert not call.ok
    assert "stdio pipe closed" in (call.error or "")


@pytest.mark.parametrize("code", ["", "S", "SGDX", "12A"])
def test_currency_codes_are_validated(code: str) -> None:
    with pytest.raises(currency.CurrencyServiceError):
        currency.convert(100, code, "SGD")


def test_negative_amounts_are_rejected() -> None:
    with pytest.raises(currency.CurrencyServiceError):
        currency.convert(-5, "INR", "SGD")


def test_same_currency_conversion_needs_no_network() -> None:
    result = currency.convert(250, "sgd", "SGD")
    assert result["converted_amount"] == 250.0
    assert result["rate"] == 1.0
