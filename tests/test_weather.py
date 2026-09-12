"""Weather helpers: the indoor/outdoor hint drives the combined itinerary."""

from __future__ import annotations

import pytest

from travel_mcp.weather import _outdoor_suitability, describe_code


@pytest.mark.parametrize(
    ("code", "probability", "expected"),
    [
        (0, 0, "outdoor"),
        (2, 20, "outdoor"),
        (61, 45, "mixed"),
        (51, 10, "mixed"),
        (65, 80, "indoor"),
        (95, 10, "indoor"),
        (3, 95, "indoor"),
        (None, None, "outdoor"),
    ],
)
def test_outdoor_suitability(code: int | None, probability: int | None, expected: str) -> None:
    assert _outdoor_suitability(code, probability) == expected


def test_known_and_unknown_weather_codes_are_described() -> None:
    assert describe_code(0) == "Clear sky"
    assert describe_code(95) == "Thunderstorm"
    assert describe_code(None) == "Unknown"
    assert describe_code(7) == "WMO code 7"
