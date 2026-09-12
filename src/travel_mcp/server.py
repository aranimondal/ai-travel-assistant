"""MCP server exposing live travel data as tools.

Run standalone (stdio transport):

    python -m travel_mcp.server

The assistant starts this module as a subprocess and talks to it over the MCP
protocol, so the tools are equally usable from any MCP-compatible client
(Claude Desktop, VS Code, MCP Inspector).
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from travel_mcp import currency, weather

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "travel-live-data",
    instructions=(
        "Live travel data for trip planning. Use get_weather_forecast for current conditions "
        "or a short-range forecast, and convert_currency for exchange-rate conversions. "
        "These tools return time-sensitive facts only; destination descriptions come from the "
        "caller's own knowledge base."
    ),
)


@mcp.tool()
def get_weather_forecast(location: str = "Singapore", days: int = 3) -> dict[str, Any]:
    """Get current weather and a daily forecast for a destination.

    Args:
        location: City or place name, e.g. "Singapore".
        days: Number of forecast days to return, 1-7.

    Returns current conditions plus, for each day, temperature range, rain
    probability and an ``outdoor_suitability`` hint ("outdoor", "mixed" or
    "indoor") that callers can use to plan indoor alternatives.
    """
    try:
        place = weather.geocode(location)
        forecast = weather.fetch_forecast(
            latitude=place["latitude"],
            longitude=place["longitude"],
            days=days,
            timezone=place["timezone"] or "auto",
        )
    except weather.WeatherServiceError as exc:
        logger.warning("weather tool failed: %s", exc)
        return {"ok": False, "error": str(exc), "tool": "get_weather_forecast"}

    return {
        "ok": True,
        "tool": "get_weather_forecast",
        "location": {
            "name": place["name"],
            "country": place["country"],
            "latitude": place["latitude"],
            "longitude": place["longitude"],
        },
        **forecast,
    }


@mcp.tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict[str, Any]:
    """Convert an amount between two currencies using published reference rates.

    Args:
        amount: Amount to convert, e.g. 60000.
        from_currency: ISO 4217 code of the source currency, e.g. "INR".
        to_currency: ISO 4217 code of the target currency, e.g. "SGD".
    """
    try:
        result = currency.convert(amount, from_currency, to_currency)
    except currency.CurrencyServiceError as exc:
        logger.warning("currency tool failed: %s", exc)
        return {"ok": False, "error": str(exc), "tool": "convert_currency"}

    return {"ok": True, "tool": "convert_currency", **result}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mcp.run()


if __name__ == "__main__":
    main()
