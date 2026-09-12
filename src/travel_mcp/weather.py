"""Open-Meteo weather lookups used by the MCP weather tool.

Open-Meteo is a keyless, non-commercial-friendly forecast API, which keeps the
assignment reproducible without credentials.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SOURCE_NAME = "Open-Meteo Forecast API"
SOURCE_URL = "https://open-meteo.com/"

# Subset of WMO weather codes that matters for "indoor vs outdoor" decisions.
WMO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

WET_CODES = frozenset({51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99})


class WeatherServiceError(RuntimeError):
    """Raised when the upstream forecast service cannot be used."""


def describe_code(code: int | None) -> str:
    if code is None:
        return "Unknown"
    return WMO_CODES.get(int(code), f"WMO code {code}")


def _outdoor_suitability(code: int | None, precipitation_probability: int | None) -> str:
    """Coarse indoor/outdoor hint derived from the forecast numbers."""
    probability = precipitation_probability or 0
    if code is not None and int(code) in {95, 96, 99}:
        return "indoor"
    if probability >= 70 or (code is not None and int(code) in {63, 65, 82}):
        return "indoor"
    if probability >= 40 or (code is not None and int(code) in WET_CODES):
        return "mixed"
    return "outdoor"


def geocode(place: str, *, timeout: float = 15.0) -> dict[str, Any]:
    """Resolve a place name to coordinates."""
    try:
        response = httpx.get(
            GEOCODING_URL,
            params={"name": place, "count": 1, "language": "en", "format": "json"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:  # network / status problems
        raise WeatherServiceError(f"Geocoding request failed: {exc}") from exc

    results = payload.get("results") or []
    if not results:
        raise WeatherServiceError(f"No coordinates found for '{place}'.")

    top = results[0]
    return {
        "name": top["name"],
        "country": top.get("country"),
        "latitude": top["latitude"],
        "longitude": top["longitude"],
        "timezone": top.get("timezone", "auto"),
    }


CURRENT_FIELDS = (
    "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,"
    "wind_speed_10m,precipitation"
)
DAILY_FIELDS = (
    "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
    "precipitation_probability_max,sunrise,sunset"
)


def fetch_forecast(
    *,
    latitude: float,
    longitude: float,
    days: int,
    timezone: str = "auto",
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Fetch current conditions plus a daily forecast."""
    days = max(1, min(int(days), 7))
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "forecast_days": days,
        "current": CURRENT_FIELDS,
        "daily": DAILY_FIELDS,
    }
    try:
        response = httpx.get(FORECAST_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        raise WeatherServiceError(f"Forecast request failed: {exc}") from exc

    current_raw = payload.get("current") or {}
    current_code = current_raw.get("weather_code")
    current = {
        "observed_at": current_raw.get("time"),
        "temperature_c": current_raw.get("temperature_2m"),
        "feels_like_c": current_raw.get("apparent_temperature"),
        "relative_humidity_pct": current_raw.get("relative_humidity_2m"),
        "wind_speed_kmh": current_raw.get("wind_speed_10m"),
        "precipitation_mm": current_raw.get("precipitation"),
        "conditions": describe_code(current_code),
    }

    daily_raw = payload.get("daily") or {}
    dates: list[str] = daily_raw.get("time") or []
    daily: list[dict[str, Any]] = []
    for i, date in enumerate(dates):

        def at(key: str, i: int = i) -> Any:
            values = daily_raw.get(key) or []
            return values[i] if i < len(values) else None

        code = at("weather_code")
        rain_chance = at("precipitation_probability_max")
        daily.append(
            {
                "date": date,
                "conditions": describe_code(code),
                "temp_min_c": at("temperature_2m_min"),
                "temp_max_c": at("temperature_2m_max"),
                "precipitation_mm": at("precipitation_sum"),
                "rain_probability_pct": rain_chance,
                "rain_expected": bool(
                    (rain_chance or 0) >= 40 or (code is not None and int(code) in WET_CODES)
                ),
                "outdoor_suitability": _outdoor_suitability(code, rain_chance),
                "sunrise": at("sunrise"),
                "sunset": at("sunset"),
            }
        )

    return {
        "current": current,
        "daily": daily,
        "units": {
            "temperature": payload.get("current_units", {}).get("temperature_2m", "°C"),
            "wind_speed": payload.get("current_units", {}).get("wind_speed_10m", "km/h"),
            "precipitation": "mm",
        },
        "timezone": payload.get("timezone", timezone),
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": SOURCE_NAME,
        "source_url": SOURCE_URL,
    }
