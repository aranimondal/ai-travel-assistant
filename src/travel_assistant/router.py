"""Intent routing: decide which sources a question needs.

Rule-based on purpose. Routing must stay predictable and testable, must work
without an LLM, and the signals here (money amounts, weather words, itinerary
words) are unambiguous enough that a model round-trip would add latency and
non-determinism without improving accuracy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

WEATHER_TERMS = (
    "weather",
    "forecast",
    "rain",
    "raining",
    "rainy",
    "temperature",
    "humid",
    "humidity",
    "hot",
    "storm",
    "thunder",
    "sunny",
    "climate today",
    "wind",
    "monsoon",
    "umbrella",
)

CURRENCY_TERMS = ("convert", "exchange rate", "currency", "how much is", "in rupees", "worth in")

KB_TERMS = (
    "attraction",
    "attractions",
    "visit",
    "see",
    "do",
    "neighbourhood",
    "neighborhood",
    "district",
    "itinerary",
    "plan",
    "trip",
    "travelling",
    "traveling",
    "travel",
    "transport",
    "travel around",
    "mrt",
    "bus",
    "taxi",
    "food",
    "eat",
    "hawker",
    "culture",
    "cultural",
    "temple",
    "museum",
    "family",
    "children",
    "kids",
    "indoor",
    "outdoor",
    "shopping",
    "nightlife",
    "beach",
    "park",
    "zoo",
    "tips",
    "sightseeing",
    "activities",
    "recommend",
    "suggest",
)

# "60,000 INR", "INR 60000", "$200", "200 SGD", "60k rupees"
CURRENCY_SYMBOLS = {"₹": "INR", "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "s$": "SGD"}
CURRENCY_WORDS = {
    "rupee": "INR",
    "rupees": "INR",
    "inr": "INR",
    "dollar": "USD",
    "dollars": "USD",
    "usd": "USD",
    "sgd": "SGD",
    "singapore dollar": "SGD",
    "singapore dollars": "SGD",
    "euro": "EUR",
    "euros": "EUR",
    "eur": "EUR",
    "pound": "GBP",
    "pounds": "GBP",
    "gbp": "GBP",
    "yen": "JPY",
    "jpy": "JPY",
    "aed": "AED",
    "aud": "AUD",
    "cad": "CAD",
    "myr": "MYR",
    "thb": "THB",
    "idr": "IDR",
}

AMOUNT_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{2,3})+|\d+(?:\.\d+)?)\s*(k|lakh|lakhs)?", re.I)
DAYS_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven)[\s-]*(?:day|days|night|nights)\b", re.I
)
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}


@dataclass(slots=True)
class CurrencyRequest:
    amount: float | None = None
    from_currency: str | None = None
    to_currency: str | None = None


@dataclass(slots=True)
class Intent:
    needs_kb: bool = False
    needs_weather: bool = False
    needs_currency: bool = False
    forecast_days: int = 0
    currency: CurrencyRequest = field(default_factory=CurrencyRequest)

    @property
    def uses_mcp(self) -> bool:
        return self.needs_weather or self.needs_currency

    @property
    def label(self) -> str:
        parts = []
        if self.needs_kb:
            parts.append("knowledge base")
        if self.needs_weather:
            parts.append("weather tool")
        if self.needs_currency:
            parts.append("currency tool")
        return " + ".join(parts) if parts else "knowledge base"


def parse_amount(text: str) -> float | None:
    """Extract a monetary amount, understanding "60,000", "60k" and "1.2 lakh"."""
    match = AMOUNT_RE.search(text)
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    suffix = (match.group(2) or "").lower()
    if suffix == "k":
        value *= 1_000
    elif suffix in {"lakh", "lakhs"}:
        value *= 100_000
    return value


def currency_codes(text: str) -> list[str]:
    """Currency codes in the order they appear in the text."""
    lowered = text.lower()
    found: list[tuple[int, str]] = []

    for symbol, code in CURRENCY_SYMBOLS.items():
        index = lowered.find(symbol)
        if index != -1:
            found.append((index, code))

    for word, code in CURRENCY_WORDS.items():
        for match in re.finditer(rf"\b{re.escape(word)}\b", lowered):
            found.append((match.start(), code))

    ordered: list[str] = []
    for _, code in sorted(found):
        if code not in ordered:
            ordered.append(code)
    return ordered


def _forecast_days(text: str) -> int:
    match = DAYS_RE.search(text)
    if match:
        token = match.group(1).lower()
        days = WORD_NUMBERS.get(token)
        if days is None:
            try:
                days = int(token)
            except ValueError:
                days = 3
        return max(1, min(days, 7))
    if re.search(r"\btomorrow\b", text, re.I):
        return 2
    if re.search(r"\b(next week|this week|upcoming week)\b", text, re.I):
        return 7
    return 3


def _contains(text: str, terms: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def classify(question: str, *, destination_currency: str = "SGD") -> Intent:
    """Decide which sources a question needs and extract tool arguments."""
    text = question.lower().strip()
    intent = Intent()

    intent.needs_weather = _contains(text, WEATHER_TERMS) or bool(
        re.search(r"\b(indoor|outdoor)\b.*\b(tomorrow|today|this week|next week)\b", text)
    )

    codes = currency_codes(text)
    amount = parse_amount(text)
    money_words = _contains(text, CURRENCY_TERMS) or bool(
        re.search(r"\bbudget\b|\bcost\b.*\bin\b\s*[a-z]{3}\b", text)
    )
    # A conversion needs an explicit currency mention; "3 days" alone must not trigger it.
    intent.needs_currency = bool(codes) and (money_words or amount is not None)

    if intent.needs_currency:
        source = codes[0] if codes else None
        target = codes[1] if len(codes) > 1 else destination_currency
        if source == target and len(codes) > 1:
            target = destination_currency
        intent.currency = CurrencyRequest(
            amount=amount, from_currency=source, to_currency=target
        )

    intent.needs_kb = _contains(text, KB_TERMS)
    if not intent.needs_kb and not intent.uses_mcp:
        # Unclassified questions default to the knowledge base, which is
        # allowed to answer "not enough information".
        intent.needs_kb = True

    if intent.needs_weather:
        intent.forecast_days = _forecast_days(text)

    return intent
