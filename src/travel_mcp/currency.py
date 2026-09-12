"""Currency conversion backed by the Frankfurter API (ECB reference rates)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

BASE_URL = "https://api.frankfurter.dev/v1"
SOURCE_NAME = "Frankfurter API (European Central Bank reference rates)"
SOURCE_URL = "https://frankfurter.dev/"


class CurrencyServiceError(RuntimeError):
    """Raised when the upstream exchange-rate service cannot be used."""


def _normalise(code: str) -> str:
    code = (code or "").strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise CurrencyServiceError(
            f"'{code}' is not a 3-letter ISO 4217 currency code (for example INR, SGD, USD)."
        )
    return code


def supported_currencies(*, timeout: float = 15.0) -> dict[str, str]:
    try:
        response = httpx.get(f"{BASE_URL}/currencies", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        raise CurrencyServiceError(f"Currency list request failed: {exc}") from exc


def convert(
    amount: float, from_currency: str, to_currency: str, *, timeout: float = 20.0
) -> dict[str, Any]:
    """Convert `amount` from one currency to another using published rates."""
    source = _normalise(from_currency)
    target = _normalise(to_currency)

    try:
        amount = float(amount)
    except (TypeError, ValueError) as exc:
        raise CurrencyServiceError(f"'{amount}' is not a valid numeric amount.") from exc
    if amount < 0:
        raise CurrencyServiceError("Amount must not be negative.")

    if source == target:
        return {
            "amount": amount,
            "from_currency": source,
            "to_currency": target,
            "rate": 1.0,
            "converted_amount": round(amount, 2),
            "rate_date": datetime.now(UTC).date().isoformat(),
            "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "source": SOURCE_NAME,
            "source_url": SOURCE_URL,
            "note": "Source and target currency are identical; no conversion applied.",
        }

    try:
        response = httpx.get(
            f"{BASE_URL}/latest",
            params={"base": source, "symbols": target},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in {400, 404, 422}:
            raise CurrencyServiceError(
                f"The exchange-rate service does not publish a {source}->{target} rate."
            ) from exc
        raise CurrencyServiceError(f"Exchange-rate request failed: {exc}") from exc
    except httpx.HTTPError as exc:
        raise CurrencyServiceError(f"Exchange-rate request failed: {exc}") from exc

    rate = (payload.get("rates") or {}).get(target)
    if rate is None:
        raise CurrencyServiceError(
            f"The exchange-rate service returned no {source}->{target} rate."
        )

    return {
        "amount": amount,
        "from_currency": source,
        "to_currency": target,
        "rate": rate,
        "converted_amount": round(amount * float(rate), 2),
        "rate_date": payload.get("date"),
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": SOURCE_NAME,
        "source_url": SOURCE_URL,
    }
