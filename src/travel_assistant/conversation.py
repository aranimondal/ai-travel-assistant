"""Conversation state: rolling history plus extracted traveller preferences.

Preferences are tracked separately from the transcript because they must
survive beyond the history window -- the assignment requires that stated
preferences ("travelling with kids", "budget INR 60,000") keep shaping later
answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

PREFERENCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "travelling_with",
        re.compile(
            r"\bwith (?:my )?(kids|children|family|toddler|toddlers|parents|partner|"
            r"spouse|friends)\b",
            re.I,
        ),
    ),
    ("travelling_with", re.compile(r"\b(family) (?:trip|holiday|vacation)\b", re.I)),
    ("travelling_with", re.compile(r"\b(solo) (?:trip|travel|traveller|traveler)\b", re.I)),
    (
        "trip_length",
        re.compile(
            r"\b(\d+|one|two|three|four|five|six|seven)[\s-]*(?:day|days|night|nights)\b", re.I
        ),
    ),
    (
        "budget",
        re.compile(r"\b((?:inr|usd|sgd|eur|gbp)\s?\d[\d,]*(?:\.\d+)?\s?(?:k|lakh|lakhs)?)", re.I),
    ),
    ("budget", re.compile(r"([₹$€£]\s?\d[\d,]*(?:\.\d+)?\s?(?:k|lakh|lakhs)?)", re.I)),
    (
        "budget",
        re.compile(
            r"\b(\d[\d,]*(?:\.\d+)?\s?(?:k|lakh|lakhs)?\s?"
            r"(?:inr|usd|sgd|eur|gbp|rupees|dollars))\b",
            re.I,
        ),
    ),
    (
        "interests",
        re.compile(
            r"\b(cultural|heritage|food|foodie|shopping|nature|museum|nightlife|adventure|"
            r"photography|luxury)\b",
            re.I,
        ),
    ),
    (
        "constraints",
        re.compile(r"\b(vegetarian|vegan|halal|wheelchair|limited walking|avoid crowds)\b", re.I),
    ),
    ("timing", re.compile(r"\b(next week|this weekend|tomorrow|next month)\b", re.I)),
)

MULTI_VALUE_KEYS = {"interests", "constraints"}
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}

# A follow-up is short *and* refers back: "and in SGD?", "what about indoors?".
FOLLOW_UP_MARKERS = re.compile(
    r"\b(and|also|what about|how about|instead|then|there|that|those|it|us|we|same|"
    r"they|them|other|else)\b",
    re.I,
)


@dataclass(slots=True)
class Conversation:
    """Chat transcript plus sticky preferences."""

    max_turns: int = 6
    messages: list[BaseMessage] = field(default_factory=list)
    preferences: dict[str, str] = field(default_factory=dict)

    def add_user(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))
        self._extract_preferences(text)

    def add_assistant(self, text: str) -> None:
        self.messages.append(AIMessage(content=text))

    def history(self) -> list[BaseMessage]:
        """The last `max_turns` exchanges, excluding the message just added."""
        window = self.max_turns * 2
        return self.messages[-(window + 1) : -1] if len(self.messages) > 1 else []

    def history_text(self) -> str:
        lines = [
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in self.history()
        ]
        return "\n".join(lines) if lines else "(no earlier turns)"

    def preferences_block(self) -> str:
        if not self.preferences:
            return ""
        rendered = "\n".join(
            f"- {key.replace('_', ' ')}: {value}" for key, value in sorted(self.preferences.items())
        )
        return f"TRAVELLER PREFERENCES CARRIED OVER FROM THE CONVERSATION\n{rendered}\n\n"

    def contextualise(self, question: str) -> str:
        """Expand a short follow-up using the previous user question.

        Retrieval quality collapses on inputs like "what about indoors?", so a
        follow-up is embedded together with the previous question. A short but
        self-contained question ("Convert INR 50,000 to SGD.") is left alone --
        prefixing it with unrelated history pollutes retrieval. The model still
        answers the user's original wording either way.
        """
        if len(question.split()) > 10 or not FOLLOW_UP_MARKERS.search(question):
            return question
        previous = next(
            (m.content for m in reversed(self.history()) if isinstance(m, HumanMessage)), None
        )
        return f"{previous} {question}" if previous else question

    def _extract_preferences(self, text: str) -> None:
        for key, pattern in PREFERENCE_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            value = match.group(1).strip().rstrip(".,;:")
            if key == "trip_length":
                token = value.lower()
                days = WORD_NUMBERS.get(token) or (int(token) if token.isdigit() else None)
                if days is None:
                    continue
                value = f"{days} day{'s' if days != 1 else ''}"
            if key in MULTI_VALUE_KEYS:
                existing = {
                    part.strip().lower()
                    for part in self.preferences.get(key, "").split(",")
                    if part.strip()
                }
                existing.add(value.lower())
                self.preferences[key] = ", ".join(sorted(existing))
            else:
                self.preferences[key] = value

    def reset(self) -> None:
        self.messages.clear()
        self.preferences.clear()
