"""The assistant: routes a question, gathers evidence, generates the answer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from travel_assistant import composer
from travel_assistant.config import Settings, get_settings
from travel_assistant.context import format_kb_context, format_tool_context
from travel_assistant.conversation import Conversation
from travel_assistant.kb.retriever import IndexNotBuiltError, KnowledgeBase, RetrievedChunk
from travel_assistant.llm import LlmSelection, select_llm
from travel_assistant.mcp_client import McpToolbox, ToolCall
from travel_assistant.prompts import ANSWER_PROMPT
from travel_assistant.router import Intent, classify

logger = logging.getLogger(__name__)

WEATHER_TOOL = "get_weather_forecast"
CURRENCY_TOOL = "convert_currency"


@dataclass(slots=True)
class Answer:
    """An answer plus everything needed to audit where it came from."""

    question: str
    text: str
    intent: Intent
    chunks: list[RetrievedChunk] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    generator: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def citations(self) -> list[dict[str, str]]:
        seen: set[tuple[str, str]] = set()
        out: list[dict[str, str]] = []
        for chunk in self.chunks:
            key = (chunk.title, chunk.section)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "title": chunk.title,
                    "section": chunk.section,
                    "url": chunk.source_url,
                    "score": f"{chunk.score:.3f}",
                }
            )
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "route": self.intent.label,
            "generator": self.generator,
            "kb_citations": self.citations,
            "mcp_tool_calls": [
                {
                    "tool": call.tool,
                    "arguments": call.arguments,
                    "ok": call.ok,
                    "error": call.error,
                    "called_at": call.called_at,
                }
                for call in self.tool_calls
            ],
            "notes": self.notes,
        }


class TravelAssistant:
    """Wires intent routing, knowledge-base retrieval, MCP tools and generation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.knowledge_base = KnowledgeBase(self.settings)
        self.toolbox = McpToolbox(self.settings)
        self.llm: LlmSelection = select_llm(self.settings)
        self.conversation = Conversation(max_turns=self.settings.history_turns)

    # --- evidence gathering ---------------------------------------------
    def _retrieve(self, query: str, notes: list[str]) -> list[RetrievedChunk]:
        try:
            return self.knowledge_base.retrieve(query)
        except IndexNotBuiltError as exc:
            notes.append(str(exc))
            return []
        except Exception as exc:  # noqa: BLE001 - retrieval must not break the answer
            logger.exception("retrieval failed")
            notes.append(f"Knowledge-base retrieval failed: {exc}")
            return []

    def _currency_call(self, intent: Intent, notes: list[str]) -> ToolCall | None:
        from travel_assistant.router import currency_codes, parse_amount

        request = intent.currency
        amount = request.amount
        source = request.from_currency
        target = request.to_currency or self.settings.destination_currency

        # A remembered budget lets follow-ups such as "and that budget in SGD?"
        # work: the question names only the target currency, so the amount and
        # the source come from the preference captured earlier in the conversation.
        if amount is None and (remembered := self.conversation.preferences.get("budget")):
            amount = parse_amount(remembered)
            remembered_source = next(iter(currency_codes(remembered)), None)
            if remembered_source and (source is None or source == target):
                source = remembered_source

        if amount is None or not source or source == target:
            notes.append(
                "A currency conversion was requested but the amount or the pair of currencies "
                "could not be determined, so the tool was not called. "
                "Example: 'Convert INR 60,000 to SGD'."
            )
            return None
        return self.toolbox.call(
            CURRENCY_TOOL, {"amount": amount, "from_currency": source, "to_currency": target}
        )

    def _call_tools(self, intent: Intent, notes: list[str]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        if intent.needs_weather:
            calls.append(
                self.toolbox.call(
                    WEATHER_TOOL,
                    {
                        "location": self.settings.destination_name,
                        "days": intent.forecast_days or 3,
                    },
                )
            )
        if intent.needs_currency and (call := self._currency_call(intent, notes)):
            calls.append(call)
        return calls

    # --- generation ------------------------------------------------------
    def _compose_fallback(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        tool_calls: list[ToolCall],
        intent: Intent,
    ) -> str:
        return composer.compose(
            question,
            chunks=chunks,
            tool_calls=tool_calls,
            conversation=self.conversation,
            needs_kb=intent.needs_kb,
        )

    def _generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        tool_calls: list[ToolCall],
        intent: Intent,
        notes: list[str],
    ) -> tuple[str, str]:
        if not self.llm.is_llm:
            return self._compose_fallback(question, chunks, tool_calls, intent), self.llm.label

        messages = ANSWER_PROMPT.format_messages(
            destination=self.settings.destination_name,
            preferences_block=self.conversation.preferences_block(),
            kb_context=format_kb_context(chunks),
            tool_context=format_tool_context(tool_calls),
            question=question,
            history=self.conversation.history(),
        )
        try:
            response = self.llm.chat_model.invoke(messages)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - degrade instead of failing the request
            logger.exception("LLM call failed")
            notes.append(
                f"The language model call failed ({exc}). The answer below was composed "
                "directly from the retrieved evidence instead."
            )
            return (
                self._compose_fallback(question, chunks, tool_calls, intent),
                "grounded template composer (LLM call failed)",
            )

        content = response.content
        return (content if isinstance(content, str) else str(content)).strip(), self.llm.label

    # --- public API ------------------------------------------------------
    def ask(self, question: str) -> Answer:
        """Answer one question and update the conversation state."""
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty.")

        notes: list[str] = []
        self.conversation.add_user(question)

        intent = classify(question, destination_currency=self.settings.destination_currency)
        retrieval_query = self.conversation.contextualise(question)
        if retrieval_query != question:
            notes.append(f"Follow-up expanded for retrieval: “{retrieval_query}”")

        chunks = self._retrieve(retrieval_query, notes) if intent.needs_kb else []
        tool_calls = self._call_tools(intent, notes)

        if intent.needs_kb and not chunks and not any(call.ok for call in tool_calls):
            notes.append("No grounded evidence was available for this question.")

        text, generator = self._generate(question, chunks, tool_calls, intent, notes)
        self.conversation.add_assistant(text)

        return Answer(
            question=question,
            text=text,
            intent=intent,
            chunks=chunks,
            tool_calls=tool_calls,
            generator=generator,
            notes=notes,
        )

    def reset(self) -> None:
        self.conversation.reset()

    def status(self) -> dict[str, Any]:
        """Readiness of each subsystem, for the UI panel and the CLI banner."""
        return {
            "destination": self.settings.destination_name,
            "knowledge_base": {
                "index_available": self.knowledge_base.available,
                "index_dir": str(self.settings.faiss_dir),
                "embedding_model": self.settings.embedding_model,
            },
            "mcp": self.toolbox.status,
            "generator": {"label": self.llm.label, "reason": self.llm.reason},
        }
