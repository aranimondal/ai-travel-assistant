"""Knowledge-base retriever used by the assistant."""

from __future__ import annotations

import logging
import re

from travel_assistant.config import Settings
from travel_assistant.kb.store import IndexNotBuiltError, RetrievedChunk, load_index, search

logger = logging.getLogger(__name__)

# Conversational filler dilutes the query embedding: "Suggest activities for us"
# retrieves markedly worse than "activities Singapore" against the same index.
# Stripping it before embedding roughly doubles the top similarity on chat-style
# questions while leaving genuinely off-topic questions below the score floor.
FILLER = re.compile(
    r"\b(what|which|who|how|when|where|why|can|could|should|would|will|do|does|did|is|are|am|"
    r"was|were|i|we|you|my|our|us|me|please|suggest|recommend|tell|give|show|there|any|some|"
    r"a|an|the|to|for|of|in|on|at|and|or|if|it|that|this|these|those|have|has|had|be|been|"
    r"like|want|need|about|with|from|plan|create|make)\b",
    re.I,
)


class KnowledgeBase:
    """Thin wrapper around the FAISS index with lazy loading."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store = None

    @property
    def available(self) -> bool:
        return (self._settings.faiss_dir / "index.faiss").exists()

    def build_query(self, question: str) -> str:
        """Reduce a chat-style question to its retrieval keywords."""
        core = FILLER.sub(" ", question)
        core = re.sub(r"[^\w\s-]", " ", core)
        core = re.sub(r"\s+", " ", core).strip()
        if not core:
            return question
        destination = self._settings.destination_name
        if destination.lower() not in core.lower():
            core = f"{core} {destination}"
        return core

    def retrieve(self, question: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return relevant chunks, or an empty list when nothing clears the score floor."""
        if self._store is None:
            self._store = load_index(self._settings)  # raises IndexNotBuiltError

        chunks = search(
            self._store,
            self.build_query(question),
            top_k=top_k or self._settings.retrieval_top_k,
            min_score=self._settings.retrieval_min_score,
        )
        logger.debug("retrieved %d chunk(s) for %r", len(chunks), question)
        return chunks


__all__ = ["IndexNotBuiltError", "KnowledgeBase", "RetrievedChunk"]
