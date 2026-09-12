"""FAISS-backed vector store: build, persist, load and search."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# The FAISS integration still lives in langchain-community, which emits a sunset
# warning on import. There is no standalone langchain-faiss package yet, so the
# warning is suppressed here rather than leaking into every CLI run.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*langchain-community.*")
    from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document

from travel_assistant.config import Settings
from travel_assistant.kb.embeddings import FastEmbedEmbeddings
from travel_assistant.kb.loader import load_documents, split_documents

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A knowledge-base chunk with its similarity score and citation metadata."""

    text: str
    title: str
    source_url: str
    section: str
    score: float

    @property
    def citation(self) -> str:
        label = f"{self.title} — {self.section}" if self.section else self.title
        return f"{label} ({self.source_url})" if self.source_url else label


class IndexNotBuiltError(RuntimeError):
    """Raised when the FAISS index is missing."""


def build_index(settings: Settings) -> dict[str, object]:
    """Embed every KB chunk and persist a FAISS index plus a manifest."""
    documents = load_documents(settings.kb_dir)
    chunks = split_documents(
        documents, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    logger.info("embedding %d chunks from %d documents", len(chunks), len(documents))

    embeddings = FastEmbedEmbeddings(settings.embedding_model)
    # Cosine distance keeps scores comparable across chunk lengths, which the
    # "not enough information" threshold depends on.
    store = FAISS.from_documents(chunks, embeddings, distance_strategy="COSINE")

    settings.faiss_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(settings.faiss_dir))

    manifest = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "documents": len(documents),
        "chunks": len(chunks),
        "sources": sorted(
            {
                f"{doc.metadata.get('title', '')} <{doc.metadata.get('source_url', '')}>"
                for doc in documents
            }
        ),
    }
    (settings.index_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def load_index(settings: Settings) -> FAISS:
    if not (settings.faiss_dir / "index.faiss").exists():
        raise IndexNotBuiltError(
            f"No FAISS index at {settings.faiss_dir}. Run: python scripts/build_kb.py"
        )
    embeddings = FastEmbedEmbeddings(settings.embedding_model)
    # The index is generated locally by this project, so deserialisation is safe.
    return FAISS.load_local(
        str(settings.faiss_dir), embeddings, allow_dangerous_deserialization=True
    )


def read_manifest(index_dir: Path) -> dict[str, object] | None:
    path = index_dir / MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _to_similarity(raw_score: float) -> float:
    """FAISS returns cosine *distance*; convert to a 0-1 similarity."""
    return max(0.0, min(1.0, 1.0 - float(raw_score)))


def search(
    store: FAISS, query: str, *, top_k: int, min_score: float
) -> list[RetrievedChunk]:
    """Return the chunks above `min_score`, best first."""
    hits: list[tuple[Document, float]] = store.similarity_search_with_score(query, k=top_k)
    results = [
        RetrievedChunk(
            text=document.page_content,
            title=str(document.metadata.get("title", "")),
            source_url=str(document.metadata.get("source_url", "")),
            section=str(document.metadata.get("section", "")),
            score=_to_similarity(raw_score),
        )
        for document, raw_score in hits
    ]
    return [chunk for chunk in results if chunk.score >= min_score]
