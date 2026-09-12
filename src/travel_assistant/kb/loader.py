"""Load knowledge-base documents and split them into retrievable chunks."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]

HEADING_LINE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+[^\r\n]*", re.MULTILINE)
MIN_CONTENT_CHARS = 40


def parse_document(path: Path) -> tuple[dict[str, str], str]:
    """Split a KB file into its front-matter metadata and Markdown body."""
    raw = path.read_text(encoding="utf-8")
    match = FRONT_MATTER.match(raw)
    if not match:
        return {"title": path.stem, "source_url": ""}, raw

    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip()
    return metadata, raw[match.end() :]


def load_documents(kb_dir: Path) -> list[Document]:
    """Load every Markdown document in `kb_dir` as a LangChain Document."""
    if not kb_dir.exists():
        raise FileNotFoundError(
            f"Knowledge-base directory {kb_dir} not found. Run scripts/build_kb.py first."
        )

    documents: list[Document] = []
    for path in sorted(kb_dir.glob("*.md")):
        metadata, body = parse_document(path)
        documents.append(
            Document(
                page_content=body,
                metadata={
                    "title": metadata.get("title", path.stem),
                    "source_url": metadata.get("source_url", ""),
                    "publisher": metadata.get("publisher", ""),
                    "license": metadata.get("license", ""),
                    "topics": metadata.get("topics", ""),
                    "file": path.name,
                },
            )
        )

    if not documents:
        raise FileNotFoundError(
            f"No Markdown documents found in {kb_dir}. Run scripts/build_kb.py first."
        )
    return documents


def split_documents(
    documents: list[Document], *, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Chunk on Markdown headings first, then on size.

    Heading-aware splitting keeps a chunk about a single topic (one attraction
    list, one transport section), and the recovered heading path is stored on
    the chunk so citations can point at the exact section.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for document in documents:
        sections = header_splitter.split_text(document.page_content)
        for section in sections:
            section_path = " > ".join(
                section.metadata[key] for key in ("h1", "h2", "h3") if section.metadata.get(key)
            )
            for piece in size_splitter.split_text(section.page_content):
                text = piece.strip()
                # Headings are kept for context but must not count as content:
                # a chunk that is only a heading carries no retrievable facts.
                if len(HEADING_LINE.sub("", text).strip()) < MIN_CONTENT_CHARS:
                    continue
                chunks.append(
                    Document(
                        page_content=text,
                        metadata={
                            **document.metadata,
                            "section": section_path or document.metadata.get("title", ""),
                            "chunk_id": f"{document.metadata['file']}#{len(chunks)}",
                        },
                    )
                )
    return chunks
