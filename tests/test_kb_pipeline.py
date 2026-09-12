"""Knowledge-base ingestion: front matter, Markdown conversion and chunking."""

from __future__ import annotations

from pathlib import Path

from travel_assistant.kb.fetch import _to_markdown
from travel_assistant.kb.loader import load_documents, parse_document, split_documents
from travel_assistant.kb.sources import SINGAPORE_SOURCES

EXTRACT = """Singapore is a city-state.


== See ==
Gardens by the Bay is a horticultural attraction.

=== Museums ===
The National Museum covers local history.


== References ==
[1] Some citation that adds no travel information.

== Get around ==
The MRT is the fastest way across the island.
"""


def test_markdown_conversion_keeps_headings_and_drops_reference_sections() -> None:
    markdown = _to_markdown(EXTRACT)

    assert "## See" in markdown
    assert "### Museums" in markdown
    assert "## Get around" in markdown
    assert "References" not in markdown
    assert "Some citation" not in markdown
    assert "\n\n\n" not in markdown


def test_front_matter_is_parsed_into_metadata(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text(
        "---\n"
        "title: Wikivoyage: Singapore travel guide\n"
        "source_url: https://en.wikivoyage.org/wiki/Singapore\n"
        "license: CC BY-SA 4.0\n"
        "---\n\n"
        "## See\nGardens by the Bay.\n",
        encoding="utf-8",
    )

    metadata, body = parse_document(path)
    assert metadata["title"] == "Wikivoyage: Singapore travel guide"
    assert metadata["source_url"].endswith("/Singapore")
    assert body.lstrip().startswith("## See")


def test_chunks_carry_the_heading_path_and_source_metadata(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text(
        "---\ntitle: Test guide\nsource_url: https://example.org/guide\n---\n\n"
        + _to_markdown(EXTRACT),
        encoding="utf-8",
    )

    documents = load_documents(tmp_path)
    chunks = split_documents(documents, chunk_size=400, chunk_overlap=40)

    assert chunks, "expected at least one chunk"
    assert all(chunk.metadata["title"] == "Test guide" for chunk in chunks)
    assert all(chunk.metadata["source_url"] == "https://example.org/guide" for chunk in chunks)
    sections = {chunk.metadata["section"] for chunk in chunks}
    assert any("See" in section for section in sections)
    assert any("Museums" in section for section in sections)


def test_source_catalogue_covers_the_required_topics() -> None:
    # The brief requires at least three public resources covering attractions,
    # transport, culture and itineraries.
    assert len(SINGAPORE_SOURCES) >= 3
    assert len({source.publisher for source in SINGAPORE_SOURCES}) >= 2

    topics = {topic for source in SINGAPORE_SOURCES for topic in source.topics}
    for required in ("attractions", "transport", "food", "culture", "itineraries", "indoor"):
        assert required in topics, f"no source covers '{required}'"

    for source in SINGAPORE_SOURCES:
        assert source.url.startswith("https://")
        assert source.license, "every source must record its licence for attribution"
