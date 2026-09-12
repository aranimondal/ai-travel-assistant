"""Fetch knowledge-base documents from the public MediaWiki APIs.

Each page is written to ``knowledge_base/documents`` as Markdown with a YAML
front-matter block that keeps the source title, URL and licence, so retrieval
can cite the original resource.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx

from travel_assistant.kb.sources import KbSource

logger = logging.getLogger(__name__)

USER_AGENT = (
    "ai-travel-assistant/1.0 (https://github.com/aranimondal/ai-travel-assistant) httpx"
)

# Reference/navigation sections that add noise but no travel information.
SKIP_SECTIONS = {
    "see also",
    "references",
    "further reading",
    "external links",
    "notes",
    "bibliography",
    "citations",
    "gallery",
    "sources",
}


def _plaintext_extract(source: KbSource, *, timeout: float = 30.0) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "format": "json",
        "redirects": "1",
        "titles": source.page,
        "formatversion": "2",
    }
    response = httpx.get(
        source.api_url,
        params=params,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", [])
    if not pages or "extract" not in pages[0]:
        raise RuntimeError(f"No extract returned for {source.page} on {source.wiki}")
    return pages[0]["extract"]


def _to_markdown(extract: str) -> str:
    """Convert MediaWiki plaintext extracts to Markdown, dropping noise sections.

    Extracts mark sections as ``== Title ==`` / ``=== Title ===``; converting
    them to ATX headings lets the splitter chunk along real topic boundaries.
    """
    lines: list[str] = []
    skipping = False

    for raw_line in extract.splitlines():
        line = raw_line.rstrip()
        heading = re.match(r"^(={2,6})\s*(.+?)\s*\1$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            skipping = title.lower() in SKIP_SECTIONS
            if not skipping:
                lines.append("")
                lines.append(f"{'#' * min(level, 6)} {title}")
                lines.append("")
            continue
        if skipping or not line.strip():
            if not skipping and lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(line.strip())

    markdown = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"


def _front_matter(source: KbSource, *, characters: int) -> str:
    fetched = datetime.now(UTC).isoformat(timespec="seconds")
    topics = ", ".join(source.topics)
    return (
        "---\n"
        f"title: {source.title}\n"
        f"source_url: {source.url}\n"
        f"publisher: {source.publisher}\n"
        f"license: {source.license}\n"
        f"topics: [{topics}]\n"
        f"fetched_at: {fetched}\n"
        f"characters: {characters}\n"
        "---\n\n"
    )


def fetch_source(source: KbSource, target_dir: Path) -> Path:
    """Download one source and write it as a Markdown document."""
    body = _to_markdown(_plaintext_extract(source))
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{source.slug}.md"
    path.write_text(_front_matter(source, characters=len(body)) + body, encoding="utf-8")
    logger.info("wrote %s (%d characters)", path.name, len(body))
    return path


def fetch_all(sources: tuple[KbSource, ...], target_dir: Path) -> list[Path]:
    """Download every source; a single failure does not abort the run."""
    written: list[Path] = []
    failures: list[str] = []
    for source in sources:
        try:
            written.append(fetch_source(source, target_dir))
        except (httpx.HTTPError, RuntimeError) as exc:
            failures.append(f"{source.slug}: {exc}")
            logger.error("failed to fetch %s: %s", source.slug, exc)

    if failures and not written:
        raise RuntimeError("All knowledge-base downloads failed:\n" + "\n".join(failures))
    if failures:
        logger.warning("%d source(s) failed and were skipped", len(failures))
    return written
