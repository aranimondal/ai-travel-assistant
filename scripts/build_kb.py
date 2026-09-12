"""Build the knowledge base: download the public sources, then embed and index them.

    python scripts/build_kb.py              # download (if needed) + index
    python scripts/build_kb.py --refresh    # re-download every source
    python scripts/build_kb.py --index-only # index the documents already on disk
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from travel_assistant.config import get_settings  # noqa: E402
from travel_assistant.kb.fetch import fetch_all  # noqa: E402
from travel_assistant.kb.sources import SINGAPORE_SOURCES  # noqa: E402
from travel_assistant.kb.store import build_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the travel knowledge base and index.")
    parser.add_argument("--refresh", action="store_true", help="re-download all source documents")
    parser.add_argument(
        "--index-only", action="store_true", help="skip downloading; index existing documents"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()

    existing = list(settings.kb_dir.glob("*.md")) if settings.kb_dir.exists() else []
    if not args.index_only and (args.refresh or not existing):
        print(f"Downloading {len(SINGAPORE_SOURCES)} source document(s) to {settings.kb_dir} ...")
        written = fetch_all(SINGAPORE_SOURCES, settings.kb_dir)
        print(f"Downloaded {len(written)} document(s).")
    else:
        print(f"Using {len(existing)} document(s) already present in {settings.kb_dir}.")

    print(f"Embedding with {settings.embedding_model} and writing the FAISS index ...")
    manifest = build_index(settings)
    print(
        f"Indexed {manifest['chunks']} chunks from {manifest['documents']} documents "
        f"into {settings.faiss_dir}"
    )
    print("Sources in the index:")
    for source in manifest["sources"]:
        print(f"  - {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
