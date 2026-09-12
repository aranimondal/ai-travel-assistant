# Knowledge base

`documents/` holds the ingested travel content; `index/` holds the generated FAISS index. Both are
produced by `python scripts/build_kb.py` from the top of the repository.

## What is committed here

Twelve documents fetched from Wikivoyage and Wikipedia through the MediaWiki API. Both projects
publish under **CC BY-SA 4.0**, which permits redistribution with attribution, so the extracted
text is committed and the application is usable immediately after `build_kb.py`.

Every file starts with front matter recording the original title, URL, publisher and licence:

```yaml
---
title: Wikivoyage: Singapore travel guide
source_url: https://en.wikivoyage.org/wiki/Singapore
publisher: Wikivoyage
license: CC BY-SA 4.0
topics: [overview, districts, attractions, transport, food, practicalities, itineraries]
fetched_at: 2026-09-12T17:00:00+00:00
characters: 175618
---
```

The retriever reads this metadata, so citations point at the real source rather than at a local
filename. Content is quoted for retrieval only; the originals remain the authoritative version.
The source list lives in `src/travel_assistant/kb/sources.py`.

## Sources not committed, and how to add them

The assignment brief also suggests three Visit Singapore pages (Essential Travel Information,
Sample Itineraries, Things to Do). Their terms of use do not grant permission to redistribute
extracted content, so they are **not** included. The ingestion pipeline does not depend on them —
it reads whatever Markdown files are in `documents/`.

To add them (or any other source) for your own use:

1. Open the page and save the relevant sections as text or Markdown. Respect the site's terms;
   keep it to personal/evaluation use and do not commit the result to a public repository.
2. Save it as `documents/<slug>.md` with front matter in exactly the format above. `title` and
   `source_url` are what appear in citations; `license` documents the terms you are relying on.
3. Re-index without re-downloading the committed sources:

   ```bash
   python scripts/build_kb.py --index-only
   ```

4. Confirm the document is in the index:

   ```bash
   python -c "import json;print(json.load(open('knowledge_base/index/manifest.json'))['sources'])"
   ```

If a source is fetchable from a public API, add a `KbSource` entry to
`src/travel_assistant/kb/sources.py` instead and run `python scripts/build_kb.py --refresh`.

## Ingestion and indexing

| Step | Where | Notes |
|---|---|---|
| Fetch | `kb/fetch.py` | MediaWiki `action=query&prop=extracts`; drops reference/navigation sections; writes front matter. A failed source is logged and skipped, not fatal. |
| Chunk | `kb/loader.py` | Markdown-heading split, then 1100-character chunks with 150 overlap; heading path stored per chunk. |
| Embed | `kb/embeddings.py` | `BAAI/bge-small-en-v1.5` via FastEmbed on CPU; no API key. |
| Index | `kb/store.py` | FAISS, cosine distance, saved to `index/faiss` with `manifest.json`. |

The index is regenerable at any time and is committed so the demos are reproducible without a
build step. If you change `CHUNK_SIZE`, `CHUNK_OVERLAP` or `EMBEDDING_MODEL`, rebuild it —
embeddings from different models are not comparable.

## Attribution

Wikivoyage and Wikipedia content © their contributors, licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Individual page histories are
available at the `source_url` recorded in each document.
