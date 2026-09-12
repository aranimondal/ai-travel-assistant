# AI Travel Planning Assistant — Singapore

A context-aware travel assistant that answers destination questions from an indexed document
knowledge base (RAG) and answers time-sensitive questions by calling tools over the **Model
Context Protocol** (MCP). Its main purpose is the combination: *"Create a three-day Singapore
itinerary for next week and adjust it according to the weather forecast"* pulls attractions and
itinerary guidance from the knowledge base, calls the MCP weather tool, and returns a day-wise,
weather-aware plan in which knowledge-base facts, live tool data and the assistant's own
recommendations are clearly separated.

Scope is deliberately narrow, per the brief: destination knowledge, weather and currency for one
destination. No flights, hotels, payments, bookings or navigation.

---

## 1. Architecture

```
                      ┌──────────────────────────────────────────────┐
  Streamlit UI ─────► │ TravelAssistant  (assistant.py)              │
  CLI (scripts/chat)  │  1. classify intent      (router.py)         │
                      │  2. retrieve KB chunks   (kb/retriever.py)   │
                      │  3. call MCP tools       (mcp_client.py)     │
                      │  4. build prompt context (context.py)        │
                      │  5. generate answer      (llm.py/composer.py)│
                      └───────┬───────────────────────┬──────────────┘
                              │                       │
              ┌───────────────▼──────────┐   ┌────────▼─────────────────────────┐
              │ FAISS vector store       │   │ MCP server (stdio subprocess)    │
              │ 12 documents / ~700      │   │ travel_mcp.server (FastMCP)      │
              │ chunks, BGE embeddings   │   │  • get_weather_forecast          │
              │ knowledge_base/index     │   │  • convert_currency              │
              └──────────────────────────┘   └────────┬─────────────────────────┘
                                                      │
                                     Open-Meteo API ──┴── Frankfurter API (ECB)
```

### Components

| Path | Responsibility |
|---|---|
| `src/travel_assistant/assistant.py` | Orchestrator. Routes the question, gathers evidence, generates the answer, returns an auditable `Answer` (text + citations + tool calls + notes). |
| `src/travel_assistant/router.py` | Rule-based intent classification and tool-argument extraction (amounts, currency codes, forecast days). |
| `src/travel_assistant/kb/` | Ingestion (`fetch.py`, `sources.py`), chunking (`loader.py`), embeddings (`embeddings.py`), FAISS index (`store.py`), retrieval (`retriever.py`). |
| `src/travel_assistant/mcp_client.py` | MCP client bridge: starts the MCP server, loads its tools via `langchain-mcp-adapters`, invokes them, records every call and failure. |
| `src/travel_assistant/prompts.py` | System and user prompt templates encoding the grounding rules. |
| `src/travel_assistant/context.py` | Renders retrieved chunks and tool results into labelled prompt context. |
| `src/travel_assistant/composer.py` | Deterministic grounded composer used when no LLM credential is configured. |
| `src/travel_assistant/conversation.py` | Rolling history plus sticky traveller preferences and follow-up expansion. |
| `src/travel_mcp/` | The MCP server: `server.py` (FastMCP tools), `weather.py` (Open-Meteo), `currency.py` (Frankfurter/ECB). |
| `app/streamlit_app.py` | Chat UI with a provenance panel and a subsystem status sidebar. |
| `scripts/` | `build_kb.py` (ingest + index), `chat.py` (CLI), `demo.py` (the four required scenarios). |

### Data flow for one question

1. `router.classify()` decides whether the question needs the knowledge base, the weather tool,
   the currency tool, or a combination, and extracts tool arguments.
2. If the knowledge base is needed, the question is normalised into a retrieval query and
   embedded; FAISS returns the top-k chunks above a cosine-similarity floor.
3. If a tool is needed, `McpToolbox` opens an MCP stdio session and invokes the tool. Failures
   become recorded `ToolCall` objects, never guesses.
4. Retrieved chunks and tool results are rendered as separately labelled context blocks.
5. The configured LLM generates the answer under the grounding prompt. With no credentials, the
   template composer quotes the evidence directly instead, and says so.

---

## 2. Knowledge base

### Sources

Twelve documents from two publishers, all **CC BY-SA 4.0**, which is why the extracted text can
be redistributed in this repository with attribution. Titles and URLs are stored as front matter
so answers can cite the original resource.

| Source | Coverage |
|---|---|
| Wikivoyage: Singapore travel guide | Overview, districts, transport, food, practicalities, itineraries, climate |
| Wikivoyage: Singapore/Riverside | Museums, historical buildings, nightlife |
| Wikivoyage: Singapore/Marina Bay | Gardens by the Bay, indoor attractions, events |
| Wikivoyage: Singapore/Chinatown | Temples, heritage, food |
| Wikivoyage: Singapore/Little India | Temples, shopping, food, budget stays |
| Wikivoyage: Singapore/Bugis and Kampong Glam | Heritage, culture, food |
| Wikivoyage: Singapore/Orchard | Shopping, indoor options, dining |
| Wikivoyage: Singapore/Sentosa and Harbourfront | Family attractions, beaches, theme parks |
| Wikivoyage: Singapore/East Coast | Outdoor activities, hawker food, beaches |
| Wikipedia: Tourism in Singapore | Popular destinations, visitor practicalities |
| Wikipedia: Mass Rapid Transit (Singapore) | MRT network, fares, operating details |
| Wikipedia: Singaporean cuisine | Hawker culture, dishes |

The brief also suggests the Visit Singapore pages. They are **not** ingested here: their terms of
use do not permit redistributing extracted content, and this repository commits the knowledge
base. To add them locally, see [`knowledge_base/README.md`](knowledge_base/README.md) — the
ingestion pipeline picks up any Markdown file with the same front matter, no code change needed.

### Acquisition

`scripts/build_kb.py` fetches each page through the MediaWiki **`action=query&prop=extracts`**
API (a documented, rate-limit-friendly endpoint; requests send a contact User-Agent as Wikimedia
policy requires), converts the extract to Markdown, drops reference/navigation sections, and
writes `knowledge_base/documents/<slug>.md` with front matter:

```yaml
---
title: Wikivoyage: Singapore/Marina Bay
source_url: https://en.wikivoyage.org/wiki/Singapore/Marina_Bay
publisher: Wikivoyage
license: CC BY-SA 4.0
topics: [neighbourhood, attractions, indoor, gardens, views]
fetched_at: 2026-09-12T17:00:00+00:00
characters: 20956
---
```

---

## 3. RAG workflow

| Stage | Implementation | Detail |
|---|---|---|
| **Load** | `kb/fetch.py`, `kb/loader.py` | MediaWiki extracts → Markdown + front matter → `Document` objects carrying title/URL/licence. |
| **Chunk** | `kb/loader.py` | `MarkdownHeaderTextSplitter` on `#`/`##`/`###` first, so a chunk stays about one topic, then `RecursiveCharacterTextSplitter` (1100 chars, 150 overlap). The heading path (`See > Itineraries`) is stored on each chunk and used in citations. Heading-only fragments are dropped. |
| **Embed** | `kb/embeddings.py` | `BAAI/bge-small-en-v1.5` through **FastEmbed** (quantised ONNX, CPU, no API key). Documents are embedded as-is; queries get the BGE retrieval instruction prefix. |
| **Index** | `kb/store.py` | **FAISS** with cosine distance, persisted to `knowledge_base/index/faiss` plus a `manifest.json` recording model, chunk settings and source list. |
| **Retrieve** | `kb/retriever.py` | Query normalisation strips conversational filler and pins the destination, then top-k (default 6) similarity search with a **0.32 similarity floor**. Below the floor nothing is returned, which is what makes "not enough information" possible. |
| **Context** | `context.py` | Chunks are rendered as numbered passages with title, section and URL, inside a character budget. |
| **Generate** | `prompts.py` + `llm.py` / `composer.py` | The prompt forbids destination facts from outside the context block and requires a `Sources` block. |

The current index: **692 chunks from 12 documents** (see `knowledge_base/index/manifest.json`).

Why query normalisation exists: measured against this index, `"Suggest activities for us"` scored
0.21 at best, while `"activities Singapore"` scored 0.58 for the same intent. Stripping filler
roughly doubles top similarity on chat-style questions, while genuinely off-topic questions
("How do I get a mortgage in Iceland?", top score 0.098) still fall below the floor.

---

## 4. MCP tools

The assistant never calls the weather or currency HTTP APIs directly. `travel_mcp.server` is a
**FastMCP** server started as a stdio subprocess; `langchain-mcp-adapters` turns its advertised
tools into LangChain tools with schemas derived from the MCP tool definitions.

| Tool | Arguments | Returns | Upstream |
|---|---|---|---|
| `get_weather_forecast` | `location: str = "Singapore"`, `days: int = 1..7` | Current conditions plus per-day min/max temperature, rain probability, `rain_expected`, and an `outdoor_suitability` hint (`outdoor` / `mixed` / `indoor`) | Open-Meteo (keyless) |
| `convert_currency` | `amount: float`, `from_currency: str`, `to_currency: str` | Converted amount, rate, rate publication date, source attribution | Frankfurter, ECB reference rates (keyless) |

`outdoor_suitability` is computed in the tool, not the prompt, so "replace outdoor plans with
indoor options if rain is expected" is driven by data rather than by the model's guesswork.

### Invocation flow

```
question ─► router.classify() ─► needs_weather / needs_currency + arguments
        └─► McpToolbox.call(name, args)
              ├─ MultiServerMCPClient starts `python -m travel_mcp.server` (stdio)
              ├─ list tools (cached) ─► ainvoke(tool, args) with a 20 s timeout
              └─ result ─► ToolCall(ok=True, result=…) | ToolCall(ok=False, error=…)
```

Every outcome is recorded on the `Answer`, and the UI/CLI shows the tool name, the exact
arguments and the raw JSON result. Four failure modes are handled explicitly and none of them
produces a fabricated number: server won't start, tool not offered, tool timeout, and an upstream
error (for example an unsupported currency pair).

The server is standalone and reusable — point any MCP client at it:

```json
{ "mcpServers": { "travel-live-data": {
    "command": "C:/NAGP/ai-travel-assistant/.venv/Scripts/python.exe",
    "args": ["-m", "travel_mcp.server"],
    "env": { "PYTHONPATH": "C:/NAGP/ai-travel-assistant/src" } } } }
```

Tool selection is rule-based (`router.py`) rather than model-driven. It is deterministic,
testable and instant, and the signals involved (money amounts, weather words, itinerary words)
are unambiguous. As the brief requires, MCP is not used for destination questions the knowledge
base already covers: `classify()` sets `needs_weather`/`needs_currency` only on time-sensitive or
monetary intent.

---

## 5. Prompt strategy

`prompts.py` holds one system prompt with three groups of rules:

1. **Evidence rules** — destination facts *only* from the knowledge-base context block; current
   information *only* from the MCP tool-results block; if the knowledge base is insufficient, say
   so and name what is missing; if a tool failed, say the live data is unavailable and never
   estimate it. Planning judgement is allowed but must be labelled as a recommendation.
2. **Answer format** — direct answer first, day-wise headings for itineraries, live values marked
   inline as `[Live: <tool>]`, knowledge-base claims referenced by source title, a closing
   `Sources` block, and assistant reasoning kept in its own section.
3. **Conversation rules** — honour preferences carried over from earlier turns; resolve
   references such as "there", "that day", "the same budget" against the history.

The user message is assembled by `context.py` into three clearly separated blocks —
`TRAVELLER PREFERENCES`, `KNOWLEDGE BASE CONTEXT`, `LIVE TOOL RESULTS` — followed by the
question. Keeping the blocks separate (rather than concatenating everything) is what lets the
model attribute each statement correctly.

### Context and conversation strategy

- **Rolling window**: the last 6 exchanges are passed as chat messages (`history_turns`).
- **Sticky preferences**: travelling companions, trip length, budget, interests, constraints and
  timing are extracted with patterns and kept *outside* the window, so they still apply after the
  window has scrolled past. They are shown in the sidebar and restated in recommendations.
- **Follow-up expansion**: a short question containing a back-reference ("What about indoors?") is
  concatenated with the previous question *for retrieval only*. A short but self-contained
  question ("Convert INR 50,000 to SGD.") is deliberately left alone, since prefixing unrelated
  history pollutes retrieval.
- **Preference reuse for tools**: "And how much is that budget in SGD?" resolves the amount and
  source currency from the remembered budget, then calls `convert_currency`.

---

## 6. LLM configuration

Any LangChain chat model can be used. `LLM_PROVIDER=auto` (default) picks the first provider that
has credentials:

| Provider | Variables | Extra package |
|---|---|---|
| OpenAI / compatible | `OPENAI_API_KEY`, optional `OPENAI_BASE_URL` | `pip install langchain-openai` |
| Google Gemini | `GOOGLE_API_KEY` | `pip install langchain-google-genai` |
| Anthropic | `ANTHROPIC_API_KEY` | `pip install langchain-anthropic` |
| Ollama (local) | `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL` | `pip install langchain-ollama` |

**With no credentials configured**, the assistant still runs end to end: `composer.py` builds the
answer by quoting the retrieved passages and the MCP tool values directly, and labels itself
`grounded template composer (no LLM credentials configured)` in the UI, the CLI and every
generated sample. Retrieval, MCP invocation, provenance, conversation memory and refusal
behaviour are all fully exercised in this mode — only the prose fluency differs. Output is never
presented as model-generated when it is not. If a configured LLM call fails at runtime, the
assistant falls back to the same composer and records the reason in the answer notes.

---

## 7. Configuration

All settings are environment variables (see `.env.example`); defaults work out of the box.

| Variable | Default | Purpose |
|---|---|---|
| `DESTINATION_NAME` | `Singapore` | Destination used for tool arguments and query pinning |
| `DESTINATION_CURRENCY` | `SGD` | Default target currency for conversions |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1100` / `150` | Chunking |
| `RETRIEVAL_TOP_K` | `6` | Chunks retrieved per question |
| `RETRIEVAL_MIN_SCORE` | `0.32` | Similarity floor; below it the KB reports "not enough information" |
| `LLM_PROVIDER` | `auto` | `auto`, `openai`, `google`, `anthropic`, `ollama`, or `template` |
| `LLM_MODEL` / `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` | provider default / `0.2` / `1200` | Generation |
| `MCP_TOOL_TIMEOUT_SECONDS` | `20` | Per-tool-call timeout |
| `HISTORY_TURNS` | `6` | Conversation window |

---

## 8. Setup

**Prerequisites:** Python 3.11+ (developed and validated on 3.13.2), outbound HTTPS to
`en.wikivoyage.org`, `en.wikipedia.org`, `huggingface.co` (one-time model download),
`api.open-meteo.com` and `api.frankfurter.dev`. No API key, GPU, Docker or database required.

```bash
git clone https://github.com/aranimondal/ai-travel-assistant.git
cd ai-travel-assistant

python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS / Linux

pip install -r requirements.txt
pip install -r requirements-dev.txt   # only for tests / linting

copy .env.example .env         # optional; defaults work without it
```

### Build the knowledge base (required once, ~1–2 minutes)

```bash
python scripts/build_kb.py
```

Downloads the 12 source documents, embeds the chunks and writes the FAISS index. Expected tail:

```
Indexed 692 chunks from 12 documents into knowledge_base/index/faiss
```

`--refresh` re-downloads the sources; `--index-only` re-indexes what is already on disk (use it
after changing chunking settings or adding your own documents).

### Run

```bash
streamlit run app/streamlit_app.py        # UI at http://localhost:8501
python scripts/chat.py                    # interactive CLI
python scripts/chat.py --ask "What indoor attractions can I visit?"
python scripts/chat.py --ask "Convert INR 50,000 to SGD." --json
```

Both entry points print a status banner covering the index, the MCP tool list and the active
generator, so a misconfiguration is visible immediately.

### Test

```bash
pytest -m "not network and not index"     # 67 unit tests, offline, ~3 s
pytest -m "index and network"             # 9 end-to-end tests: real FAISS index + real MCP subprocess
pytest                                    # everything
ruff check src tests scripts app          # lint
```

---

## 9. Demonstrations

`scripts/demo.py` runs all four required scenarios plus a refusal case against the live
application and prints real output. Committed transcript:
[`docs/SAMPLE_QA.md`](docs/SAMPLE_QA.md).

```bash
python scripts/demo.py                            # print
python scripts/demo.py --write docs/SAMPLE_QA.md  # regenerate the transcript
```

Because weather and exchange rates change, a fresh run will show different live values — that is
the point of the MCP path. Everything else is reproducible.

### A. RAG

```bash
python scripts/chat.py --ask "What are the must-visit attractions in Singapore?"
```

Expect `[route: knowledge base]`, no MCP calls, and `KB` citation lines with source titles,
sections, similarity scores and URLs.

### B. MCP

```bash
python scripts/chat.py --ask "What is the weather forecast in Singapore for the next three days?"
python scripts/chat.py --ask "Convert INR 50,000 to SGD."
```

Expect `[route: weather tool]` / `[route: currency tool]`, a
`MCP get_weather_forecast {'location': 'Singapore', 'days': 3} -> ok` line, and values labelled
`[Live: MCP tool ...]` with the upstream source named.

To see failure handling without breaking anything, ask for a currency the rate service does not
publish — the tool reports the error and the answer states the data is unavailable instead of
inventing a rate:

```bash
python scripts/chat.py --ask "Convert 100 ZWL to SGD."
```

### C. RAG + MCP combined

```bash
python scripts/chat.py --ask "Create a three-day Singapore itinerary for next week and adjust it according to the weather forecast."
python scripts/chat.py --ask "I have a budget of INR 60,000. Convert it to SGD and suggest a three-day itinerary."
```

Expect `[route: knowledge base + weather tool]`, both KB citations *and* an MCP tool line, and a
day-wise plan whose indoor/outdoor guidance follows the forecast's `outdoor_suitability`.

### D. Conversational context

In the UI, or via `python scripts/chat.py`:

```
> I am travelling with my kids for three days and my budget is INR 80,000.
> Suggest activities for us.
> What indoor options do we have if it rains?
> And how much is that budget in SGD?
```

The second turn has no destination, the third no companions, the fourth no amount or source
currency. The sidebar shows the retained preferences, and the last turn calls
`convert_currency {'amount': 80000.0, 'from_currency': 'INR', 'to_currency': 'SGD'}` — proof the
budget survived three turns.

---

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No FAISS index at ...` | Run `python scripts/build_kb.py`. |
| Sidebar shows *MCP tools unavailable* | The server subprocess failed to start. Run `python -m travel_mcp.server` with `PYTHONPATH=src` and read the error; check the venv interpreter is the one running Streamlit. |
| First run stalls for ~30 s | One-time FastEmbed model download from `huggingface.co`; cached afterwards. |
| `403 Forbidden` during `build_kb.py` | Wikimedia requires a descriptive User-Agent with contact info; keep the one in `kb/fetch.py`, and do not run parallel refreshes. |
| Tool call reports *timed out* | Raise `MCP_TOOL_TIMEOUT_SECONDS`, or check egress to `api.open-meteo.com` / `api.frankfurter.dev`. |
| "Not enough information" for a topic the KB should cover | Lower `RETRIEVAL_MIN_SCORE` (e.g. `0.28`) or raise `RETRIEVAL_TOP_K`, then re-ask. Rebuild the index after changing `CHUNK_SIZE`. |
| Answers labelled *grounded template composer* | No LLM credential was found. Expected; set a provider key to enable model prose (§6). |
| `ResolutionImpossible` on install | Use the ranges in `requirements.txt`; over-pinning `langchain-core` conflicts with `langchain-community`. |

---

## 11. Security notes

- **No secrets in the repository.** Credentials are read from the environment only; `.env` is
  git-ignored and `.env.example` contains empty placeholders. The committed transcript in
  `docs/SAMPLE_QA.md` contains no keys, tokens or personal data.
- **Keyless by default.** Both upstream services are public and unauthenticated, so the default
  configuration has no credential to leak.
- **Outbound only, allow-listed hosts.** The application opens no listening port other than the
  Streamlit dev server on localhost, and contacts only the hosts listed in §8.
- **Least privilege.** The MCP server exposes exactly two read-only tools with typed arguments.
  It does not read the filesystem, execute commands or write anywhere.
- **Untrusted input handling.** Currency codes are validated against the ISO-4217 shape before
  use, amounts must be non-negative, and `days` is clamped to 1–7. Tool arguments are passed as
  structured values, never interpolated into shell commands or hand-built URLs.
- **Safe deserialisation.** `FAISS.load_local` needs `allow_dangerous_deserialization=True`
  because the index is a pickle. Only load an index this project generated locally; never point
  `INDEX_DIR` at a downloaded artefact.
- **Content licensing.** Ingested content is CC BY-SA 4.0 and retains title, URL and licence in
  front matter. Sources whose terms forbid redistribution are documented but not committed.
- If you expose the UI beyond localhost, put it behind an authenticating reverse proxy — the
  Streamlit dev server has no authentication.

---

## 12. Repository layout

```
ai-travel-assistant/
├─ app/streamlit_app.py            # chat UI with provenance panel
├─ docs/SAMPLE_QA.md               # generated transcript of the demo scenarios
├─ knowledge_base/
│  ├─ README.md                    # sources, licences, how to add your own documents
│  ├─ documents/*.md               # 12 ingested documents with front matter
│  └─ index/                       # FAISS index + manifest.json (generated)
├─ scripts/{build_kb,chat,demo}.py
├─ src/travel_assistant/           # orchestration, RAG, MCP client, prompts
├─ src/travel_mcp/                 # MCP server and its upstream clients
├─ tests/                          # 67 unit tests + 9 end-to-end scenario tests
├─ requirements.txt / requirements-dev.txt
└─ .env.example
```
