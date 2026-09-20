# AI Travel Planning Assistant — Singapore

A context-aware travel assistant that answers destination questions from an indexed document knowledge base (RAG) and answers time-sensitive questions by calling tools over the Model Context Protocol (MCP). Its main purpose is the combination: "Create a three-day Singapore itinerary for next week and adjust it according to the weather forecast" pulls attractions and itinerary guidance from the knowledge base, calls the MCP weather tool, and returns a day-wise, weather-aware plan in which knowledge-base facts, live tool data and the assistant's own recommendations are clearly separated.

Scope is deliberately narrow: destination knowledge, weather and currency for one destination. No flights, hotels, payments, bookings or navigation.

## 1. Architecture

```text
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
| `src/travel_assistant/router.py` | Rule-based intent classification and tool-argument extraction. |
| `src/travel_assistant/kb/` | Ingestion, chunking, embeddings, FAISS index and retrieval. |
| `src/travel_assistant/mcp_client.py` | MCP client bridge: starts the local custom MCP server, loads its tools via `langchain-mcp-adapters`, invokes them, records every call and failure. |
| `src/travel_assistant/prompts.py` | System and user prompt templates encoding the grounding and source/provenance rules. |
| `src/travel_assistant/context.py` | Renders retrieved chunks and tool results into labelled prompt context. |
| `src/travel_assistant/composer.py` | Deterministic grounded composer used when no LLM credential is configured. |
| `src/travel_assistant/conversation.py` | Rolling history plus sticky traveller preferences and follow-up expansion. |
| `src/travel_assistant/llm.py` | Provider-agnostic LangChain LLM integration, including local Ollama support. |
| `src/travel_mcp/` | Custom MCP server implemented for this assignment: FastMCP weather and currency tools. No ready-made travel MCP server is required. |
| `app/streamlit_app.py` | Chat UI with provenance and subsystem status panels. |
| `scripts/` | KB build, CLI chat and repeatable demo scenarios. |
| `knowledge_base/` | Local Singapore travel source documents and the persisted FAISS index. |

### Important implementation links

| Link | Purpose |
|---|---|
| [`assistant.py`](src/travel_assistant/assistant.py) | Main orchestration: routes the request, retrieves RAG evidence, calls MCP tools, builds context and invokes the LLM. |
| [`router.py`](src/travel_assistant/router.py) | Determines the request intent and whether RAG, weather MCP, currency MCP or a combination is required. |
| [`mcp_client.py`](src/travel_assistant/mcp_client.py) | MCP client bridge that starts the custom server over stdio, invokes tools and records tool failures/results. |
| [`server.py`](src/travel_mcp/server.py) | Custom FastMCP server exposing `get_weather_forecast` and `convert_currency`. |
| [`kb/retriever.py`](src/travel_assistant/kb/retriever.py) | Performs semantic retrieval of relevant knowledge-base chunks. |
| [`kb/store.py`](src/travel_assistant/kb/store.py) | Builds, persists, loads and searches the FAISS vector index. |
| [`kb/embeddings.py`](src/travel_assistant/kb/embeddings.py) | Embedding implementation using FastEmbed/BGE. |
| [`prompts.py`](src/travel_assistant/prompts.py) | Grounding and provenance rules used by the LLM prompt. |
| [`context.py`](src/travel_assistant/context.py) | Separates traveller preferences, KB evidence and live MCP results before generation. |
| [`llm.py`](src/travel_assistant/llm.py) | Provider-agnostic LangChain chat-model selection, including Ollama for local LLM execution. |
| [`conversation.py`](src/travel_assistant/conversation.py) | Maintains rolling conversation history, preferences and contextual follow-ups. |
| [`build_kb.py`](scripts/build_kb.py) | Fetches/processes supported sources and rebuilds the local knowledge-base/index. |
| [`streamlit_app.py`](app/streamlit_app.py) | Streamlit UI and visible provenance/status panels for evaluator demonstrations. |
| [`knowledge_base/`](knowledge_base/) | Local Singapore travel documents, metadata and persisted retrieval assets. |

### Data flow for one question

1. `router.classify()` decides whether the question needs the knowledge base, weather, currency, or a combination, and extracts tool arguments.
2. If the knowledge base is needed, the question is normalized into a retrieval query and FAISS returns the top-k chunks above a similarity floor.
3. If a tool is needed, `McpToolbox` opens an MCP stdio session with the local custom MCP server and invokes it. Failures become recorded `ToolCall` objects; the system never guesses live values.
4. Retrieved chunks and tool results are rendered as separately labelled context blocks.
5. The configured LLM generates the answer under the grounding prompt. With no credentials, the template composer uses the evidence directly and says so.

## 2. Knowledge base

The repository contains a Singapore-focused knowledge base built from public sources, with source title and URL metadata retained for citations. The ingestion pipeline can fetch and index the supported sources locally.

The current design uses meaningful chunks, BGE embeddings through FastEmbed, and FAISS with cosine similarity. Retrieval normalizes conversational filler and applies a similarity floor; below that floor, the KB returns no grounded evidence rather than forcing a weak match.

## 3. MCP tools

The assistant never calls the weather or currency HTTP APIs directly. `travel_mcp.server` is a custom FastMCP server implemented in this repository and started as a stdio subprocess; `langchain-mcp-adapters` exposes its tools to the application.

| Tool | Purpose | Upstream |
|---|---|---|
| `get_weather_forecast` | Current conditions and multi-day forecast, including rain probability and outdoor suitability | Open-Meteo |
| `convert_currency` | Currency conversion with rate/date/source information | Frankfurter / ECB reference rates |

### MCP boundary

- **RAG** is the source for stable destination knowledge.
- **MCP** is the source for current weather and currency values.
- MCP is **not** used for ordinary attraction, neighbourhood, transport, food, culture, or itinerary facts already covered by the KB.
- Tool failures are surfaced explicitly; the assistant does not replace unavailable live data with guesses.

This separation is enforced by deterministic routing in `router.py` and again by the system prompt, so the LLM cannot reinterpret evidence provenance.

### No ready-made MCP server

The weather and currency integrations are exposed through the **custom MCP server shipped in this repository**. The assignment can therefore be evaluated as a client/server MCP implementation rather than as a wrapper around a pre-built travel MCP server.

## 4. Prompt and context strategy

The prompt contains explicit provenance contracts:

1. Destination facts must come only from retrieved KB passages.
2. Current information must come only from MCP tool results.
3. Missing KB evidence must be reported instead of fabricated.
4. Failed tools must be reported instead of estimated.
5. Recommendations may combine evidence, but recommendations must remain clearly labelled as recommendations.
6. The final response includes a `Sources` block listing used KB sources and MCP tools.

The user message is assembled into separate `TRAVELLER PREFERENCES`, `KNOWLEDGE BASE CONTEXT`, and `LIVE TOOL RESULTS` blocks, followed by the question. This keeps provenance boundaries visible to the model.

Conversation uses a rolling history, sticky traveller preferences, follow-up expansion for contextual retrieval, and preference reuse for later tool calls.

## 5. LLM configuration

Any LangChain chat model can be used. `LLM_PROVIDER=auto` selects the first configured provider among OpenAI, Google Gemini, Anthropic and Ollama. For evaluation, choose and document one concrete provider/model in `.env` so the reviewer can reproduce the same demo. The repository remains provider-agnostic.

With no credentials, the grounded template composer still exercises retrieval, MCP invocation, provenance, conversation memory and refusal behaviour without pretending that an LLM generated the text.

## 6. Development setup

Python 3.11+ is supported.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create `.env` from `.env.example` and configure the LLM provider you intend to demo.

Build the local knowledge base/index:

```bash
python scripts/build_kb.py
```

Run the Streamlit application:

```bash
python -m streamlit run app/streamlit_app.py
```

The application opens at `http://localhost:8501`.

A terminal-only smoke/demo path is also available:

```bash
python scripts/chat.py
```

The terminal path is supported, but the **Streamlit UI is the recommended evaluator experience** because it shows the user/assistant interaction and provenance panels.

## 7. Demo / evaluator checklist

The assignment evaluator should be able to run these flows:

1. **RAG:** `What are the must-visit attractions in Singapore?` → KB-grounded answer with sources and no MCP call.
2. **Weather MCP:** `What is the weather forecast in Singapore for the next three days?` → weather tool call and live data.
3. **Currency MCP:** `Convert INR 60,000 to SGD.` → currency tool call and live rate.
4. **Combined RAG + MCP:** `Create a three-day Singapore itinerary for next week and adjust it according to the weather forecast.` → KB attractions + MCP weather + day-wise recommendations.
5. **Conversation:** set a budget/family preference, then ask `What indoor options do we have if it rains?` → prior context is retained.

A repeatable scripted demonstration is available:

```bash
python scripts/demo.py
```

or:

```bash
python scripts/demo.py --write docs/SAMPLE_QA.md
```

### Short demo video

A short video is recommended for evaluation. Record the Streamlit window while showing the five flows above. Keep it focused on the user/assistant interaction, source/provenance panel, MCP tool calls, the combined RAG+MCP response, and retained conversational context.

## 8. Tests

Run locally:

```bash
python -m pytest -q
```

Tests cover routing, tool-argument extraction, RAG/MCP combinations, currency edge cases and prompt evidence-boundary contracts.

## 9. Scope

The assistant is intentionally limited to travel planning. It does not perform bookings, payments, reservations or route navigation.

## 10. Important URLs

| URL | Purpose |
|---|---|
| [GitHub Repository](https://github.com/aranimondal/ai-travel-assistant) | Main project repository containing the complete source code, knowledge base, MCP server/client, tests and documentation. |
| [README](https://github.com/aranimondal/ai-travel-assistant/blob/main/README.md) | Main project documentation, architecture, setup instructions, demo flows and evaluator checklist. |
| [Development Setup & Local Evaluation](docs/DEVELOPMENT_SETUP.md) | Complete evaluator runbook for cloning the repository, creating the Python environment, installing dependencies, configuring the LLM, building the KB, starting the Streamlit app in development mode, running tests, and verifying RAG + MCP + LLM flows. |
| [Local Streamlit App](http://localhost:8501) | Application URL after starting Streamlit with `python -m streamlit run app/streamlit_app.py`. This is the local development/evaluation URL; the project intentionally has no deployed public application URL. |
| [Localhost](http://localhost:8501) | Direct browser endpoint for the Streamlit application during local execution. |
| [Open-Meteo](https://open-meteo.com/) | Public weather data source used by the custom MCP weather tool. |
| [Frankfurter](https://www.frankfurter.app/) | Public exchange-rate service used by the custom MCP currency tool. |
| [Model Context Protocol](https://modelcontextprotocol.io/) | Protocol documentation relevant to the custom MCP client/server implementation. |
| [Ollama](https://ollama.com/) | Local LLM runtime supported by the provider-agnostic LLM integration. |
| [LangChain](https://www.langchain.com/) | Framework used for LLM orchestration and MCP/RAG integration. |
| [FAISS](https://github.com/facebookresearch/faiss) | Vector similarity-search library used for the local RAG index. |
| [FastEmbed](https://qdrant.github.io/fastembed/) | Local embedding library used to generate semantic retrieval vectors. |

> **Note:** The application is intentionally configured for local evaluation. There is no Render/AWS/other hosted deployment URL, Docker deployment, or GitHub Actions deployment workflow in this project.