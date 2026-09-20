# Development Setup & Local Evaluation Guide

This document is the evaluator-focused runbook for starting the **AI Travel Planning Assistant — Singapore** in development mode and verifying the complete RAG + custom MCP + LLM workflow locally.

## 1. Prerequisites

Install the following on the evaluation machine:

- **Python 3.11 or newer**
- **Git**
- A supported LLM provider. For a fully local evaluation, **Ollama** can be used.
- Internet access for the public data sources used by the MCP tools and, when rebuilding the KB, the configured source pages.

No Docker, cloud deployment, Render/AWS hosting, or GitHub Actions workflow is required for local evaluation.

## 2. Get the repository

Clone the repository and enter the project directory:

```bash
git clone https://github.com/aranimondal/ai-travel-assistant.git
cd ai-travel-assistant
```

Confirm that the evaluator is on the `main` branch:

```bash
git checkout main
git pull
```

## 3. Create and activate the Python environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current user, activate from Command Prompt instead:

```bat
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Verify the interpreter:

```bash
python --version
```

Python 3.11+ is supported.

## 4. Install dependencies

Upgrade pip and install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For running the test suite, install the development dependencies as well:

```bash
python -m pip install -r requirements-dev.txt
```

## 5. Configure the environment

Create the local environment file from the provided template:

### Windows

```powershell
Copy-Item .env.example .env
```

### macOS / Linux

```bash
cp .env.example .env
```

The application is provider-agnostic. The repository supports OpenAI, Google, Anthropic, Ollama and the grounded template fallback.

For the recommended self-contained local development/demo setup, configure Ollama in `.env`:

```text
LLM_PROVIDER=ollama
LLM_MODEL=<installed-ollama-model>
OLLAMA_BASE_URL=http://localhost:11434
```

Use the name of a model that is actually installed in the evaluator's Ollama instance.

The remaining values in `.env.example` have development defaults and normally do not need to be changed.

## 6. Optional: start the local Ollama LLM

If using the local Ollama provider, install Ollama from the official site and make sure the Ollama service is running.

Verify that Ollama is reachable:

```bash
ollama list
```

If the evaluator has not downloaded a model yet, pull the model they intend to use, for example:

```bash
ollama pull <model-name>
```

Then set the same model name in `.env`:

```text
LLM_PROVIDER=ollama
LLM_MODEL=<model-name>
OLLAMA_BASE_URL=http://localhost:11434
```

The project sends the retrieved RAG context and MCP tool results to the configured LLM; the LLM generates the final response.

## 7. Build / verify the local knowledge base

The repository contains the Singapore knowledge base and persisted retrieval assets. To rebuild the knowledge base from the configured public sources:

```bash
python scripts/build_kb.py
```

Wait for the command to complete successfully before starting the application.

The resulting local index is used by the RAG retriever. If the evaluator only wants to run the already-prepared repository, the checked-in knowledge-base assets can be used directly; rebuilding is useful when validating the ingestion pipeline.

## 8. Start the application in development mode

With the virtual environment active and the environment configured:

```bash
python -m streamlit run app/streamlit_app.py
```

Open the local application in a browser:

**http://localhost:8501**

The Streamlit UI is the recommended evaluator entry point because it exposes the conversation, knowledge-base provenance, MCP tool calls/results, and generator information.

## 9. Verify the custom MCP server

The application starts the repository's custom MCP server as a local stdio subprocess when an MCP tool is required.

The custom server is implemented in:

```text
src/travel_mcp/server.py
```

It exposes:

- `get_weather_forecast`
- `convert_currency`

The evaluator does **not** need to install a separate ready-made travel MCP server.

## 10. Recommended evaluator checks

Run the following questions through the Streamlit UI.

### A. RAG-only

```text
What are the must-visit attractions in Singapore?
```

Expected behavior:
- Uses the local knowledge base.
- Shows knowledge-base sources.
- Does not require a weather/currency MCP call.

### B. Weather MCP

```text
What is the weather forecast in Singapore for the next three days?
```

Expected behavior:
- Calls `get_weather_forecast`.
- Displays the MCP tool call and live result.

### C. Currency MCP

```text
Convert INR 60,000 to SGD.
```

Expected behavior:
- Calls `convert_currency`.
- Displays the live conversion result and source information.

### D. Combined RAG + MCP + LLM

```text
Create a three-day Singapore itinerary for next week and adjust it according to the weather forecast.
```

Expected behavior:
- Retrieves Singapore attractions/itinerary guidance from RAG.
- Calls the weather MCP tool for current forecast information.
- Sends the separated KB evidence and live MCP result to the configured LLM.
- Returns a weather-aware, day-wise itinerary with sources/provenance.

### E. Multi-turn context

Start with a preference such as:

```text
I am travelling with family and have a budget of INR 60,000.
```

Then ask:

```text
What indoor options do we have if it rains?
```

Expected behavior:
- Retains the relevant conversation/preferences.
- Uses the appropriate KB and MCP information for the follow-up.
- Does not require the user to repeat the preference.

## 11. Terminal smoke test

A CLI path is also available:

```bash
python scripts/chat.py
```

A repeatable demo script is available:

```bash
python scripts/demo.py
```

To write the sample Q&A output to a markdown file:

```bash
python scripts/demo.py --write docs/SAMPLE_QA.md
```

## 12. Run the test suite

With the development dependencies installed:

```bash
python -m pytest -q
```

This validates the project's routing, prompt contracts, MCP behavior, retrieval/integration paths and related unit/integration tests.

## 13. What the evaluator should see in the UI

The Streamlit provenance/status area is intentionally useful for evaluation.

- **Route** shows the detected request type.
- **Generator** identifies the configured answer generator, such as the Ollama model, rather than hiding the generation path.
- **Knowledge-base sources** show the RAG evidence used.
- **MCP tool calls** show which custom MCP tool was invoked, its arguments, result, or failure.
- The final response is generated from the separated evidence/context when an LLM provider is configured.

If a live MCP service fails, the application should surface the failure rather than inventing a current weather or currency value.

## 14. Troubleshooting

### Streamlit does not start

Confirm the virtual environment is active and reinstall dependencies:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

### Ollama is unavailable

Check that the Ollama service is running and that the configured model exists:

```bash
ollama list
```

Also confirm:

```text
OLLAMA_BASE_URL=http://localhost:11434
LLM_PROVIDER=ollama
LLM_MODEL=<installed-ollama-model>
```

### The KB/index is missing or stale

Rebuild it:

```bash
python scripts/build_kb.py
```

### MCP data is unavailable

Check internet connectivity and retry. The weather and currency tools depend on their public upstream services. The application records tool failures instead of fabricating live values.

### Port 8501 is already in use

Start Streamlit on another local port:

```bash
python -m streamlit run app/streamlit_app.py --server.port 8502
```

Then open the displayed local URL.

## 15. Evaluator quick-start

For a normal local evaluation, the shortest path is:

```bash
git clone https://github.com/aranimondal/ai-travel-assistant.git
cd ai-travel-assistant
python -m venv .venv
```

Activate `.venv`, then:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Configure `.env` with the chosen LLM provider, build the KB if needed, and start:

```bash
python scripts/build_kb.py
python -m streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** and run the evaluator checks in Section 10.

For the complete reproducible procedure, follow Sections 1–14 above.
