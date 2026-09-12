"""Streamlit chat interface for the travel assistant.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from travel_assistant.assistant import Answer, TravelAssistant  # noqa: E402

EXAMPLES = [
    "What are the must-visit attractions in Singapore?",
    "How can a tourist travel around Singapore?",
    "What is the weather in Singapore for the next three days?",
    "Convert INR 60,000 to SGD.",
    "Create a three-day Singapore itinerary for next week and adjust it to the weather forecast.",
]


@st.cache_resource(show_spinner="Starting the assistant (loading index and MCP tools) ...")
def get_assistant() -> TravelAssistant:
    return TravelAssistant()


def render_provenance(answer: Answer) -> None:
    """Show where the answer came from: KB citations, MCP calls, routing notes."""
    st.caption(f"Route: **{answer.intent.label}** · Generator: **{answer.generator}**")

    if answer.citations:
        with st.expander(f"Knowledge-base sources ({len(answer.citations)})"):
            for citation in answer.citations:
                heading = citation["title"]
                if citation["section"]:
                    heading += f" — {citation['section']}"
                st.markdown(f"**{heading}** · similarity {citation['score']}")
                if citation["url"]:
                    st.markdown(citation["url"])

    if answer.tool_calls:
        with st.expander(f"MCP tool calls ({len(answer.tool_calls)})"):
            for call in answer.tool_calls:
                status = "succeeded" if call.ok else "failed"
                st.markdown(f"`{call.tool}` — {status} · arguments `{call.arguments}`")
                if call.ok:
                    st.json(call.result, expanded=False)
                else:
                    st.warning(call.error)

    if answer.notes:
        with st.expander("Notes"):
            for note in answer.notes:
                st.markdown(f"- {note}")


def sidebar(assistant: TravelAssistant) -> None:
    status = assistant.status()
    st.sidebar.title("Assistant status")
    st.sidebar.markdown(f"**Destination:** {status['destination']}")

    kb = status["knowledge_base"]
    if kb["index_available"]:
        st.sidebar.success(f"Knowledge base indexed\n\nEmbeddings: `{kb['embedding_model']}`")
    else:
        st.sidebar.error("No FAISS index found. Run `python scripts/build_kb.py`.")

    mcp = status["mcp"]
    if mcp["connected"]:
        names = ", ".join(f"`{tool['name']}`" for tool in mcp["tools"])
        st.sidebar.success(f"MCP server connected\n\nTools: {names}")
    else:
        st.sidebar.error(f"MCP tools unavailable: {mcp['error']}")

    generator = status["generator"]
    st.sidebar.info(f"**Generator:** {generator['label']}\n\n{generator['reason']}")

    if assistant.conversation.preferences:
        st.sidebar.markdown("**Remembered preferences**")
        for key, value in sorted(assistant.conversation.preferences.items()):
            st.sidebar.markdown(f"- {key.replace('_', ' ')}: {value}")

    if st.sidebar.button("Clear conversation"):
        assistant.reset()
        st.session_state.transcript = []
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="AI Travel Planning Assistant", page_icon="🧭", layout="wide")
    st.title("AI Travel Planning Assistant")
    st.caption(
        "Destination knowledge from an indexed travel knowledge base (RAG), "
        "current conditions and exchange rates from MCP tools."
    )

    assistant = get_assistant()
    sidebar(assistant)
    st.session_state.setdefault("transcript", [])

    with st.expander("Example questions"):
        for example in EXAMPLES:
            st.markdown(f"- {example}")

    for entry in st.session_state.transcript:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("answer") is not None:
                render_provenance(entry["answer"])

    question = st.chat_input("Ask about Singapore, the weather, or your budget")
    if not question:
        return

    st.session_state.transcript.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"), st.spinner("Retrieving knowledge and calling tools ..."):
        try:
            answer = assistant.ask(question)
        except Exception as exc:  # noqa: BLE001 - keep the UI usable on unexpected errors
            st.error(f"The request could not be completed: {exc}")
            return
        st.markdown(answer.text)
        render_provenance(answer)

    st.session_state.transcript.append(
        {"role": "assistant", "content": answer.text, "answer": answer}
    )


main()
