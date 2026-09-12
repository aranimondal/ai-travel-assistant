"""Interactive command-line chat, useful for testing without the browser.

    python scripts/chat.py
    python scripts/chat.py --ask "What are the must-visit attractions in Singapore?"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from travel_assistant.assistant import Answer, TravelAssistant  # noqa: E402


def print_status(assistant: TravelAssistant) -> None:
    status = assistant.status()
    kb = status["knowledge_base"]
    mcp = status["mcp"]
    print(f"Destination      : {status['destination']}")
    print(
        "Knowledge base   : "
        + (
            f"indexed ({kb['embedding_model']})"
            if kb["index_available"]
            else "NOT INDEXED - run python scripts/build_kb.py"
        )
    )
    print(
        "MCP tools        : "
        + (
            ", ".join(tool["name"] for tool in mcp["tools"])
            if mcp["connected"]
            else f"unavailable ({mcp['error']})"
        )
    )
    print(f"Generator        : {status['generator']['label']}")
    print()


def print_answer(answer: Answer, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(answer.as_dict(), indent=2, ensure_ascii=False))
        return

    print(answer.text)
    print()
    print(f"[route: {answer.intent.label} | generator: {answer.generator}]")
    for citation in answer.citations:
        section = f" — {citation['section']}" if citation["section"] else ""
        print(f"  KB  {citation['title']}{section} (similarity {citation['score']})")
        if citation["url"]:
            print(f"      {citation['url']}")
    for call in answer.tool_calls:
        state = "ok" if call.ok else f"FAILED: {call.error}"
        print(f"  MCP {call.tool} {call.arguments} -> {state}")
    for note in answer.notes:
        print(f"  note: {note}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with the travel assistant.")
    parser.add_argument("--ask", action="append", help="ask a question and exit (repeatable)")
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    parser.add_argument("--verbose", action="store_true", help="show library logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    assistant = TravelAssistant()
    if not args.json:
        print_status(assistant)

    if args.ask:
        for question in args.ask:
            if not args.json:
                print(f"> {question}\n")
            print_answer(assistant.ask(question), as_json=args.json)
        return 0

    print("Type a question, or 'exit' to quit. 'reset' clears the conversation.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit"}:
            return 0
        if question.lower() == "reset":
            assistant.reset()
            print("Conversation cleared.\n")
            continue
        if not question:
            continue
        print()
        print_answer(assistant.ask(question), as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
