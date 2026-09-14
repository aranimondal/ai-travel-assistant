"""Regression tests for the evidence and tool-boundary contract in the system prompt."""

from travel_assistant.prompts import SYSTEM_PROMPT



def test_prompt_requires_kb_grounding_for_destination_facts() -> None:
    assert "must come only from the" in SYSTEM_PROMPT
    assert "knowledge-base passages" in SYSTEM_PROMPT



def test_prompt_requires_mcp_for_current_information() -> None:
    assert "Current information" in SYSTEM_PROMPT
    assert "MCP tools" in SYSTEM_PROMPT



def test_prompt_forbids_using_mcp_for_ordinary_destination_questions() -> None:
    assert "MCP is not a destination-knowledge source" in SYSTEM_PROMPT
    assert "ordinary attraction" in SYSTEM_PROMPT



def test_prompt_requires_explicit_limits_when_evidence_is_insufficient() -> None:
    assert "prefer an explicit limitation" in SYSTEM_PROMPT
    assert "Do not invent" in SYSTEM_PROMPT


def test_prompt_requires_source_block() -> None:
    assert 'End with a "Sources" block' in SYSTEM_PROMPT
