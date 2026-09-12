"""MCP client bridge.

The assistant does not call the weather/currency HTTP APIs directly: it starts
``travel_mcp.server`` as an MCP server over stdio and loads its tools through
``langchain-mcp-adapters``, so the tools arrive as LangChain ``BaseTool``
objects with schemas generated from the MCP tool definitions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from travel_assistant.config import Settings

logger = logging.getLogger(__name__)

PROJECT_SRC = str(Path(__file__).resolve().parents[1])


@dataclass(slots=True)
class ToolCall:
    """Record of one MCP tool invocation, used for provenance in the answer."""

    tool: str
    arguments: dict[str, Any]
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    called_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def label(self) -> str:
        return f"MCP tool `{self.tool}`"


class McpUnavailableError(RuntimeError):
    """Raised when the MCP server cannot be started or its tools cannot be listed."""


def _coerce(raw: Any) -> Any:
    """Normalise an MCP tool result into a plain Python object.

    ``langchain-mcp-adapters`` returns the MCP content blocks, i.e. a list of
    ``{"type": "text", "text": "<json>"}`` dicts. Unwrap the text blocks and
    decode the JSON payload the tools produce.
    """
    if isinstance(raw, list):
        texts = [
            block["text"]
            for block in raw
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block
        ]
        if len(texts) == 1:
            return _coerce(texts[0])
        if texts:
            return _coerce("\n".join(texts))
        return raw
    if isinstance(raw, dict):
        # A single content block, or an already-decoded payload.
        if raw.get("type") == "text" and isinstance(raw.get("text"), str):
            return _coerce(raw["text"])
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _run(coro):
    """Run a coroutine from sync code, tolerating an already-running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Streamlit reruns can leave a loop installed on the thread; execute in a
    # dedicated thread with its own loop rather than nesting.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class McpToolbox:
    """Loads MCP tools once and invokes them by name.

    A fresh stdio session is opened per call. That is slightly slower than
    holding a long-lived session, but it keeps the Streamlit/CLI code free of
    background event-loop ownership and makes each call independently
    recoverable when the server subprocess dies.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tools: dict[str, Any] | None = None

    # --- connection ------------------------------------------------------
    def _server_config(self) -> dict[str, dict[str, Any]]:
        existing = os.environ.get("PYTHONPATH", "")
        python_path = PROJECT_SRC if not existing else f"{PROJECT_SRC}{os.pathsep}{existing}"
        return {
            "travel-live-data": {
                "command": self._settings.mcp_server_command or sys.executable,
                "args": list(self._settings.mcp_server_args),
                "transport": "stdio",
                "env": {**os.environ, "PYTHONPATH": python_path},
            }
        }

    async def _aload_tools(self) -> dict[str, Any]:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(self._server_config())
        tools = await client.get_tools()
        return {tool.name: tool for tool in tools}

    def load(self) -> dict[str, Any]:
        """Connect to the MCP server and cache its tool list."""
        if self._tools is not None:
            return self._tools
        try:
            self._tools = _run(self._aload_tools())
        except Exception as exc:  # noqa: BLE001 - surfaced as a degraded state, not a crash
            logger.error("could not load MCP tools: %s", exc)
            raise McpUnavailableError(str(exc)) from exc
        logger.info("loaded MCP tools: %s", ", ".join(sorted(self._tools)))
        return self._tools

    @property
    def status(self) -> dict[str, Any]:
        """Non-throwing view of the toolbox, for the UI status panel."""
        try:
            tools = self.load()
        except McpUnavailableError as exc:
            return {"connected": False, "tools": [], "error": str(exc)}
        return {
            "connected": True,
            "tools": [
                {"name": name, "description": (tool.description or "").strip()}
                for name, tool in sorted(tools.items())
            ],
            "error": None,
        }

    # --- invocation ------------------------------------------------------
    def call(self, name: str, arguments: dict[str, Any]) -> ToolCall:
        """Invoke an MCP tool, turning every failure mode into a ToolCall record."""
        try:
            tools = self.load()
        except McpUnavailableError as exc:
            return ToolCall(
                tool=name, arguments=arguments, ok=False, error=f"MCP server unavailable: {exc}"
            )

        tool = tools.get(name)
        if tool is None:
            return ToolCall(
                tool=name,
                arguments=arguments,
                ok=False,
                error=f"Tool '{name}' is not offered by the MCP server.",
            )

        try:
            raw = _run(
                asyncio.wait_for(
                    tool.ainvoke(arguments), timeout=self._settings.mcp_tool_timeout_seconds
                )
            )
        except TimeoutError:
            return ToolCall(
                tool=name,
                arguments=arguments,
                ok=False,
                error=f"Tool timed out after {self._settings.mcp_tool_timeout_seconds:.0f}s.",
            )
        except Exception as exc:  # noqa: BLE001 - includes MCP transport errors
            logger.warning("tool %s failed: %s", name, exc)
            return ToolCall(tool=name, arguments=arguments, ok=False, error=str(exc))

        payload = _coerce(raw)
        if isinstance(payload, dict) and payload.get("ok") is False:
            return ToolCall(
                tool=name,
                arguments=arguments,
                ok=False,
                error=str(payload.get("error", "Tool reported a failure.")),
            )
        if not isinstance(payload, dict):
            return ToolCall(
                tool=name,
                arguments=arguments,
                ok=False,
                error=f"Unexpected tool response type: {type(payload).__name__}",
            )
        return ToolCall(tool=name, arguments=arguments, ok=True, result=payload)
