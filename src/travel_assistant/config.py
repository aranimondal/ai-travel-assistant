"""Runtime configuration, loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Destination -----------------------------------------------------
    destination_name: str = "Singapore"
    destination_latitude: float = 1.3521
    destination_longitude: float = 103.8198
    destination_currency: str = "SGD"
    destination_timezone: str = "Asia/Singapore"

    # --- Knowledge base --------------------------------------------------
    kb_dir: Path = PROJECT_ROOT / "knowledge_base" / "documents"
    index_dir: Path = PROJECT_ROOT / "knowledge_base" / "index"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 1100
    chunk_overlap: int = 150
    retrieval_top_k: int = 6
    # Cosine similarity below this is treated as "not in the knowledge base".
    retrieval_min_score: float = 0.32

    # --- LLM -------------------------------------------------------------
    # "auto" picks the first provider with credentials, else the grounded
    # template composer (deterministic, no credentials required).
    llm_provider: str = "auto"
    llm_model: str = ""
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1200
    openai_api_key: str = ""
    openai_base_url: str = ""
    google_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # --- MCP -------------------------------------------------------------
    mcp_server_command: str = ""  # defaults to the running interpreter
    mcp_server_args: list[str] = Field(default_factory=lambda: ["-m", "travel_mcp.server"])
    mcp_tool_timeout_seconds: float = 20.0

    # --- Conversation ----------------------------------------------------
    history_turns: int = 6

    @property
    def faiss_dir(self) -> Path:
        return self.index_dir / "faiss"


@lru_cache
def get_settings() -> Settings:
    return Settings()
