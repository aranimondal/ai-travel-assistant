"""Shared fixtures."""

from __future__ import annotations

import pytest

from travel_assistant.config import Settings, get_settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def index_available(settings: Settings) -> bool:
    return (settings.faiss_dir / "index.faiss").exists()
