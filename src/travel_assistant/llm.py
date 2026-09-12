"""LLM selection.

Any LangChain chat model can be plugged in; ``auto`` picks the first provider
with credentials. When no provider is configured the assistant falls back to a
deterministic template composer so the RAG + MCP pipeline stays fully
demonstrable without an API key. The fallback is always reported as such -- it
is never presented as model output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from travel_assistant.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "google": "gemini-2.0-flash",
    "anthropic": "claude-3-5-haiku-latest",
    "ollama": "llama3.1:8b",
}

INSTALL_HINTS = {
    "openai": "pip install langchain-openai",
    "google": "pip install langchain-google-genai",
    "anthropic": "pip install langchain-anthropic",
    "ollama": "pip install langchain-ollama  (and run an Ollama server)",
}


@dataclass(frozen=True, slots=True)
class LlmSelection:
    """Which generator the assistant is using, and why."""

    provider: str  # "openai" | "google" | "anthropic" | "ollama" | "template"
    model_name: str
    chat_model: BaseChatModel | None
    reason: str

    @property
    def is_llm(self) -> bool:
        return self.chat_model is not None

    @property
    def label(self) -> str:
        if self.is_llm:
            return f"{self.provider}:{self.model_name}"
        return "grounded template composer (no LLM credentials configured)"


def _build(provider: str, settings: Settings) -> BaseChatModel:
    model = settings.llm_model or DEFAULT_MODELS[provider]
    common = {"temperature": settings.llm_temperature}

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, object] = {
            "model": model,
            "api_key": settings.openai_api_key,
            "max_tokens": settings.llm_max_tokens,
            **common,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.google_api_key,
            max_output_tokens=settings.llm_max_tokens,
            **common,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key,
            max_tokens=settings.llm_max_tokens,
            **common,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, base_url=settings.ollama_base_url, **common)

    raise ValueError(f"Unknown LLM provider '{provider}'")


def _credentialled_providers(settings: Settings) -> list[str]:
    available = []
    if settings.openai_api_key:
        available.append("openai")
    if settings.google_api_key:
        available.append("google")
    if settings.anthropic_api_key:
        available.append("anthropic")
    return available


def select_llm(settings: Settings) -> LlmSelection:
    """Resolve the configured provider, degrading to the template composer."""
    requested = (settings.llm_provider or "auto").strip().lower()

    if requested in {"", "none", "template"}:
        return LlmSelection(
            "template", "-", None, "LLM_PROVIDER is set to 'template'; no model is called."
        )

    candidates = _credentialled_providers(settings) if requested == "auto" else [requested]

    for provider in candidates:
        try:
            chat_model = _build(provider, settings)
        except ImportError as exc:
            logger.warning("provider %s unavailable: %s", provider, exc)
            if requested != "auto":
                return LlmSelection(
                    "template",
                    "-",
                    None,
                    f"LLM_PROVIDER={provider} requested but its package is missing "
                    f"({INSTALL_HINTS.get(provider, 'install the integration package')}).",
                )
            continue
        except Exception as exc:  # noqa: BLE001 - bad credentials/config
            logger.warning("provider %s could not be initialised: %s", provider, exc)
            if requested != "auto":
                return LlmSelection(
                    "template", "-", None, f"LLM_PROVIDER={provider} failed to initialise: {exc}"
                )
            continue

        model_name = settings.llm_model or DEFAULT_MODELS[provider]
        return LlmSelection(provider, model_name, chat_model, "Configured chat model in use.")

    return LlmSelection(
        "template",
        "-",
        None,
        "No LLM credentials found (set OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY, "
        "or LLM_PROVIDER=ollama). Answers are composed from retrieved knowledge-base "
        "passages and MCP tool results without a language model.",
    )
