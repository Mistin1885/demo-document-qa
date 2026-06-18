"""Application configuration loaded from environment variables / .env file.

Usage
-----
    from app.config import get_settings

    settings = get_settings()          # cached; safe to call repeatedly
    get_settings.cache_clear()         # force re-read (tests only)

Security contract
-----------------
- ``openai_api_key`` and ``gemini_api_key`` are ``SecretStr``; their plain
  values are never interpolated into logs or repr strings.
- ``app_encryption_key`` is a plain ``str`` (used as a symmetric key for
  encrypting provider credentials at rest); it must never be logged.
- No module-level ``Settings`` singleton is exported; all callers go through
  ``get_settings()``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised application settings.

    All values are read from environment variables (case-insensitive) or from
    a ``.env`` file in the working directory.  Unknown variables are silently
    ignored (``extra="ignore"``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- MinerU hybrid client ---
    mineru_server_url: str = "http://localhost:8001"

    # --- Database (required — raises ValidationError if absent) ---
    database_url: str

    # --- Vespa ---
    vespa_endpoint: str = "http://localhost:8080"
    vespa_enabled: bool = True
    """Set to False to use the no-op NullVespaIndexer (useful in CI / test)."""

    # --- Application secrets (required — raises ValidationError if absent) ---
    app_encryption_key: str

    # --- LLM / embedding provider keys (optional) ---
    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    # --- Env-level default LLM (used when a chat has no default_chat_profile) ---
    # Lets you point the QA agent at a real LLM without inserting a
    # provider_profiles row.  Mirrors OpenAICompatChatProvider parameters.
    llm_provider: Literal["openai_compatible", "openai", "gemini_native", "mock"] = "mock"
    llm_api_url: str | None = None
    """Base URL of the OpenAI-compatible server (e.g. ``http://host:8599/v1``).

    A trailing ``/chat/completions`` is stripped automatically because the
    OpenAI SDK appends it.
    """
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_max_tokens: int = 2048
    """Max output tokens for the answer call (used when the request body
    leaves ``max_answer_tokens`` unset). Frontend may override per-request."""

    llm_temperature: float = 0.0
    llm_timeout: float = 60.0
    llm_context_window: int = 10_000
    """Total input-side token budget (system + history + plan + evidence +
    answer_reserve). Drives ``ContextBudgetManager`` when no per-request
    override is supplied."""

    # --- Vespa embedding dimension (Phase 6 schema DIM) ---
    embedding_dim: int = 1024

    # --- Local file storage root (Phase 3.3) ---
    app_data_root: str = "data"
    """Root directory for local blob storage.  Storage path: {app_data_root}/storage/."""

    # --- Runtime environment ---
    app_env: Literal["development", "production", "test"] = "development"

    # --- CORS ---
    # Comma-separated list of extra allowed origins. Localhost dev origins are
    # always allowed; this is for containerised / remote frontends.
    cors_extra_origins: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application ``Settings`` instance.

    Call ``get_settings.cache_clear()`` in tests to force a re-read after
    mutating environment variables via ``monkeypatch``.
    """
    return Settings()
