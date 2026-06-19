"""Unit tests for app.config — Settings loading and security invariants.

Rules obeyed
------------
- No real .env file is read: ``monkeypatch.setenv`` is used to inject every
  required variable, and ``monkeypatch.delenv`` removes ones that must be
  absent.
- ``get_settings.cache_clear()`` is called before each test so that
  ``monkeypatch`` mutations take effect.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_ENV: dict[str, str] = {
    "DATABASE_URL": "postgresql+psycopg://user:pass@localhost/testdb",
    "APP_ENCRYPTION_KEY": "super-secret-key-for-tests",
}


def _patch_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject the minimum required env vars and clear the lru_cache."""
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Happy-path: defaults and overrides
# ---------------------------------------------------------------------------


def test_default_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Required fields work; optional fields fall back to documented defaults."""
    _patch_required(monkeypatch)

    s = get_settings()

    assert s.mineru_server_url == "http://localhost:8001"
    assert s.vespa_endpoint == "http://localhost:8080"
    assert s.embedding_dim == 384
    assert s.app_env == "development"
    assert s.database_url == REQUIRED_ENV["DATABASE_URL"]
    assert s.openai_api_key is None
    assert s.gemini_api_key is None


def test_embedding_dim_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """EMBEDDING_DIM and custom env vars are read back correctly."""
    _patch_required(monkeypatch)
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("MINERU_SERVER_URL", "http://remote-vlm:9000")
    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()

    s = get_settings()

    assert s.embedding_dim == 1536
    assert isinstance(s.embedding_dim, int)
    assert s.mineru_server_url == "http://remote-vlm:9000"
    assert s.app_env == "production"


# ---------------------------------------------------------------------------
# Security: SecretStr must NOT leak in repr / str
# ---------------------------------------------------------------------------


def test_secret_not_in_repr_or_str(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain-text API key values must not appear in repr(settings) or str(settings)."""
    _patch_required(monkeypatch)
    plain_openai = "sk-super-secret-openai-key-12345"
    plain_gemini = "gem-super-secret-gemini-key-67890"
    monkeypatch.setenv("OPENAI_API_KEY", plain_openai)
    monkeypatch.setenv("GEMINI_API_KEY", plain_gemini)
    get_settings.cache_clear()

    s = get_settings()
    representation = repr(s)
    text = str(s)

    assert plain_openai not in representation
    assert plain_gemini not in representation
    assert plain_openai not in text

    # Raw value still accessible via get_secret_value()
    assert s.openai_api_key is not None
    assert s.openai_api_key.get_secret_value() == plain_openai


# ---------------------------------------------------------------------------
# Validation errors for required fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing_env,present_env",
    [
        ("DATABASE_URL", {"APP_ENCRYPTION_KEY": "some-key"}),
        ("APP_ENCRYPTION_KEY", {"DATABASE_URL": "postgresql+psycopg://u:p@h/db"}),
    ],
)
def test_missing_required_field_raises(
    monkeypatch: pytest.MonkeyPatch,
    missing_env: str,
    present_env: dict,
) -> None:
    """Settings must raise ValidationError when a required env var is absent."""
    for key, value in present_env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv(missing_env, raising=False)
    get_settings.cache_clear()

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


def test_cache_returns_same_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_settings() returns the same object on repeated calls."""
    _patch_required(monkeypatch)

    s1 = get_settings()
    s2 = get_settings()

    assert s1 is s2
