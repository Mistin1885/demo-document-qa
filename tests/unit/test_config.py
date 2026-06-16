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
    """Inject the minimum required env vars and clear the lru_cache.

    Also clears APP_ENV / EMBEDDING_DIM so per-test overrides start from the
    documented defaults (conftest pre-sets APP_ENV=test for integration tests).
    """
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
    assert s.embedding_dim == 1024
    assert s.app_env == "development"
    assert s.database_url == REQUIRED_ENV["DATABASE_URL"]
    assert s.app_encryption_key == REQUIRED_ENV["APP_ENCRYPTION_KEY"]
    assert s.openai_api_key is None
    assert s.gemini_api_key is None


def test_env_override_mineru_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """MINERU_SERVER_URL env var overrides the default."""
    _patch_required(monkeypatch)
    monkeypatch.setenv("MINERU_SERVER_URL", "http://remote-vlm:9000")
    get_settings.cache_clear()

    s = get_settings()

    assert s.mineru_server_url == "http://remote-vlm:9000"


def test_embedding_dim_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """EMBEDDING_DIM=1536 is read back as an int."""
    _patch_required(monkeypatch)
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    get_settings.cache_clear()

    s = get_settings()

    assert s.embedding_dim == 1536
    assert isinstance(s.embedding_dim, int)


def test_app_env_production(monkeypatch: pytest.MonkeyPatch) -> None:
    """APP_ENV can be set to 'production'."""
    _patch_required(monkeypatch)
    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()

    s = get_settings()

    assert s.app_env == "production"


def test_optional_api_keys_populated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional SecretStr keys are parsed when present."""
    _patch_required(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-test-key")
    get_settings.cache_clear()

    s = get_settings()

    assert s.openai_api_key is not None
    assert s.gemini_api_key is not None


# ---------------------------------------------------------------------------
# Security: SecretStr must NOT leak in repr / str
# ---------------------------------------------------------------------------


def test_secret_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain-text API key values must not appear in repr(settings)."""
    _patch_required(monkeypatch)
    plain_openai = "sk-super-secret-openai-key-12345"
    plain_gemini = "gem-super-secret-gemini-key-67890"
    monkeypatch.setenv("OPENAI_API_KEY", plain_openai)
    monkeypatch.setenv("GEMINI_API_KEY", plain_gemini)
    get_settings.cache_clear()

    s = get_settings()
    representation = repr(s)

    assert plain_openai not in representation, (
        "openai_api_key plain text must not appear in repr()"
    )
    assert plain_gemini not in representation, (
        "gemini_api_key plain text must not appear in repr()"
    )


def test_secret_not_in_str(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain-text API key values must not appear in str(settings)."""
    _patch_required(monkeypatch)
    plain_openai = "sk-another-secret-openai-key"
    monkeypatch.setenv("OPENAI_API_KEY", plain_openai)
    get_settings.cache_clear()

    s = get_settings()
    text = str(s)

    assert plain_openai not in text, (
        "openai_api_key plain text must not appear in str()"
    )


def test_secret_get_secret_value_accessible(monkeypatch: pytest.MonkeyPatch) -> None:
    """The raw key is accessible via .get_secret_value() but not in repr."""
    _patch_required(monkeypatch)
    raw = "sk-accessible-via-method-only"
    monkeypatch.setenv("OPENAI_API_KEY", raw)
    get_settings.cache_clear()

    s = get_settings()

    assert s.openai_api_key is not None
    assert s.openai_api_key.get_secret_value() == raw
    assert raw not in repr(s)


# ---------------------------------------------------------------------------
# Validation errors for required fields
# ---------------------------------------------------------------------------


def test_missing_database_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings must raise ValidationError when DATABASE_URL is absent."""
    monkeypatch.setenv("APP_ENCRYPTION_KEY", "some-key")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_settings.cache_clear()

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # bypass .env file; force env-only resolution


def test_missing_encryption_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings must raise ValidationError when APP_ENCRYPTION_KEY is absent."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@h/db")
    monkeypatch.delenv("APP_ENCRYPTION_KEY", raising=False)
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


def test_cache_clear_allows_new_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """After cache_clear(), a new Settings object is constructed."""
    _patch_required(monkeypatch)
    s1 = get_settings()

    get_settings.cache_clear()
    s2 = get_settings()

    # They are equal in content but a freshly constructed object
    assert s1 is not s2
    assert s1.mineru_server_url == s2.mineru_server_url
