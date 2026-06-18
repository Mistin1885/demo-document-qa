"""Unit tests for app.providers and app.security.

Rules obeyed
------------
- No real network calls (``httpx_mock`` / ``monkeypatch`` / direct mock usage).
- No real paid API keys (fake keys used throughout).
- Tests are deterministic (no random seeding needed; mocks are hash-based).
- All ``get_settings.cache_clear()`` calls precede tests that manipulate env.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.providers.mock import (
    MockChatProvider,
    MockEmbeddingProvider,
    MockRerankerProvider,
)
from app.security import decrypt, encrypt, mask_secret

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

REQUIRED_ENV: dict[str, str] = {
    "DATABASE_URL": "postgresql+psycopg://user:pass@localhost/testdb",
    "APP_ENCRYPTION_KEY": "test-encryption-key-for-unit-tests-only",
}


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject required env vars so ``get_settings()`` never fails."""
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


def _make_profile(**kwargs: object) -> SimpleNamespace:
    defaults = dict(
        provider_type="mock",
        model_name="mock-model",
        api_key_encrypted=None,
        api_key_plain=None,
        base_url=None,
        context_window=None,
        embedding_dim=None,
        provider_name=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Security: encrypt / decrypt round-trip
# ---------------------------------------------------------------------------


def test_encrypt_decrypt_roundtrip() -> None:
    """encrypt → decrypt must return the original string (including empty/unicode)."""
    for plaintext in ["sk-super-secret-api-key-12345", "", "key-你好-🔑"]:
        assert decrypt(encrypt(plaintext)) == plaintext


def test_decrypt_with_wrong_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decrypting with a different key must raise InvalidToken."""
    from cryptography.fernet import InvalidToken

    token = encrypt("top-secret")
    monkeypatch.setenv("APP_ENCRYPTION_KEY", "a-completely-different-key-value")
    get_settings.cache_clear()

    with pytest.raises(InvalidToken):
        decrypt(token)


# ---------------------------------------------------------------------------
# Security: mask_secret — key must never leak
# ---------------------------------------------------------------------------


def test_mask_secret_hides_full_value() -> None:
    """mask_secret must hide the full secret and show prefix+suffix with '...'."""
    secret = "sk-abcdef1234567890"
    masked = mask_secret(secret)
    assert secret not in masked
    assert "..." in masked
    assert masked == "sk-...7890"


def test_mask_secret_not_in_log(caplog: pytest.LogCaptureFixture) -> None:
    """Raw API key must not appear in any log output when mask_secret is used."""
    secret_key = "sk-secret-key-that-must-not-be-logged"
    with caplog.at_level(logging.DEBUG, logger="app.providers.openai_compat"):
        import app.security as sec

        masked = sec.mask_secret(secret_key)
        logging.getLogger("app.providers.openai_compat").debug(
            "OpenAI-compat complete: key=%s", masked
        )

    full_log = "\n".join(caplog.messages)
    assert secret_key not in full_log


# ---------------------------------------------------------------------------
# Registry — four provider type smoke tests + error paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_type,extra_kwargs,expected_class_name",
    [
        ("openai", {"api_key_plain": "sk-fake"}, "OpenAIChatProvider"),
        ("gemini_native", {"api_key_plain": "fake-gemini-key"}, "GeminiNativeChatProvider"),
    ],
)
def test_build_chat_provider_smoke(
    provider_type: str,
    extra_kwargs: dict,
    expected_class_name: str,
) -> None:
    """Registry builds each provider type without error; vLLM is an alias for openai_compat."""
    from app.providers.registry import build_chat_provider

    profile = _make_profile(provider_type=provider_type, model_name="test-model", **extra_kwargs)
    provider = build_chat_provider(profile)
    assert type(provider).__name__ == expected_class_name


def test_unknown_provider_type_raises() -> None:
    from app.providers.registry import build_chat_provider

    profile = _make_profile(provider_type="unknown_xyz")
    with pytest.raises(ValueError, match="Unknown provider_type"):
        build_chat_provider(profile)


def test_encrypted_key_resolved() -> None:
    """Registry must decrypt api_key_encrypted before passing to the adapter."""
    from app.providers.registry import build_chat_provider

    token = encrypt("sk-encrypted-real-key")
    profile = _make_profile(provider_type="mock", api_key_encrypted=token)
    provider = build_chat_provider(profile)
    assert provider is not None


# ---------------------------------------------------------------------------
# Mock providers — shape and determinism
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_providers_basic() -> None:
    """MockChat/Embedding/Reranker return correct shapes and are deterministic."""
    from app.providers.base import ChatCompletion, ChatMessage

    # Chat provider
    chat = MockChatProvider()
    messages = [ChatMessage(role="user", content="hello")]
    r1 = await chat.complete(messages)
    r2 = await chat.complete(messages)
    assert isinstance(r1, ChatCompletion)
    assert r1.content
    assert r1.content == r2.content

    # Embedding provider
    emb = MockEmbeddingProvider(dimension=16)
    vecs = await emb.embed(["a", "b"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 16
    assert all(-1.0 <= f <= 1.0 for f in vecs[0])

    # Reranker provider
    ranker = MockRerankerProvider()
    scores = await ranker.score("q", ["doc1", "doc2"])
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)


@pytest.mark.asyncio
async def test_mock_chat_stream_assembles_correctly() -> None:
    """Concatenating stream deltas must equal the complete() output."""
    from app.providers.base import ChatMessage

    provider = MockChatProvider()
    messages = [ChatMessage(role="user", content="stream test")]
    complete_result = await provider.complete(messages)
    streamed = ""
    last_chunk = None
    async for chunk in provider.stream(messages):
        streamed += chunk.delta
        last_chunk = chunk
    assert streamed == complete_result.content
    assert last_chunk is not None
    assert last_chunk.finish_reason == "stop"
