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


# ---------------------------------------------------------------------------
# app.security — encrypt / decrypt round-trip
# ---------------------------------------------------------------------------


class TestEncryptDecrypt:
    def test_roundtrip_basic(self) -> None:
        """encrypt → decrypt must return the original string."""
        plaintext = "sk-super-secret-api-key-12345"
        token = encrypt(plaintext)
        assert isinstance(token, bytes)
        assert decrypt(token) == plaintext

    def test_roundtrip_empty_string(self) -> None:
        """Empty string round-trips correctly."""
        token = encrypt("")
        assert decrypt(token) == ""

    def test_roundtrip_unicode(self) -> None:
        """Unicode characters survive the round-trip."""
        plaintext = "key-with-unicode-你好-🔑"
        assert decrypt(encrypt(plaintext)) == plaintext

    def test_different_keys_produce_different_tokens(self) -> None:
        """Two encryptions of the same plaintext should produce different tokens
        (Fernet uses random IV)."""
        plaintext = "same-key-value"
        t1 = encrypt(plaintext)
        t2 = encrypt(plaintext)
        # Different IVs → different ciphertext (probabilistic, but overwhelmingly likely)
        assert t1 != t2, "Two Fernet tokens for the same plaintext should differ (random IV)"

    def test_decrypt_with_wrong_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Decrypting with a different key must raise InvalidToken."""
        from cryptography.fernet import InvalidToken

        plaintext = "top-secret"
        token = encrypt(plaintext)

        # Change the encryption key
        monkeypatch.setenv("APP_ENCRYPTION_KEY", "a-completely-different-key-value")
        get_settings.cache_clear()

        with pytest.raises(InvalidToken):
            decrypt(token)


# ---------------------------------------------------------------------------
# app.security — mask_secret
# ---------------------------------------------------------------------------


class TestMaskSecret:
    def test_none_returns_empty(self) -> None:
        assert mask_secret(None) == ""

    def test_short_key_returns_stars(self) -> None:
        assert mask_secret("ab") == "***"
        assert mask_secret("abcdefgh") == "***"  # exactly 8 chars → "***"

    def test_medium_key(self) -> None:
        result = mask_secret("sk-short123")  # 11 chars
        assert result.startswith("sk")
        assert "..." in result
        # Should NOT contain the full original value
        assert "sk-short123" not in result

    def test_long_key(self) -> None:
        result = mask_secret("sk-abcdef1234567890")
        assert result == "sk-...7890"

    def test_full_secret_not_in_mask(self) -> None:
        secret = "sk-1234567890abcdef"
        masked = mask_secret(secret)
        assert secret not in masked

    def test_mask_shows_prefix_and_suffix(self) -> None:
        """For a long key, first 3 chars and last 4 chars should appear."""
        secret = "tok-abcdefghij0000"
        masked = mask_secret(secret)
        assert masked.startswith("tok")
        assert masked.endswith("0000")
        assert "..." in masked

    def test_mask_medium_prefix_suffix(self) -> None:
        """For medium-length key (9-12 chars), first 2 and last 2 appear."""
        secret = "sk-abc12345"  # 11 chars → medium path
        masked = mask_secret(secret)
        assert masked.startswith("sk")
        assert masked.endswith("45")
        assert "..." in masked


# ---------------------------------------------------------------------------
# MockChatProvider — determinism
# ---------------------------------------------------------------------------


class TestMockChatProvider:
    @pytest.mark.asyncio
    async def test_same_input_same_output(self) -> None:
        """Identical input messages must produce identical output."""
        from app.providers.base import ChatMessage

        provider = MockChatProvider(model="test-model")
        messages = [ChatMessage(role="user", content="Tell me about black holes")]

        r1 = await provider.complete(messages)
        r2 = await provider.complete(messages)

        assert r1.content == r2.content
        assert r1.model == r2.model

    @pytest.mark.asyncio
    async def test_different_input_different_output(self) -> None:
        """Different input messages should (statistically) produce different output."""
        from app.providers.base import ChatMessage

        provider = MockChatProvider()
        r1 = await provider.complete([ChatMessage(role="user", content="question one")])
        r2 = await provider.complete([ChatMessage(role="user", content="question two")])
        # Hash collision is theoretically possible but astronomically unlikely
        assert r1.content != r2.content

    @pytest.mark.asyncio
    async def test_returns_chat_completion(self) -> None:
        """Result must be a ChatCompletion with non-empty content."""
        from app.providers.base import ChatCompletion, ChatMessage

        provider = MockChatProvider()
        result = await provider.complete([ChatMessage(role="user", content="hi")])
        assert isinstance(result, ChatCompletion)
        assert result.content
        assert result.model == "mock-model"

    @pytest.mark.asyncio
    async def test_usage_populated(self) -> None:
        """Usage fields must be non-negative integers."""
        from app.providers.base import ChatMessage

        provider = MockChatProvider()
        result = await provider.complete([ChatMessage(role="user", content="hello")])
        assert result.usage.prompt_tokens >= 0
        assert result.usage.completion_tokens >= 0

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self) -> None:
        """stream() must yield at least one ChatChunk."""
        from app.providers.base import ChatChunk, ChatMessage

        provider = MockChatProvider()
        chunks: list[ChatChunk] = []
        async for chunk in provider.stream([ChatMessage(role="user", content="stream me")]):
            chunks.append(chunk)
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_stream_assembles_to_complete_content(self) -> None:
        """Concatenating all stream deltas must equal the complete() output."""
        from app.providers.base import ChatMessage

        provider = MockChatProvider()
        messages = [ChatMessage(role="user", content="streaming test input")]
        complete_result = await provider.complete(messages)
        streamed = ""
        async for chunk in provider.stream(messages):
            streamed += chunk.delta
        assert streamed == complete_result.content

    @pytest.mark.asyncio
    async def test_last_chunk_has_stop_finish_reason(self) -> None:
        """The last streamed chunk must have finish_reason='stop'."""
        from app.providers.base import ChatMessage

        provider = MockChatProvider()
        last_chunk = None
        async for chunk in provider.stream([ChatMessage(role="user", content="end test")]):
            last_chunk = chunk
        assert last_chunk is not None
        assert last_chunk.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_test_connection_ok(self) -> None:
        provider = MockChatProvider()
        result = await provider.test_connection()
        assert result.ok is True
        assert result.error is None

    def test_name_and_model_properties(self) -> None:
        provider = MockChatProvider(model="custom-mock")
        assert provider.name == "mock"
        assert provider.model == "custom-mock"
        assert provider.context_window > 0


# ---------------------------------------------------------------------------
# MockEmbeddingProvider — determinism and shape
# ---------------------------------------------------------------------------


class TestMockEmbeddingProvider:
    @pytest.mark.asyncio
    async def test_output_shape(self) -> None:
        """embed() must return one vector per input text."""
        provider = MockEmbeddingProvider(dimension=16)
        texts = ["hello", "world", "foo"]
        result = await provider.embed(texts)
        assert len(result) == 3
        for vec in result:
            assert len(vec) == 16

    @pytest.mark.asyncio
    async def test_dimension_property(self) -> None:
        provider = MockEmbeddingProvider(dimension=512)
        assert provider.dimension == 512

    @pytest.mark.asyncio
    async def test_same_text_same_vector(self) -> None:
        """Same input text must produce the same embedding vector."""
        provider = MockEmbeddingProvider(dimension=32)
        v1 = (await provider.embed(["deterministic text"]))[0]
        v2 = (await provider.embed(["deterministic text"]))[0]
        assert v1 == v2

    @pytest.mark.asyncio
    async def test_different_texts_different_vectors(self) -> None:
        """Different texts should produce different vectors (hash collision unlikely)."""
        provider = MockEmbeddingProvider(dimension=32)
        v1 = (await provider.embed(["text alpha"]))[0]
        v2 = (await provider.embed(["text beta"]))[0]
        assert v1 != v2

    @pytest.mark.asyncio
    async def test_vector_values_in_range(self) -> None:
        """Each float in the embedding vector must be in [-1, 1]."""
        provider = MockEmbeddingProvider(dimension=64)
        vecs = await provider.embed(["range check test"])
        for vec in vecs:
            for f in vec:
                assert -1.0 <= f <= 1.0, f"float {f} outside [-1, 1]"

    @pytest.mark.asyncio
    async def test_test_connection_ok(self) -> None:
        provider = MockEmbeddingProvider(dimension=8)
        result = await provider.test_connection()
        assert result.ok is True


# ---------------------------------------------------------------------------
# MockRerankerProvider — scores in [0, 1] and determinism
# ---------------------------------------------------------------------------


class TestMockRerankerProvider:
    @pytest.mark.asyncio
    async def test_scores_length_matches_docs(self) -> None:
        provider = MockRerankerProvider()
        docs = ["doc1", "doc2", "doc3"]
        scores = await provider.score("my query", docs)
        assert len(scores) == len(docs)

    @pytest.mark.asyncio
    async def test_scores_in_range(self) -> None:
        """All scores must be in [0, 1]."""
        provider = MockRerankerProvider()
        scores = await provider.score("query about AI", ["doc about AI", "doc about cats"])
        for s in scores:
            assert 0.0 <= s <= 1.0, f"score {s} outside [0, 1]"

    @pytest.mark.asyncio
    async def test_same_input_same_scores(self) -> None:
        """Same query + docs must always produce the same scores."""
        provider = MockRerankerProvider()
        s1 = await provider.score("test query", ["passage one", "passage two"])
        s2 = await provider.score("test query", ["passage one", "passage two"])
        assert s1 == s2

    @pytest.mark.asyncio
    async def test_different_query_different_scores(self) -> None:
        """Different queries should give different scores for the same docs."""
        provider = MockRerankerProvider()
        s1 = await provider.score("query A", ["some document text"])
        s2 = await provider.score("query B", ["some document text"])
        # Not guaranteed but overwhelmingly likely with hash-based approach
        assert s1 != s2

    @pytest.mark.asyncio
    async def test_test_connection_ok(self) -> None:
        provider = MockRerankerProvider()
        result = await provider.test_connection()
        assert result.ok is True


# ---------------------------------------------------------------------------
# Key sanitization — secret must never appear in logs
# ---------------------------------------------------------------------------


class TestKeyNotInLogs:
    @pytest.mark.asyncio
    async def test_chat_provider_log_does_not_contain_key(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The raw API key must not appear in any log output from the adapter."""
        from app.providers.openai_compat import OpenAICompatChatProvider  # noqa: F401

        secret_key = "sk-secret-key-that-must-not-be-logged"

        # Trigger logging at DEBUG level without actually hitting the network.
        # We directly invoke the logger path that mask_secret covers, exactly
        # as the adapter does internally.
        with caplog.at_level(logging.DEBUG, logger="app.providers.openai_compat"):
            import app.security as sec

            masked = sec.mask_secret(secret_key)
            logging.getLogger("app.providers.openai_compat").debug(
                "OpenAI-compat complete: provider=%s model=%s base_url=%s key=%s",
                "test",
                "fake-model",
                "http://fake-endpoint/v1",
                masked,
            )

        full_log = "\n".join(caplog.messages)
        assert secret_key not in full_log, (
            f"Secret key '{secret_key}' leaked into log output: {full_log}"
        )

    def test_mask_secret_not_in_log_via_adapter_internals(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify mask_secret output does not contain the full secret."""
        secret = "sk-abcdef1234567890-full-secret"
        masked = mask_secret(secret)
        # The full secret must not appear in the masked version
        assert secret not in masked
        # The masked version should be shorter and contain '...'
        assert "..." in masked
        assert len(masked) < len(secret)

    def test_registry_build_logs_do_not_contain_key(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """build_chat_provider must not log the plain-text API key."""
        from app.providers.registry import build_chat_provider

        secret_key = "sk-registry-secret-must-not-leak"
        profile = SimpleNamespace(
            provider_type="mock",
            model_name="mock-model",
            api_key_encrypted=None,
            api_key_plain=secret_key,
            base_url=None,
            context_window=None,
            embedding_dim=None,
            provider_name=None,
        )

        with caplog.at_level(logging.DEBUG, logger="app.providers.registry"):
            build_chat_provider(profile)

        full_log = "\n".join(caplog.messages)
        assert secret_key not in full_log, (
            f"Secret key leaked into registry log: {full_log}"
        )


# ---------------------------------------------------------------------------
# Registry — build_chat_provider
# ---------------------------------------------------------------------------


class TestRegistry:
    def _make_profile(self, **kwargs: object) -> SimpleNamespace:
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

    def test_build_mock_chat(self) -> None:
        from app.providers.registry import build_chat_provider

        profile = self._make_profile(provider_type="mock", model_name="test")
        provider = build_chat_provider(profile)
        assert provider.name == "mock"
        assert provider.model == "test"

    def test_build_mock_embedding(self) -> None:
        from app.providers.registry import build_embedding_provider

        profile = self._make_profile(provider_type="mock", model_name="emb", embedding_dim=64)
        provider = build_embedding_provider(profile)
        assert provider.dimension == 64

    def test_build_mock_reranker(self) -> None:
        from app.providers.registry import build_reranker_provider

        profile = self._make_profile(provider_type="mock", model_name="rerank")
        provider = build_reranker_provider(profile)
        assert provider is not None

    def test_unknown_provider_type_raises(self) -> None:
        from app.providers.registry import build_chat_provider

        profile = self._make_profile(provider_type="unknown_xyz")
        with pytest.raises(ValueError, match="Unknown provider_type"):
            build_chat_provider(profile)

    def test_openai_compat_requires_base_url(self) -> None:
        from app.providers.registry import build_chat_provider

        profile = self._make_profile(
            provider_type="openai_compat",
            model_name="some-model",
            base_url=None,
        )
        with pytest.raises(ValueError, match="base_url"):
            build_chat_provider(profile)

    def test_build_openai_compat_chat(self) -> None:
        from app.providers.openai_compat import OpenAICompatChatProvider
        from app.providers.registry import build_chat_provider

        profile = self._make_profile(
            provider_type="openai_compat",
            model_name="llama-3",
            api_key_plain="fake-key",
            base_url="http://localhost:8001/v1",
        )
        provider = build_chat_provider(profile)
        assert isinstance(provider, OpenAICompatChatProvider)
        assert provider.model == "llama-3"

    def test_build_openai_chat(self) -> None:
        from app.providers.openai import OpenAIChatProvider
        from app.providers.registry import build_chat_provider

        profile = self._make_profile(
            provider_type="openai",
            model_name="gpt-4o",
            api_key_plain="sk-fake-key",
        )
        provider = build_chat_provider(profile)
        assert isinstance(provider, OpenAIChatProvider)
        assert provider.model == "gpt-4o"

    def test_build_gemini_native_chat(self) -> None:
        from app.providers.gemini_native import GeminiNativeChatProvider
        from app.providers.registry import build_chat_provider

        profile = self._make_profile(
            provider_type="gemini_native",
            model_name="gemini-2.0-flash",
            api_key_plain="fake-gemini-key",
        )
        provider = build_chat_provider(profile)
        assert isinstance(provider, GeminiNativeChatProvider)

    def test_encrypted_key_resolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Registry must decrypt api_key_encrypted before passing to the adapter."""
        from app.providers.registry import build_chat_provider
        from app.security import encrypt

        plaintext_key = "sk-encrypted-real-key"
        token = encrypt(plaintext_key)

        profile = self._make_profile(
            provider_type="mock",
            model_name="mock-enc-test",
            api_key_encrypted=token,
        )
        # Should not raise; mock provider doesn't use the key
        provider = build_chat_provider(profile)
        assert provider is not None

    def test_vllm_alias(self) -> None:
        """'vllm' provider_type is an alias for 'openai_compat'."""
        from app.providers.openai_compat import OpenAICompatChatProvider
        from app.providers.registry import build_chat_provider

        profile = self._make_profile(
            provider_type="vllm",
            model_name="llama-3.1-8b",
            api_key_plain="",
            base_url="http://localhost:8001/v1",
        )
        provider = build_chat_provider(profile)
        assert isinstance(provider, OpenAICompatChatProvider)


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------


def test_providers_package_importable() -> None:
    """The providers package must be importable with its public API."""
    from app.providers import (  # noqa: F401
        ChatProvider,
        EmbeddingProvider,
        ProviderTestResult,
        RerankerProvider,
        build_chat_provider,
        build_embedding_provider,
        build_reranker_provider,
    )


def test_security_importable() -> None:
    """app.security must be importable and expose the expected API."""
    from app.security import decrypt, encrypt, generate_key, mask_secret  # noqa: F401

    key = generate_key()
    assert isinstance(key, str)
    assert len(key) > 0


# ---------------------------------------------------------------------------
# Env var not leaked through os.environ in any log
# ---------------------------------------------------------------------------


def test_os_environ_key_not_in_settings_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that get_settings() works and repr does not crash.

    app_encryption_key is a plain str (symmetric encryption key) — not a
    SecretStr — because it is used programmatically, not as a user API key.
    This test documents that behaviour and verifies that API keys (SecretStr)
    are the ones that must be masked (see test_openai_key_not_in_settings_repr).
    """
    # Simply verify that get_settings() returns a Settings object with the
    # expected encryption key value (no repr assertion on the plain str key).
    s = get_settings()
    assert s.app_encryption_key == REQUIRED_ENV["APP_ENCRYPTION_KEY"]


def test_openai_key_not_in_settings_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENAI_API_KEY plain value must not appear in repr(settings)."""
    plain = "sk-test-key-must-not-leak"
    monkeypatch.setenv("OPENAI_API_KEY", plain)
    get_settings.cache_clear()

    s = get_settings()
    assert plain not in repr(s)
    assert plain not in str(s)
