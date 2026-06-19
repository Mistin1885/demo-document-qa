"""Provider registry — factory functions that build concrete adapters from profiles.

The registry is the **single translation point** between persisted
``ProviderProfile`` data (ORM or any dict-like) and the abstract provider
interfaces defined in ``base.py``.

Design goals
------------
- Decoupled from the ORM: the input is a ``ProviderProfileLike`` Protocol, so
  the registry can be used in tests without importing SQLAlchemy models.
- Key decryption happens here (using ``app.security.decrypt``), so concrete
  adapter classes only ever receive a plain-text string.
- Clear error messages when a required field is missing or the provider type
  is unrecognised.

Supported provider types (``provider_type`` field)
----------------------------------------------------
+-----------------------------+--------------------------------------------+
| ``"openai"``                | Official OpenAI SDK (``openai.py``)        |
+-----------------------------+--------------------------------------------+
| ``"gemini_native"``         | Google Gemini SDK (``gemini_native.py``)   |
+-----------------------------+--------------------------------------------+
| ``"openai_compat"``         | Generic OpenAI-compatible (vLLM, Gemini    |
|                             | OpenAI-compat, self-hosted)                |
+-----------------------------+--------------------------------------------+
| ``"vllm"``                  | Alias for ``"openai_compat"``              |
+-----------------------------+--------------------------------------------+
| ``"mock"``                  | Deterministic mock (tests only)            |
+-----------------------------+--------------------------------------------+

Profile shape
-------------
The registry accepts any object or dict-like that satisfies
``ProviderProfileLike``.  The ORM model (Phase 3) should expose all of these
attributes.  For tests, a simple ``@dataclass`` or ``SimpleNamespace`` works.

Required fields
~~~~~~~~~~~~~~~~
- ``provider_type: str``  — one of the supported types above.
- ``model_name: str``     — e.g. ``"gpt-4o"``, ``"gemini-2.0-flash"``.

Optional fields (fallback to ``None`` / sensible defaults)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``api_key_encrypted: bytes | None``  — Fernet-encrypted API key bytes.
  If ``None``, the plain ``api_key_plain`` field is tried (for tests/local use).
- ``api_key_plain: str | None``        — Unencrypted key (for tests).
- ``base_url: str | None``             — Required for ``openai_compat``/``vllm``.
- ``context_window: int | None``       — Override default context window.
- ``embedding_dim: int | None``        — For embedding providers.
- ``provider_name: str | None``        — Human label (defaults to provider_type).
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from app.providers.base import ChatProvider, EmbeddingProvider, RerankerProvider
from app.security import decrypt, mask_secret

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile Protocol (structural subtyping — no ORM import needed)
# ---------------------------------------------------------------------------


@runtime_checkable
class ProviderProfileLike(Protocol):
    """Structural protocol for provider profile objects.

    Any object with these attributes satisfies this protocol.  The ORM
    ``ProviderProfile`` model must implement all of these.
    """

    provider_type: str
    model_name: str
    api_key_encrypted: bytes | None
    api_key_plain: str | None
    base_url: str | None
    context_window: int | None
    embedding_dim: int | None
    provider_name: str | None


# ---------------------------------------------------------------------------
# Internal key-resolution helper
# ---------------------------------------------------------------------------


def _resolve_api_key(profile: ProviderProfileLike) -> str:
    """Extract and (if necessary) decrypt the API key from *profile*.

    Resolution order:
    1. ``api_key_encrypted`` → decrypt with Fernet.
    2. ``api_key_plain``     → use directly (tests / local-only servers).
    3. Empty string          → for servers that require no key (local vLLM).

    The resolved key is NEVER logged.
    """
    if profile.api_key_encrypted is not None:
        try:
            key = decrypt(profile.api_key_encrypted)
            logger.debug("Resolved encrypted API key: %s", mask_secret(key))
            return key
        except Exception as exc:
            raise ValueError(
                f"Failed to decrypt api_key_encrypted for provider "
                f"'{profile.provider_type}': {exc}"
            ) from exc
    if profile.api_key_plain is not None:
        logger.debug(
            "Using plain-text API key: %s", mask_secret(profile.api_key_plain)
        )
        return profile.api_key_plain
    return ""


# ---------------------------------------------------------------------------
# Chat provider factory
# ---------------------------------------------------------------------------


def build_chat_provider(profile: ProviderProfileLike) -> ChatProvider:
    """Build and return a :class:`ChatProvider` from *profile*.

    Parameters
    ----------
    profile:
        Any object satisfying :class:`ProviderProfileLike`.

    Returns
    -------
    ChatProvider

    Raises
    ------
    ValueError
        If ``provider_type`` is unknown or required fields are missing.
    """
    ptype = profile.provider_type.lower()
    api_key = _resolve_api_key(profile)
    model = profile.model_name

    if ptype == "openai":
        from app.providers.openai import OpenAIChatProvider

        return OpenAIChatProvider(
            api_key=api_key,
            model=model,
            context_window=profile.context_window,
        )

    if ptype == "gemini_native":
        from app.providers.gemini_native import GeminiNativeChatProvider

        return GeminiNativeChatProvider(
            api_key=api_key,
            model=model,
            context_window=profile.context_window,
        )

    if ptype in ("openai_compat", "vllm"):
        base_url = profile.base_url
        if not base_url:
            raise ValueError(
                f"'base_url' is required for provider_type='{ptype}' "
                f"(model='{model}')"
            )
        from app.providers.openai_compat import OpenAICompatChatProvider

        return OpenAICompatChatProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            context_window=profile.context_window or 32_768,
            provider_name=profile.provider_name or ptype,
        )

    if ptype == "mock":
        from app.providers.mock import MockChatProvider

        return MockChatProvider(model=model)

    raise ValueError(
        f"Unknown provider_type for chat: '{ptype}'.  "
        f"Supported: openai, gemini_native, openai_compat, vllm, mock."
    )


# ---------------------------------------------------------------------------
# Embedding provider factory
# ---------------------------------------------------------------------------


def build_embedding_provider(profile: ProviderProfileLike) -> EmbeddingProvider:
    """Build and return an :class:`EmbeddingProvider` from *profile*.

    Parameters
    ----------
    profile:
        Any object satisfying :class:`ProviderProfileLike`.

    Returns
    -------
    EmbeddingProvider

    Raises
    ------
    ValueError
        If ``provider_type`` is unknown or required fields are missing.
    """
    ptype = profile.provider_type.lower()
    api_key = _resolve_api_key(profile)
    model = profile.model_name
    dim = profile.embedding_dim or 384

    if ptype == "openai":
        from app.providers.openai import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(api_key=api_key, model=model, dimension=dim)

    if ptype == "gemini_native":
        from app.providers.gemini_native import GeminiNativeEmbeddingProvider

        return GeminiNativeEmbeddingProvider(api_key=api_key, model=model, dimension=dim)

    if ptype in ("openai_compat", "vllm"):
        base_url = profile.base_url
        if not base_url:
            raise ValueError(
                f"'base_url' is required for provider_type='{ptype}' "
                f"(model='{model}')"
            )
        from app.providers.openai_compat import OpenAICompatEmbeddingProvider

        return OpenAICompatEmbeddingProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            dimension=dim,
        )

    if ptype == "mock":
        from app.providers.mock import MockEmbeddingProvider

        return MockEmbeddingProvider(dimension=dim)

    raise ValueError(
        f"Unknown provider_type for embedding: '{ptype}'.  "
        f"Supported: openai, gemini_native, openai_compat, vllm, mock."
    )


# ---------------------------------------------------------------------------
# Reranker provider factory
# ---------------------------------------------------------------------------


def build_reranker_provider(profile: ProviderProfileLike) -> RerankerProvider:
    """Build and return a :class:`RerankerProvider` from *profile*.

    Parameters
    ----------
    profile:
        Any object satisfying :class:`ProviderProfileLike`.

    Returns
    -------
    RerankerProvider

    Raises
    ------
    ValueError
        If ``provider_type`` is unknown or required fields are missing.
    """
    ptype = profile.provider_type.lower()
    api_key = _resolve_api_key(profile)
    model = profile.model_name

    if ptype in ("openai_compat", "vllm", "openai"):
        # Reranker is always an OpenAI-compatible scoring endpoint.
        # For "openai" type, base_url may be None (will use default OpenAI URL)
        # but cross-encoder rerankers typically need a custom base_url.
        base_url = profile.base_url or "https://api.openai.com/v1"
        from app.providers.openai_compat import OpenAICompatRerankerProvider

        return OpenAICompatRerankerProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider_name=profile.provider_name or ptype,
        )

    if ptype == "mock":
        from app.providers.mock import MockRerankerProvider

        return MockRerankerProvider()

    raise ValueError(
        f"Unknown provider_type for reranker: '{ptype}'.  "
        f"Supported: openai, openai_compat, vllm, mock."
    )
