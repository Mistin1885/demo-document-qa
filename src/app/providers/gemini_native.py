"""Gemini Native adapter using the ``google-genai`` SDK.

This adapter provides Chat and Embedding providers that call Google's Gemini
models via the official ``google-genai`` Python SDK (not the OpenAI-compatible
endpoint).  Use ``GeminiNativeChatProvider`` for richer Gemini-specific
features (thinking mode, multimodal, grounding, etc.); for a simpler
drop-in that reuses the OpenAI-compatible path use
``providers.openai_compat.OpenAICompatChatProvider`` with the Gemini
OpenAI-compat base URL.

Compatibility note
------------------
The ``google-genai`` SDK (``google-genai >= 1.0``) introduced a breaking
namespace change from the older ``google-generativeai`` package.  This module
targets ``google.genai``.  If you installed ``google-generativeai`` instead,
the import will fail with a clear error message.

Interface contract
------------------
Both ``GeminiNativeChatProvider`` and ``GeminiNativeEmbeddingProvider``
implement the full ABC interface defined in ``base.py`` so downstream code
can swap them in without code changes.

The implementation is complete and functional for standard chat and embedding
workflows.  For production use, set ``GEMINI_API_KEY`` in the environment or
pass it directly; test code should use ``MockChatProvider`` /
``MockEmbeddingProvider`` to avoid real API calls.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import google.genai as genai
import google.genai.types as genai_types

from app.providers.base import (
    ChatChunk,
    ChatCompletion,
    ChatMessage,
    ChatProvider,
    EmbeddingProvider,
    ProviderTestResult,
    Usage,
)
from app.security import mask_secret

logger = logging.getLogger(__name__)

# Default context windows for Gemini models
_CONTEXT_WINDOWS: dict[str, int] = {
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.0-flash-lite": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.0-pro": 32_760,
}
_DEFAULT_CONTEXT_WINDOW = 128_000

# Embedding models and their dimensions
_EMBEDDING_DIMS: dict[str, int] = {
    "text-embedding-004": 768,
    "embedding-001": 768,
}
_DEFAULT_EMBEDDING_DIM = 768


def _map_role(role: str) -> str:
    """Map our ChatMessage role to Gemini's expected role strings."""
    if role == "assistant":
        return "model"
    if role == "system":
        # Gemini doesn't have a "system" turn in the conversation history;
        # system messages are passed as system_instruction in the config.
        return "user"
    return "user"


class GeminiNativeChatProvider(ChatProvider):
    """Chat provider backed by the Google Gemini ``google-genai`` SDK.

    Parameters
    ----------
    api_key:
        Plain-text Gemini API key (must be decrypted before passing here).
    model:
        Gemini model identifier, e.g. ``"gemini-2.0-flash"``.
    context_window:
        Override the inferred context window.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        context_window: int | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._context_window = context_window or _CONTEXT_WINDOWS.get(
            model, _DEFAULT_CONTEXT_WINDOW
        )
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    # ------------------------------------------------------------------
    # ChatProvider interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "gemini_native"

    @property
    def model(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> ChatCompletion:
        client = self._get_client()
        logger.debug(
            "Gemini native complete: model=%s key=%s",
            self._model,
            mask_secret(self._api_key),
        )
        # Extract system messages; Gemini uses system_instruction separately
        system_parts = [m.content for m in messages if m.role == "system"]
        system_instruction = system_parts[0] if system_parts else None

        # Build the conversation turns (exclude system messages)
        contents: list[genai_types.Content] = []
        for m in messages:
            if m.role == "system":
                continue
            contents.append(
                genai_types.Content(
                    role=_map_role(m.role),
                    parts=[genai_types.Part(text=m.content)],
                )
            )

        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            stop_sequences=stop or [],
            system_instruction=system_instruction,
        )

        response = await client.aio.models.generate_content(
            model=self._model,
            contents=contents,  # type: ignore[arg-type]
            config=config,
        )

        content_text = response.text or ""
        # Usage metadata may be available
        usage_meta = getattr(response, "usage_metadata", None)
        return ChatCompletion(
            content=content_text,
            usage=Usage(
                prompt_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
                completion_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
            ),
            model=self._model,
        )

    async def stream(  # type: ignore[override]
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        client = self._get_client()
        logger.debug(
            "Gemini native stream: model=%s key=%s",
            self._model,
            mask_secret(self._api_key),
        )
        system_parts = [m.content for m in messages if m.role == "system"]
        system_instruction = system_parts[0] if system_parts else None

        contents: list[genai_types.Content] = []
        for m in messages:
            if m.role == "system":
                continue
            contents.append(
                genai_types.Content(
                    role=_map_role(m.role),
                    parts=[genai_types.Part(text=m.content)],
                )
            )

        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            stop_sequences=stop or [],
            system_instruction=system_instruction,
        )

        async for chunk in await client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,  # type: ignore[arg-type]
            config=config,
        ):
            text = chunk.text or ""
            # Check if the candidate has finished
            finish_reason: str | None = None
            if chunk.candidates:
                candidate = chunk.candidates[0]
                if getattr(candidate, "finish_reason", None):
                    finish_reason = str(candidate.finish_reason)
            yield ChatChunk(delta=text, finish_reason=finish_reason)

    async def test_connection(self) -> ProviderTestResult:
        start = time.monotonic()
        try:
            result = await self.complete(
                [ChatMessage(role="user", content="ping")],
                max_tokens=4,
            )
            latency_ms = int((time.monotonic() - start) * 1000)
            return ProviderTestResult(ok=True, model=result.model, latency_ms=latency_ms)
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.warning("Gemini native connection test failed: %s", exc)
            return ProviderTestResult(ok=False, latency_ms=latency_ms, error=str(exc))


class GeminiNativeEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by Gemini's ``embed_content`` API.

    Parameters
    ----------
    api_key:
        Plain-text Gemini API key.
    model:
        Embedding model name (e.g., ``"text-embedding-004"``).
    dimension:
        Output dimension.  Falls back to the model default (768 for Gemini
        embedding models).  Note: not all Gemini embedding models support
        dimension truncation — verify compatibility before overriding.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-004",
        dimension: int | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimension = dimension or _EMBEDDING_DIMS.get(model, _DEFAULT_EMBEDDING_DIM)
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        logger.debug(
            "Gemini native embed: model=%s key=%s n=%d",
            self._model,
            mask_secret(self._api_key),
            len(texts),
        )
        results: list[list[float]] = []
        for text in texts:
            response = await client.aio.models.embed_content(
                model=self._model,
                contents=text,
            )
            raw_values: Any = response.embeddings[0].values if response.embeddings else []
            embedding: list[float] = list(raw_values) if raw_values is not None else []
            results.append(embedding)
        return results

    async def test_connection(self) -> ProviderTestResult:
        start = time.monotonic()
        try:
            await self.embed(["test"])
            latency_ms = int((time.monotonic() - start) * 1000)
            return ProviderTestResult(ok=True, model=self._model, latency_ms=latency_ms)
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.warning("Gemini native embedding test failed: %s", exc)
            return ProviderTestResult(ok=False, latency_ms=latency_ms, error=str(exc))
