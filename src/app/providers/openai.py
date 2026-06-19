"""OpenAI official SDK adapter for Chat and Embedding providers.

Uses ``openai.AsyncOpenAI`` (the official async client).  The client is
constructed lazily on first use so that import-time failures do not occur
when the key is not yet set (e.g., during tests).

Security
--------
- The API key is accepted as a plain string (already decrypted by the registry
  or test harness).  It is NEVER written to any log.
- ``mask_secret`` is used for all log statements that reference the key or
  any credential-adjacent information.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import cast

import openai
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

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

# Default context windows for common OpenAI models.
_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
}
_DEFAULT_CONTEXT_WINDOW = 16_385


def _to_oai_messages(messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
    """Convert our ``ChatMessage`` list to OpenAI's typed message format."""
    result: list[ChatCompletionMessageParam] = []
    for m in messages:
        # We use type: ignore here because the dict literal satisfies the
        # TypedDict union at runtime but mypy can't resolve the ambiguity.
        result.append({"role": m.role, "content": m.content})  # type: ignore[arg-type,misc]
    return result


class OpenAIChatProvider(ChatProvider):
    """Chat provider backed by the official OpenAI ``chat.completions`` API.

    Parameters
    ----------
    api_key:
        Plain-text API key (must already be decrypted by the caller).
    model:
        Model name, e.g. ``"gpt-4o"``.
    context_window:
        Override the inferred context window (tokens).  Defaults to a
        per-model lookup table with a safe fallback.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        context_window: int | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._context_window = context_window or _CONTEXT_WINDOWS.get(
            model, _DEFAULT_CONTEXT_WINDOW
        )
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=self._api_key)
        return self._client

    # ------------------------------------------------------------------
    # ChatProvider interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "openai"

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
            "OpenAI complete: model=%s key=%s", self._model, mask_secret(self._api_key)
        )
        kwargs: dict = {
            "model": self._model,
            "messages": _to_oai_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop is not None:
            kwargs["stop"] = stop
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage_data = response.usage
        return ChatCompletion(
            content=choice.message.content or "",
            usage=Usage(
                prompt_tokens=usage_data.prompt_tokens if usage_data else 0,
                completion_tokens=usage_data.completion_tokens if usage_data else 0,
            ),
            model=response.model,
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
            "OpenAI stream: model=%s key=%s", self._model, mask_secret(self._api_key)
        )
        kwargs: dict = {
            "model": self._model,
            "messages": _to_oai_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if stop is not None:
            kwargs["stop"] = stop
        raw_stream: AsyncStream[ChatCompletionChunk] = cast(
            AsyncStream[ChatCompletionChunk],
            await client.chat.completions.create(**kwargs),
        )
        async for chunk in raw_stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue
            delta_content = choice.delta.content or ""
            finish_reason = choice.finish_reason
            yield ChatChunk(delta=delta_content, finish_reason=finish_reason)

    async def test_connection(self) -> ProviderTestResult:
        start = time.monotonic()
        try:
            client = self._get_client()
            kwargs: dict = {
                "model": self._model,
                "messages": _to_oai_messages([ChatMessage(role="user", content="ping")]),
                "max_tokens": 1,
                "temperature": 0.0,
            }
            response = await client.chat.completions.create(**kwargs)
            latency_ms = int((time.monotonic() - start) * 1000)
            return ProviderTestResult(ok=True, model=response.model, latency_ms=latency_ms)
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.warning("OpenAI connection test failed: %s", exc)
            return ProviderTestResult(ok=False, latency_ms=latency_ms, error=str(exc))


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by ``openai.embeddings.create``.

    Parameters
    ----------
    api_key:
        Plain-text API key.
    model:
        Embedding model name (e.g., ``"text-embedding-3-large"``).
    dimension:
        Expected output dimension.  Defaults to 384 (matches default
        ``EMBEDDING_DIM`` in config).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimension: int = 384,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimension = dimension
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=self._api_key)
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        logger.debug(
            "OpenAI embed: model=%s key=%s n=%d",
            self._model,
            mask_secret(self._api_key),
            len(texts),
        )
        response = await client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimension,
        )
        return [item.embedding for item in response.data]

    async def test_connection(self) -> ProviderTestResult:
        start = time.monotonic()
        try:
            await self.embed(["test"])
            latency_ms = int((time.monotonic() - start) * 1000)
            return ProviderTestResult(ok=True, model=self._model, latency_ms=latency_ms)
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.warning("OpenAI embedding test failed: %s", exc)
            return ProviderTestResult(ok=False, latency_ms=latency_ms, error=str(exc))
