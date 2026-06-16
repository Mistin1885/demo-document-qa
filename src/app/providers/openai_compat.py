"""OpenAI-compatible endpoint adapter (vLLM, Gemini OpenAI-compat, self-hosted).

Any server that exposes the OpenAI ``/v1/chat/completions`` and
``/v1/embeddings`` REST API can be driven by this adapter.  The only
difference from ``openai.py`` is that ``base_url`` is configurable.

Typical uses:
- vLLM server at ``http://localhost:8001/v1``
- Gemini OpenAI-compatible endpoint at ``https://generativelanguage.googleapis.com/v1beta/openai``
- Any self-hosted OpenAI-compatible API

Security
--------
Same as ``openai.py``: ``mask_secret`` is used for all log messages.
The API key is accepted as a plain string (decrypted by the registry before
being passed here).
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
    RerankerProvider,
    Usage,
)
from app.security import mask_secret

logger = logging.getLogger(__name__)

_DEFAULT_CONTEXT_WINDOW = 32_768


def _to_oai_messages(messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
    """Convert our ``ChatMessage`` list to the OpenAI typed message format."""
    result: list[ChatCompletionMessageParam] = []
    for m in messages:
        result.append({"role": m.role, "content": m.content})  # type: ignore[arg-type,misc]
    return result


class OpenAICompatChatProvider(ChatProvider):
    """Chat provider for any OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Parameters
    ----------
    api_key:
        Plain-text API key (pass ``"none"`` or empty string for local servers
        that don't require authentication).
    base_url:
        Base URL of the OpenAI-compatible server, e.g.
        ``"http://localhost:8001/v1"``.
    model:
        Model name as recognized by the remote server.
    context_window:
        Override the inferred context window.
    provider_name:
        Human-readable name for this provider instance (used in ``name``).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        context_window: int = _DEFAULT_CONTEXT_WINDOW,
        provider_name: str = "openai_compat",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._context_window = context_window
        self._provider_name = provider_name
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI(
                api_key=self._api_key or "none",
                base_url=self._base_url,
            )
        return self._client

    # ------------------------------------------------------------------
    # ChatProvider interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._provider_name

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
            "OpenAI-compat complete: provider=%s model=%s base_url=%s key=%s",
            self._provider_name,
            self._model,
            self._base_url,
            mask_secret(self._api_key),
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
            "OpenAI-compat stream: provider=%s model=%s base_url=%s key=%s",
            self._provider_name,
            self._model,
            self._base_url,
            mask_secret(self._api_key),
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
            logger.warning(
                "OpenAI-compat connection test failed (provider=%s): %s",
                self._provider_name,
                exc,
            )
            return ProviderTestResult(ok=False, latency_ms=latency_ms, error=str(exc))


class OpenAICompatEmbeddingProvider(EmbeddingProvider):
    """Embedding provider for any OpenAI-compatible ``/v1/embeddings`` endpoint.

    Parameters
    ----------
    api_key:
        Plain-text API key.
    base_url:
        Base URL of the server (e.g. ``"http://localhost:8001/v1"``).
    model:
        Embedding model name.
    dimension:
        Expected vector dimension.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        dimension: int = 1024,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._dimension = dimension
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI(
                api_key=self._api_key or "none",
                base_url=self._base_url,
            )
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        logger.debug(
            "OpenAI-compat embed: model=%s base_url=%s key=%s n=%d",
            self._model,
            self._base_url,
            mask_secret(self._api_key),
            len(texts),
        )
        response = await client.embeddings.create(
            model=self._model,
            input=texts,
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
            return ProviderTestResult(ok=False, latency_ms=latency_ms, error=str(exc))


class OpenAICompatRerankerProvider(RerankerProvider):
    """Reranker backed by an OpenAI-compatible scoring endpoint.

    This adapter calls the remote model to score each (query, document) pair
    by asking it for a relevance judgment via a structured prompt.  The score
    is derived from the model's text output.

    Note: Many open-source cross-encoders expose themselves through an
    OpenAI-compatible API (e.g., via TEI or vLLM).  This adapter handles that
    case by constructing a concise relevance-scoring prompt.

    Implementation note
    -------------------
    The scoring prompt asks the model to respond with a single float in [0, 1]
    representing the relevance.  We parse the completion text directly.  If
    parsing fails we default to 0.0 and log a warning (without logging the
    document contents verbatim if they are long).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        provider_name: str = "openai_compat_reranker",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._provider_name = provider_name
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI(
                api_key=self._api_key or "none",
                base_url=self._base_url,
            )
        return self._client

    async def score(self, query: str, docs: list[str]) -> list[float]:
        client = self._get_client()
        scores: list[float] = []
        for doc in docs:
            prompt = (
                "You are a relevance judge.  Given a QUERY and a DOCUMENT, respond with "
                "a single float in [0, 1] representing how relevant the document is to "
                "the query (1=highly relevant, 0=not relevant).  Output ONLY the float.\n\n"
                f"QUERY: {query}\n\nDOCUMENT: {doc[:2000]}"
            )
            try:
                kwargs: dict = {
                    "model": self._model,
                    "messages": _to_oai_messages(
                        [ChatMessage(role="user", content=prompt)]
                    ),
                    "max_tokens": 8,
                    "temperature": 0.0,
                }
                response = await client.chat.completions.create(**kwargs)
                raw = (response.choices[0].message.content or "0").strip()
                score = float(raw)
                score = max(0.0, min(1.0, score))
            except Exception as exc:
                logger.warning(
                    "Reranker score failed (provider=%s model=%s): %s",
                    self._provider_name,
                    self._model,
                    exc,
                )
                score = 0.0
            scores.append(score)
        return scores

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
            await client.chat.completions.create(**kwargs)
            latency_ms = int((time.monotonic() - start) * 1000)
            return ProviderTestResult(ok=True, model=self._model, latency_ms=latency_ms)
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            return ProviderTestResult(ok=False, latency_ms=latency_ms, error=str(exc))
