"""Deterministic mock providers for testing.

These providers NEVER make network calls.  Their outputs are fully
deterministic for a given input (derived from SHA-256 hash), so tests
can assert on specific values without involving any paid API.

Usage in tests
--------------
    from app.providers.mock import MockChatProvider, MockEmbeddingProvider, MockRerankerProvider

    chat = MockChatProvider(model="test-model")
    resp = await chat.complete([ChatMessage(role="user", content="hello")])
    assert resp.content  # deterministic, non-empty string

    emb = MockEmbeddingProvider(dimension=8)
    vecs = await emb.embed(["hello", "world"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 8

    reranker = MockRerankerProvider()
    scores = await reranker.score("query", ["doc1", "doc2"])
    assert all(0.0 <= s <= 1.0 for s in scores)
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import AsyncIterator

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

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(value: str) -> bytes:
    """Return the 32-byte SHA-256 digest of *value* encoded as UTF-8."""
    return hashlib.sha256(value.encode()).digest()


def _hash_to_int(value: str) -> int:
    """Return an unsigned 64-bit integer from the first 8 bytes of sha256(value)."""
    digest = _sha256_bytes(value)
    return struct.unpack(">Q", digest[:8])[0]


def _bytes_to_floats(data: bytes, count: int) -> list[float]:
    """Convert *count* floats (each using 4 digest bytes, cycling) from *data*.

    Each float is in the range [-1, 1], derived from interpreting a uint32 as
    a fraction of its maximum value then shifting.

    Parameters
    ----------
    data:
        Source bytes (will be repeated / cycled if shorter than needed).
    count:
        Number of floats to produce.
    """
    floats: list[float] = []
    for i in range(count):
        # Pick 4 bytes (cycle through data if needed)
        start = (i * 4) % len(data)
        chunk = data[start : start + 4]
        # If wrapping around the end, concatenate with the beginning
        if len(chunk) < 4:
            chunk = chunk + data[: 4 - len(chunk)]
        (uint_val,) = struct.unpack(">I", chunk)
        # Map [0, 2^32) → [-1, 1]
        floats.append((uint_val / 2_147_483_647.5) - 1.0)
    return floats


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(y * y for y in b) ** 0.5
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# MockChatProvider
# ---------------------------------------------------------------------------


class MockChatProvider(ChatProvider):
    """Deterministic mock chat provider.

    The response content is derived from the SHA-256 hash of the serialized
    input messages, ensuring the same input always produces the same output
    without any network call.

    Parameters
    ----------
    model:
        Model name string to report (default: ``"mock-model"``).
    context_window:
        Context window size to report (default: 8192).
    """

    def __init__(self, model: str = "mock-model", context_window: int = 8_192) -> None:
        self._model = model
        self._context_window = context_window

    # ------------------------------------------------------------------
    # ChatProvider interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mock"

    @property
    def model(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window

    def _compute_response(self, messages: list[ChatMessage]) -> str:
        """Derive a deterministic response string from *messages*."""
        # Serialize messages into a single stable string
        serialized = "|".join(f"{m.role}:{m.content}" for m in messages)
        hash_int = _hash_to_int(serialized) % 10_000
        return f"[mock {self._model}] response-{hash_int}"

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> ChatCompletion:
        content = self._compute_response(messages)
        return ChatCompletion(
            content=content,
            usage=Usage(prompt_tokens=10, completion_tokens=len(content.split())),
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
        content = self._compute_response(messages)
        # Yield word-by-word to simulate streaming
        words = content.split()
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            yield ChatChunk(
                delta=word if i == 0 else f" {word}",
                finish_reason="stop" if is_last else None,
            )

    async def test_connection(self) -> ProviderTestResult:
        return ProviderTestResult(ok=True, model=self._model, latency_ms=0)


# ---------------------------------------------------------------------------
# MockEmbeddingProvider
# ---------------------------------------------------------------------------


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic mock embedding provider.

    Each text produces a *dimension*-dimensional float vector derived from
    SHA-256(text).  The same text always produces the same vector.  No
    network calls are made.

    Parameters
    ----------
    dimension:
        Dimensionality of the embedding vectors.
    """

    def __init__(self, dimension: int = 1024) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            digest = _sha256_bytes(text)
            vector = _bytes_to_floats(digest, self._dimension)
            results.append(vector)
        return results

    async def test_connection(self) -> ProviderTestResult:
        return ProviderTestResult(ok=True, model="mock-embedding", latency_ms=0)


# ---------------------------------------------------------------------------
# MockRerankerProvider
# ---------------------------------------------------------------------------


class MockRerankerProvider(RerankerProvider):
    """Deterministic mock reranker provider.

    Relevance is simulated by computing cosine similarity between the
    hash-derived embeddings of the query and each document, then mapping the
    result to [0, 1].

    The score is always in [0, 1] and is deterministic for a given
    (query, doc) pair.

    Parameters
    ----------
    embedding_dim:
        Internal embedding dimension used for hash-derived vectors.
        Does not affect the output score range, only internal computation.
    """

    def __init__(self, embedding_dim: int = 64) -> None:
        self._embedding_dim = embedding_dim

    def _embed(self, text: str) -> list[float]:
        digest = _sha256_bytes(text)
        return _bytes_to_floats(digest, self._embedding_dim)

    async def score(self, query: str, docs: list[str]) -> list[float]:
        query_vec = self._embed(query)
        scores: list[float] = []
        for doc in docs:
            doc_vec = self._embed(doc)
            sim = _cosine_similarity(query_vec, doc_vec)
            # Map cosine similarity [-1, 1] → [0, 1]
            scores.append((sim + 1.0) / 2.0)
        return scores

    async def test_connection(self) -> ProviderTestResult:
        return ProviderTestResult(ok=True, model="mock-reranker", latency_ms=0)
