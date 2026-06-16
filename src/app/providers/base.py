"""Abstract base interfaces for Chat, Embedding, and Reranker providers.

All provider adapters (OpenAI, Gemini Native, OpenAI-compatible / vLLM, mock)
must implement the ABCs defined here.  Downstream code (agent, enrichment,
retrieval) should depend **only** on these ABCs, never on a concrete adapter.

Design constraints (CLAUDE.md §9 / §12)
----------------------------------------
- Chat / Embedding / Reranker are **independent** profiles.  Never assume a
  chat provider also handles embeddings.
- All calls are async-first.  No sync I/O is permitted inside an adapter.
- API keys must never appear in log messages.  Adapters MUST call
  ``app.security.mask_secret`` before passing credentials to a logger.
- ``test_connection()`` on every provider returns a ``ProviderTestResult``
  (never raises unless there is a genuine programming error).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared data models (Pydantic v2 BaseModel — no unbounded dict[str, Any])
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single turn in a chat completion request."""

    role: str = Field(..., description="'system', 'user', or 'assistant'")
    content: str


class Usage(BaseModel):
    """Token usage reported by a chat completion response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatCompletion(BaseModel):
    """The full response to a chat completion request."""

    content: str
    usage: Usage
    model: str


class ChatChunk(BaseModel):
    """A single streaming chunk from a chat completion."""

    delta: str
    finish_reason: str | None = None


class ProviderTestResult(BaseModel):
    """Result of a ``test_connection()`` call on any provider."""

    ok: bool
    model: str | None = None
    latency_ms: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class ChatProvider(ABC):
    """Abstract interface for large language model chat providers.

    Subclasses
    ----------
    - ``OpenAIChatProvider``   — official OpenAI SDK (AsyncOpenAI)
    - ``OpenAICompatChatProvider`` — any OpenAI-compatible endpoint (vLLM /
      Gemini OpenAI-compat / self-hosted)
    - ``GeminiNativeChatProvider`` — google-genai SDK
    - ``MockChatProvider``     — deterministic, never makes network calls
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name (e.g., ``'openai'``, ``'mock'``)."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier string (e.g., ``'gpt-4o'``, ``'gemini-2.0-flash'``)."""

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Maximum context window in tokens for this model/provider."""

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> ChatCompletion:
        """Send a chat completion request and return the full response.

        Parameters
        ----------
        messages:
            Conversation history (system + user + assistant turns).
        temperature:
            Sampling temperature.  0.0 is deterministic.
        max_tokens:
            Maximum tokens in the completion.
        stop:
            Optional list of stop sequences.

        Returns
        -------
        ChatCompletion
        """

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Stream a chat completion as ``ChatChunk`` objects.

        This is an async generator method.  Callers use it as::

            async for chunk in provider.stream(messages, temperature=0.7):
                print(chunk.delta, end="", flush=True)
        """

    @abstractmethod
    async def test_connection(self) -> ProviderTestResult:
        """Send a minimal test request to verify connectivity and credentials.

        Returns
        -------
        ProviderTestResult
            ``ok=True`` on success; ``ok=False`` with ``error`` on failure.
            Never raises (exceptions are caught and surfaced via ``error``).
        """


class EmbeddingProvider(ABC):
    """Abstract interface for embedding (text → vector) providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors produced."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a batch of texts.

        Parameters
        ----------
        texts:
            Non-empty list of strings to embed.

        Returns
        -------
        list[list[float]]
            One embedding vector per input text (shape: ``[len(texts), dimension]``).
        """

    @abstractmethod
    async def test_connection(self) -> ProviderTestResult:
        """Verify connectivity.  Same contract as ``ChatProvider.test_connection``."""


class RerankerProvider(ABC):
    """Abstract interface for cross-encoder / reranker providers.

    The reranker takes a query and a list of candidate passages and returns a
    relevance score for each passage.  Higher scores indicate greater relevance.

    Score range is adapter-dependent (often ``[0, 1]`` or ``[-∞, +∞]``); callers
    should only rely on **relative ordering**, not absolute thresholds.
    """

    @abstractmethod
    async def score(self, query: str, docs: list[str]) -> list[float]:
        """Return a relevance score for each document in *docs*.

        Parameters
        ----------
        query:
            The search query.
        docs:
            Candidate passages to score against the query.

        Returns
        -------
        list[float]
            One score per document (same length and order as *docs*).
        """

    @abstractmethod
    async def test_connection(self) -> ProviderTestResult:
        """Verify connectivity.  Same contract as ``ChatProvider.test_connection``."""
