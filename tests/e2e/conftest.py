"""Phase 9.2 — E2E test helpers.

Reuses the session-scoped fixtures from ``tests/conftest.py`` (``api_client``,
``db_session``) which run against the real Postgres database with per-test
SAVEPOINT rollback.  The helpers below build deterministic provider mocks the
E2E flow injects via FastAPI ``dependency_overrides`` / ``unittest.mock.patch``
so the messages endpoint can be exercised end-to-end without contacting a real
LLM / Vespa.

Isolation contract (CLAUDE.md §2):
  - The mocked retrieval service only returns hits whose ``chat_id`` matches
    the request's ``chat_id`` — mirroring the production Vespa filter so a
    forged or cross-chat seed leaks nothing.
  - The mocked chat provider emits inline citation markers based on the
    evidence block, so ``validate_citations`` builds real ``CitationDraft``
    objects bound to the requested chat.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from app.evaluation.qa_eval import CitingMockChatProvider
from app.providers.base import ChatProvider
from app.retrieval.models import RetrievalRequest, RetrievalResponse, SearchHit


class ChatScopedRetrievalMock:
    """Retrieval mock that only returns hits whose chat_id matches the request.

    Tests seed an arbitrary ``hits`` list across chats; the mock filters by
    ``request.chat_id`` on every call.  Cross-chat seeds therefore yield zero
    hits, allowing the E2E suite to assert refusal behaviour without bypassing
    the production isolation contract.
    """

    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self._hits = list(hits or [])
        self.call_count = 0

    async def search(self, request: RetrievalRequest) -> RetrievalResponse:
        self.call_count += 1
        scoped = [h for h in self._hits if h.chat_id == str(request.chat_id)]
        return RetrievalResponse(hits=scoped)


def make_hit(
    *,
    chat_id: uuid.UUID,
    document_id: uuid.UUID,
    content: str,
    page_start: int = 1,
    page_end: int = 1,
    source_type: str = "chunk",
    order_index: int = 0,
) -> SearchHit:
    """Build a deterministic ``SearchHit`` for a specific chat / document."""
    return SearchHit(
        vespa_document_id=f"id::document_chunk::e2e-{document_id}-{order_index}",
        chat_id=str(chat_id),
        document_id=str(document_id),
        source_node_id=f"node-{document_id}-{order_index}",
        source_type=source_type,
        content=content,
        page_start=page_start,
        page_end=page_end,
        order_index=order_index,
        fusion_score=0.8,
        final_score=0.8,
        final_rank=order_index + 1,
    )


# Builder type used by tests: returns the provider tuple expected by
# ``app.api.messages._build_providers``.
ProviderBuilder = Callable[
    [object, uuid.UUID], Awaitable[tuple[ChatProvider, ChatScopedRetrievalMock]]
]


def make_provider_builder(hits: list[SearchHit]) -> ProviderBuilder:
    """Return an async callable that mirrors ``_build_providers`` signature.

    The same ``ChatScopedRetrievalMock`` instance is returned for every chat
    so tests can assert cross-chat isolation by seeding hits from multiple
    chats in a single list.
    """
    retrieval = ChatScopedRetrievalMock(hits=hits)
    chat_provider = CitingMockChatProvider()

    async def _build(db: object, chat_id: uuid.UUID) -> tuple[ChatProvider, ChatScopedRetrievalMock]:
        return chat_provider, retrieval

    return _build


__all__ = [
    "ChatScopedRetrievalMock",
    "ProviderBuilder",
    "make_hit",
    "make_provider_builder",
]
