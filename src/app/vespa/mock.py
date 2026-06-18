"""Null/mock RetrievalService for use in tests and development without Vespa.

NullRetrievalService returns empty results for every search request, satisfying
the interface expected by ToolDeps.retrieval_service without requiring a live
Vespa cluster.
"""

from __future__ import annotations

from app.retrieval.models import RetrievalRequest, RetrievalResponse


class NullRetrievalService:
    """No-op retrieval service that always returns zero hits.

    Used by the messages API router when no real Vespa instance is configured,
    and in unit tests to avoid network dependencies.
    """

    async def search(self, request: RetrievalRequest) -> RetrievalResponse:
        """Return an empty RetrievalResponse for any request."""
        return RetrievalResponse(hits=[])


__all__ = ["NullRetrievalService"]
