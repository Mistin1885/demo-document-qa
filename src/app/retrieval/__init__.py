"""Retrieval package — RetrievalService is the ONLY Vespa query entry point.

CLAUDE.md §7:
- All Vespa queries go through RetrievalService.search().
- chat_id filter is injected by the service layer; the LLM cannot pass it.
- BM25 + ANN candidates fused via RRF, then reranked (native or cross-encoder).
"""

from app.retrieval.models import RetrievalDebug, RetrievalRequest, RetrievalResponse, SearchHit
from app.retrieval.service import RetrievalService

__all__ = [
    "RetrievalDebug",
    "RetrievalRequest",
    "RetrievalResponse",
    "RetrievalService",
    "SearchHit",
]
