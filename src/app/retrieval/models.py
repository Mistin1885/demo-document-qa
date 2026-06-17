"""Pydantic v2 models for the RetrievalService public interface.

All models use ``extra="forbid"`` to prevent silent data loss (CLAUDE.md §12).
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# SearchHit — one returned document chunk with all stage scores
# ---------------------------------------------------------------------------


class SearchHit(BaseModel):
    """One Vespa document chunk returned by RetrievalService.search().

    Score fields are ``float | None``:
    - ``bm25_score``: weighted BM25 score from the ``bm25_only`` rank profile.
    - ``vector_score``: closeness score from the ``semantic_only`` rank profile.
    - ``fusion_score``: RRF fusion score (always present after fusion).
    - ``native_rerank_score``: Vespa second-phase score from
      ``hybrid_with_native_rerank`` (only for rerank_mode="native").
    - ``cross_encoder_score``: score from external reranker provider
      (only for rerank_mode="cross_encoder").
    - ``final_rank``: 1-based rank in the final returned list.
    - ``final_score``: whichever score drove the final ranking.
    """

    model_config = ConfigDict(extra="forbid")

    vespa_document_id: str
    chat_id: str
    document_id: str
    source_node_id: str
    parent_node_id: str | None = None
    source_type: str
    title: str | None = None
    heading_path: str | None = None
    content: str
    page_start: int
    page_end: int
    order_index: int

    # Per-stage scores (all optional)
    bm25_score: float | None = None
    vector_score: float | None = None
    fusion_score: float | None = None
    native_rerank_score: float | None = None
    cross_encoder_score: float | None = None
    final_rank: int = 0
    final_score: float | None = None


# ---------------------------------------------------------------------------
# RetrievalRequest
# ---------------------------------------------------------------------------


class RetrievalRequest(BaseModel):
    """Input to RetrievalService.search().

    ``chat_id`` is the mandatory isolation boundary — injected by the service
    layer from AgentState, never by the LLM.

    ``rerank_mode`` controls how the fused list is reranked:
    - ``"none"`` — skip reranking, use RRF order directly.
    - ``"native"`` — fire a third Vespa query using the
      ``hybrid_with_native_rerank`` rank profile.
    - ``"cross_encoder"`` — call the external RerankerProvider.

    ``retrieval_mode`` controls which sub-queries are issued:
    - ``"hybrid"`` — BM25 + ANN + RRF (default production mode).
    - ``"bm25_only"`` — skip ANN; return BM25 hits directly (no fusion).
    - ``"vector_only"`` — skip BM25; return ANN hits directly (no fusion).
    Used by the retrieval evaluation harness to compare search variants.
    """

    model_config = ConfigDict(extra="forbid")

    chat_id: UUID
    query: str = Field(..., min_length=1)

    # Optional filters
    document_ids: list[UUID] | None = None
    source_types: list[str] | None = None

    # Retrieval sizing knobs
    bm25_top_k: int = Field(default=60, ge=1)
    ann_top_k: int = Field(default=60, ge=1)
    fusion_top_k: int = Field(default=80, ge=1)
    rerank_top_k: int = Field(default=30, ge=1)
    final_top_k: int = Field(default=8, ge=1)

    rerank_mode: Literal["none", "native", "cross_encoder"] = "native"
    retrieval_mode: Literal["hybrid", "bm25_only", "vector_only"] = "hybrid"
    debug: bool = False


# ---------------------------------------------------------------------------
# RetrievalDebug
# ---------------------------------------------------------------------------


class RetrievalTimings(BaseModel):
    """Per-stage latency in milliseconds."""

    model_config = ConfigDict(extra="forbid")

    embed_ms: float = 0.0
    bm25_ms: float = 0.0
    ann_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0


class RetrievalDebug(BaseModel):
    """Debug payload attached to RetrievalResponse when ``debug=True``."""

    model_config = ConfigDict(extra="forbid")

    bm25_hits_count: int = 0
    ann_hits_count: int = 0
    fused_hits_count: int = 0
    after_rerank_count: int = 0
    timings_ms: RetrievalTimings = Field(default_factory=RetrievalTimings)
    queries: list[str] = Field(default_factory=list)
    """YQL strings sent to Vespa (in order: BM25, ANN, [native-rerank])."""


# ---------------------------------------------------------------------------
# RetrievalResponse
# ---------------------------------------------------------------------------


class RetrievalResponse(BaseModel):
    """Output of RetrievalService.search()."""

    model_config = ConfigDict(extra="forbid")

    hits: list[SearchHit] = Field(default_factory=list)
    debug: RetrievalDebug | None = None
