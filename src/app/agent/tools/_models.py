"""Pydantic v2 parameter and result models for all 7 agent tools.

Design rules (CLAUDE.md §8, §12):
- ``extra="forbid"`` on every ``*Params`` model.
- NO ``chat_id`` field on any ``*Params`` model — the LLM cannot inject it.
  chat_id is always taken from ``AgentState.chat_id`` inside the tool.
- No ``dict[str, Any]`` except in ``ToolCallRecord.params`` (documented exception).
- All fields use concrete typed Pydantic models.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# inspect_chat
# ---------------------------------------------------------------------------


class InspectChatParams(BaseModel):
    """Parameters for inspect_chat tool.

    No chat_id — injected from AgentState.
    """

    model_config = ConfigDict(extra="forbid")

    include_topics: bool = True
    max_tokens: int = Field(default=2_000, ge=1)


# ---------------------------------------------------------------------------
# inspect_document
# ---------------------------------------------------------------------------


class InspectDocumentParams(BaseModel):
    """Parameters for inspect_document tool.

    No chat_id — injected from AgentState.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: uuid.UUID
    include_section_tree: bool = True
    max_tokens: int = Field(default=2_000, ge=1)


# ---------------------------------------------------------------------------
# fetch_structural_nodes
# ---------------------------------------------------------------------------


class FetchStructuralNodesParams(BaseModel):
    """Parameters for fetch_structural_nodes (deterministic PostgreSQL fetch).

    No chat_id — injected from AgentState.
    """

    model_config = ConfigDict(extra="forbid")

    document_ids: list[uuid.UUID] | None = None
    node_types: list[str] | None = None
    source_types: list[str] | None = None
    limit: int = Field(default=50, ge=1, le=500)
    max_tokens: int = Field(default=4_000, ge=1)


# ---------------------------------------------------------------------------
# search_hybrid
# ---------------------------------------------------------------------------


class SearchHybridParams(BaseModel):
    """Parameters for search_hybrid tool (Vespa hybrid retrieval).

    No chat_id — injected from AgentState.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    document_ids: list[uuid.UUID] | None = None
    source_types: list[str] | None = None
    top_k: int = Field(default=8, ge=1, le=60)
    rerank_mode: Literal["none", "native", "cross_encoder"] = "native"
    max_tokens: int = Field(default=5_000, ge=1)
    preset: Literal["default", "broad"] = "default"


# ---------------------------------------------------------------------------
# grep_document_chunks
# ---------------------------------------------------------------------------


class GrepDocumentChunksParams(BaseModel):
    """Parameters for deterministic chunk grep over stored DocumentNode rows.

    No chat_id — injected from AgentState.  This tool complements
    ``search_hybrid`` by directly scanning literal chunk/table/figure/equation
    text when exact labels or HTML tables matter.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    document_ids: list[uuid.UUID] | None = None
    source_types: list[str] | None = None
    required_terms: list[str] = Field(default_factory=list, max_length=12)
    include_html: bool = False
    scan_limit: int = Field(default=500, ge=1, le=5_000)
    limit: int = Field(default=8, ge=1, le=50)
    max_tokens: int = Field(default=6_000, ge=1)


# ---------------------------------------------------------------------------
# query_structured_facts
# ---------------------------------------------------------------------------


class QueryStructuredFactsParams(BaseModel):
    """Parameters for query_structured_facts tool.

    No chat_id — injected from AgentState (FactsFilter receives it internally).
    """

    model_config = ConfigDict(extra="forbid")

    document_ids: list[uuid.UUID] | None = None
    kinds: list[str] | None = None
    keys: list[str] | None = None
    page_range: tuple[int, int] | None = None
    numeric_min: float | None = None
    numeric_max: float | None = None
    unit_in: list[str] | None = None
    limit: int = Field(default=50, ge=1, le=200)
    max_tokens: int = Field(default=3_000, ge=1)


# ---------------------------------------------------------------------------
# aggregate_sources
# ---------------------------------------------------------------------------


class AggregateSourcesParams(BaseModel):
    """Parameters for aggregate_sources tool.

    No chat_id — chat scope is implicit via state.evidence_items.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["per_document", "per_section"] = "per_section"
    max_summary_tokens: int = Field(default=400, ge=50, le=2_000)
    max_tokens: int = Field(default=5_000, ge=1)


# ---------------------------------------------------------------------------
# expand_evidence
# ---------------------------------------------------------------------------


class ExpandEvidenceParams(BaseModel):
    """Parameters for expand_evidence tool.

    No chat_id — injected from AgentState.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., min_length=1)
    neighborhood: Literal["section", "page", "paragraphs"] = "section"
    max_extra: int = Field(default=3, ge=1, le=20)
    max_tokens: int = Field(default=3_000, ge=1)


__all__ = [
    "AggregateSourcesParams",
    "ExpandEvidenceParams",
    "FetchStructuralNodesParams",
    "GrepDocumentChunksParams",
    "InspectChatParams",
    "InspectDocumentParams",
    "QueryStructuredFactsParams",
    "SearchHybridParams",
]
