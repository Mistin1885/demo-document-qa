"""Pydantic v2 models for Section-level enrichment (Phase 5.1).

Design rules (CLAUDE.md §12)
------------------------------
- All models use ``ConfigDict(extra="forbid")`` — no silent data loss.
- No ``dict[str, Any]``; every structured field is a named Pydantic model.
- ``SectionEnrichment`` is the canonical output consumed by:
    - ``app.services.enrichment_service`` (writes to ``summaries`` table).
    - Phase 6 ingestion (Vespa feed with different ``source_type`` values).

``SectionEnrichment.source_block_ids`` is a *subset* of the originating
``DocumentNodeOut.source_block_ids`` so consumers can trace back to raw
``ParsedBlock`` instances.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field  # noqa: TCH002

from app.models.domain import IngestionState

# ---------------------------------------------------------------------------
# Item sub-types
# ---------------------------------------------------------------------------


class ClaimItem(BaseModel):
    """A single claim extracted from a section."""

    model_config = ConfigDict(extra="forbid")

    text: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class DefinitionItem(BaseModel):
    """A term-definition pair extracted from a section."""

    model_config = ConfigDict(extra="forbid")

    term: str
    definition: str


class MethodItem(BaseModel):
    """A method / technique described in a section."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str


class LimitationItem(BaseModel):
    """A limitation or constraint mentioned in a section."""

    model_config = ConfigDict(extra="forbid")

    text: str


class PerformanceFactItem(BaseModel):
    """A quantitative performance fact extracted from a section."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    value: str
    context: str | None = None


# ---------------------------------------------------------------------------
# SectionEnrichment — primary output of Phase 5.1
# ---------------------------------------------------------------------------


class SectionEnrichment(BaseModel):
    """Enrichment payload for one section / subsection / appendix node.

    Persistence
    -----------
    ``app.services.enrichment_service.persist_section_summaries`` writes two
    rows into ``summaries`` per enrichment:

    - ``kind="section_detailed"``  — ``detailed_summary``
    - ``kind="section_compact"``   — ``compact_summary``

    The remaining structured lists (claims, definitions, …) are Phase 6
    Vespa-feed inputs and are not persisted to ``summaries``.
    """

    model_config = ConfigDict(extra="forbid")

    # ---- identity (must match the originating DocumentNodeOut) ----
    node_id: uuid.UUID
    chat_id: uuid.UUID
    document_id: uuid.UUID
    node_type: str  # "section" | "subsection" | "appendix"
    title: str | None

    # ---- page range (copied from DocumentNodeOut) ----
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    # ---- provenance ----
    source_block_ids: list[uuid.UUID]
    """IDs of ParsedBlock instances that belong to this section node."""

    # ---- summaries ----
    detailed_summary: str
    compact_summary: str

    # ---- keywords (list[str] — no dict[str, Any]) ----
    keywords: list[str] = Field(default_factory=list)
    technical_keywords: list[str] = Field(default_factory=list)

    # ---- named entities ----
    entities: list[str] = Field(default_factory=list)

    # ---- structured extractions ----
    definitions: list[DefinitionItem] = Field(default_factory=list)
    claims: list[ClaimItem] = Field(default_factory=list)
    methods: list[MethodItem] = Field(default_factory=list)
    limitations: list[LimitationItem] = Field(default_factory=list)
    performance_facts: list[PerformanceFactItem] = Field(default_factory=list)

    # ---- cross-references within document ----
    related_figure_ids: list[uuid.UUID] = Field(default_factory=list)
    related_table_ids: list[uuid.UUID] = Field(default_factory=list)

    # ---- token estimate ----
    token_count: int = Field(ge=0, default=0)


# ---------------------------------------------------------------------------
# DocumentEnrichment — primary output of Phase 5.2
# ---------------------------------------------------------------------------


class DocumentEnrichment(BaseModel):
    """Enrichment payload aggregated across the full document (Phase 5.2).

    Persistence
    -----------
    ``app.services.enrichment_service.persist_document_summaries`` writes at
    most two rows into ``summaries``:

    - ``kind="document_overview"`` — ``document_overview``
    - ``kind="chapter_summary"``   — ``abstract_summary`` (only when non-None
      and ``abstract_node_id`` is provided)

    All other structured lists (contributions, methods, findings, …) are
    Phase 6 Vespa-feed inputs and are **not** persisted to ``summaries``.

    Design rules (CLAUDE.md §12)
    ----------------------------
    - ``chat_id`` is taken exclusively from ``hierarchy.chat_id``; never set
      by an external caller.
    - No ``dict[str, Any]``; every field is a named Pydantic type.
    - ``ConfigDict(extra="forbid")`` prevents silent data loss.
    """

    model_config = ConfigDict(extra="forbid")

    # ---- identity ----
    chat_id: uuid.UUID
    document_id: uuid.UUID

    # ---- primary text outputs ----
    document_overview: str
    """Concise document overview (≤ 1500 chars)."""

    abstract_summary: str | None
    """Condensed abstract (≤ 400 chars); ``None`` when no abstract node exists."""

    # ---- structured aggregations (union-deduped from SectionEnrichment) ----
    main_contributions: list[str]
    main_methods: list[str]
    main_technologies: list[str]
    main_findings: list[str]
    main_limitations: list[str]
    main_datasets: list[str]
    main_metrics: list[str]
    main_experimental_results: list[str]
    main_conclusions: list[str]

    # ---- provenance ----
    source_section_node_ids: list[uuid.UUID]
    """IDs of section/subsection/appendix nodes whose enrichments were consumed."""

    # ---- token estimate ----
    token_count_estimate: int
    """Rough word-count-based token estimate (not tiktoken)."""


# ---------------------------------------------------------------------------
# Phase 5.4 — Chat-level manifest models
# ---------------------------------------------------------------------------


class DocumentManifestEntry(BaseModel):
    """Manifest entry for a single document within a Chat.

    All fields are derived read-time from PostgreSQL — no Vespa, no LLM call.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: uuid.UUID
    title: str | None
    """Root DocumentNode title, or None when the document node has no title."""
    authors: list[str]
    """Author list; empty when not yet extracted."""
    page_count: int | None
    """From Document.page_count."""

    abstract_summary: str | None
    """From summaries.kind='chapter_summary' content; None when absent."""
    main_topics: list[str]
    """Top-8 deduplicated topics from document_overview summary keywords + entities."""

    section_count: int
    """Number of section/subsection document_nodes belonging to this document."""
    token_estimate: int
    """Sum of token_count from document_overview + section summaries; fallback heuristic."""

    available_source_types: list[str]
    """Distinct summary kinds + 'fact:<kind>' for distinct structured_fact kinds.

    Only reflects what is actually persisted in PostgreSQL (Phase 6 Vespa
    chunks are not included yet).
    """
    ingestion_status: IngestionState
    """State of the most-recent IngestionJob; 'pending' when no jobs exist."""


class ChatManifest(BaseModel):
    """Chat-level manifest aggregated read-time from PostgreSQL.

    Never generated in the ingestion pipeline; always computed on demand
    (GUIDE §8.3).
    """

    model_config = ConfigDict(extra="forbid")

    chat_id: uuid.UUID
    generated_at: datetime
    """UTC naive (timezone-stripped) to match Phase 3 convention."""
    document_count: int
    total_token_estimate: int
    documents: list[DocumentManifestEntry]
    ingestion_summary: dict[IngestionState, int]
    """Counts of latest-per-document ingestion state across the chat."""


__all__ = [
    "ChatManifest",
    "ClaimItem",
    "DefinitionItem",
    "DocumentEnrichment",
    "DocumentManifestEntry",
    "LimitationItem",
    "MethodItem",
    "PerformanceFactItem",
    "SectionEnrichment",
]
