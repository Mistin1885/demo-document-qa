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
from typing import Literal

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
    token_count_estimate: int = Field(ge=0, default=0)
    """Tiktoken approximation of detailed_summary length (Phase 5.1 new field)."""

    # ---- provider tracing ----
    model_used: str = Field(default="mock-chat")
    """Model identifier used to generate this enrichment (e.g. 'mock-chat' / 'gpt-4o')."""
    traceable: bool = True
    """Always True; reserved for future audit tooling."""

    @property
    def source_node_id(self) -> uuid.UUID:
        """Alias for ``node_id`` — matches the Phase 5.1 public API spec."""
        return self.node_id


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
# Phase 5.2 (LLM JSON) — DocumentOverview and related item types
# ---------------------------------------------------------------------------


class Contribution(BaseModel):
    """A single paper contribution extracted by the document-level LLM call."""

    model_config = ConfigDict(extra="forbid")

    title: str
    summary: str
    """1~2 sentence description of the contribution."""
    source_section_ids: list[uuid.UUID]
    """IDs of SectionEnrichment.source_node_id that evidence this contribution."""


class FindingItem(BaseModel):
    """A key empirical or theoretical finding from the paper."""

    model_config = ConfigDict(extra="forbid")

    statement: str
    evidence: str | None = None
    """Which section or performance fact supports this finding."""
    source_section_ids: list[uuid.UUID]


class DatasetMention(BaseModel):
    """A dataset referenced in the paper."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: Literal["training", "evaluation", "benchmark", "ablation", "other"]
    size_hint: str | None = None
    """E.g. '5K examples' or '32B tokens' — free text extracted from the paper."""
    source_section_ids: list[uuid.UUID]


class MetricMention(BaseModel):
    """A quantitative metric result reported in the paper."""

    model_config = ConfigDict(extra="forbid")

    name: str
    """E.g. 'F1', 'MMLU', 'BLEU'."""
    best_value: float | None = None
    baseline_value: float | None = None
    improvement: str | None = None
    """Free-text improvement description, e.g. '+3.2% over BM25 baseline'."""
    source_section_ids: list[uuid.UUID]


class ConclusionItem(BaseModel):
    """A conclusion, caveat, or future-work item from the paper."""

    model_config = ConfigDict(extra="forbid")

    statement: str
    category: Literal["finding", "future_work", "caveat", "claim"]
    source_section_ids: list[uuid.UUID]


class MethodMention(BaseModel):
    """A method or technique described in the paper (document-level)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    source_section_ids: list[uuid.UUID] = Field(default_factory=list)


class LimitationMention(BaseModel):
    """A limitation mentioned in the paper (document-level)."""

    model_config = ConfigDict(extra="forbid")

    text: str
    source_section_ids: list[uuid.UUID] = Field(default_factory=list)


class DocumentOverview(BaseModel):
    """LLM-produced paper-wide overview (Phase 5.2 JSON-parsing path).

    Produced by :func:`app.enrichment.document_overview.enrich_document_overview`
    from aggregated section enrichments.  All structured sub-fields carry
    ``source_section_ids`` referencing ``SectionEnrichment.source_node_id`` so
    every claim is traceable to the originating section node.

    Design rules (CLAUDE.md §12)
    ----------------------------
    - No ``dict[str, Any]``; ``metadata_`` uses a typed union.
    - ``ConfigDict(extra="forbid")`` prevents silent data loss.
    - ``chat_id`` is set exclusively from ``hierarchy.chat_id``.
    """

    model_config = ConfigDict(extra="forbid")

    # ---- identity ----
    chat_id: uuid.UUID
    document_id: uuid.UUID

    # ---- document metadata (may be None when not in the paper) ----
    doc_title: str | None = None
    authors_excerpt: str | None = None
    """First 200 chars of the authors string extracted from the paper."""
    page_count: int

    # ---- primary text output ----
    overview: str
    """5~10 sentence paper-wide overview."""

    # ---- structured aggregations ----
    contributions: list[Contribution] = Field(default_factory=list)
    methods: list[MethodMention] = Field(default_factory=list)
    findings: list[FindingItem] = Field(default_factory=list)
    limitations: list[LimitationMention] = Field(default_factory=list)
    datasets: list[DatasetMention] = Field(default_factory=list)
    metrics: list[MetricMention] = Field(default_factory=list)
    conclusions: list[ConclusionItem] = Field(default_factory=list)

    # ---- provenance ----
    source_section_ids: list[uuid.UUID] = Field(default_factory=list)
    """IDs of SectionEnrichment.source_node_id consumed to produce this overview."""

    # ---- token estimate ----
    token_count_estimate: int = Field(ge=0, default=0)
    """Tiktoken cl100k_base estimate of the overview text length."""

    # ---- provider tracing ----
    model_used: str = Field(default="mock-chat")

    # ---- extensible metadata ----
    metadata_: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


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


# ---------------------------------------------------------------------------
# Phase 5.4 additional models — richer manifest types for Phase 7 agent
# ---------------------------------------------------------------------------


class ManifestTopic(BaseModel):
    """A topic term aggregated across documents in a chat.

    Used by :class:`ChatManifestV2` to give the Phase 7 ``inspect_chat`` tool
    a cross-document topic map without loading full summaries.
    """

    model_config = ConfigDict(extra="forbid")

    term: str
    weight: float = Field(ge=0.0, le=1.0)
    document_ids: list[uuid.UUID]
    """IDs of documents whose summaries mention this topic term."""


class ChatManifestEntry(BaseModel):
    """Richer per-document entry for the Phase 7 agent manifest.

    Carries enrichment state, fact count, and status breakdowns so the
    ``inspect_chat`` tool can decide retrieval routing without fetching full
    summaries.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: uuid.UUID
    chat_id: uuid.UUID
    title: str | None
    """Document title; falls back to Document.original_filename stem."""
    authors_excerpt: str | None
    """First ≤200 chars of authors string when persisted; None otherwise."""
    abstract_summary: str | None
    """Compact abstract summary when persisted (kind='chapter_summary'); None otherwise."""
    overview_summary: str | None
    """Paper-level overview compact when persisted (kind='document_overview'); None otherwise."""
    page_count: int | None
    source_type: Literal["upload", "arxiv", "url"]
    status: str
    """Document.status raw value."""
    checksum_sha256: str
    token_count_estimate: int
    """Sum of all Summary.token_count rows for this document; 0 if none."""
    enrichment_state: Literal["none", "sections_only", "document_overview", "facts_only", "full"]
    facts_count: int
    """COUNT(structured_facts) for this document scoped to current chat."""
    sections_count: int
    """COUNT(document_nodes WHERE node_type IN ('section','subsection'))."""
    added_at: datetime
    """Document.created_at."""
    updated_at: datetime
    """Document.updated_at."""


class ChatManifestV2(BaseModel):
    """Extended chat-level manifest for the Phase 7 ``inspect_chat`` tool.

    Adds cross-document topic aggregation and breakdowns on top of the
    lightweight :class:`ChatManifest`.  Always computed on demand from
    PostgreSQL — no Vespa, no LLM call.
    """

    model_config = ConfigDict(extra="forbid")

    chat_id: uuid.UUID
    chat_name: str | None
    generated_at: datetime
    document_count: int
    total_token_count_estimate: int
    documents: list[ChatManifestEntry]
    topics: list[ManifestTopic]
    """Top-N topics aggregated across all document summaries."""
    source_type_breakdown: dict[str, int]
    """Count of documents per source_type ('upload', 'arxiv', 'url')."""
    status_breakdown: dict[str, int]
    """Count of documents per Document.status value."""
    metadata_: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


__all__ = [
    # Phase 5.1
    "ChatManifest",
    "ChatManifestEntry",
    "ChatManifestV2",
    "ClaimItem",
    "DefinitionItem",
    "DocumentEnrichment",
    "DocumentManifestEntry",
    "LimitationItem",
    "ManifestTopic",
    "MethodItem",
    "PerformanceFactItem",
    "SectionEnrichment",
    # Phase 5.2 LLM-JSON path
    "Contribution",
    "FindingItem",
    "DatasetMention",
    "MetricMention",
    "ConclusionItem",
    "MethodMention",
    "LimitationMention",
    "DocumentOverview",
]
