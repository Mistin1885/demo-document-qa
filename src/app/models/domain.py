"""Pydantic v2 domain models for Paper Notebook Agent.

Design rules
------------
- Every ORM entity has three companion models:
    ``XxxRead``   — response shape (includes ``id`` + timestamps).
    ``XxxCreate`` — request shape (caller-supplied fields only).
    ``XxxUpdate`` — patch shape (all fields Optional for partial updates).
- ``model_config = ConfigDict(from_attributes=True)`` on Read models enables
  ``XxxRead.model_validate(orm_instance)``.
- Enumerated string literals use ``Literal[...]`` instead of plain ``str``
  so the schema is self-documenting and validators catch invalid values early.
- JSONB sub-structures (Citations, ToolTrace, NodeMetadata) are given named
  Pydantic models with a minimal field set; Phase 3/4 will extend them.
  ``extra="forbid"`` on sub-structures prevents silent data loss.
- Never use ``dict[str, Any]`` — CLAUDE.md §12 prohibition.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# JSONB sub-structures
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """Source citation attached to an assistant message.

    Maps to CLAUDE.md §13 ``Citation`` contract.
    """

    model_config = ConfigDict(extra="forbid")

    citation_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    chat_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    page_start: int
    page_end: int
    section_title: str | None = None
    source_node_id: uuid.UUID | None = None
    excerpt: str


class ToolTraceStep(BaseModel):
    """A single step recorded in a tool execution trace."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    status: Literal["ok", "overflow", "error", "info"]
    token_estimate: int | None = None
    note: str | None = None


class ToolTrace(BaseModel):
    """Trace of all tool calls made during agent execution for a message."""

    model_config = ConfigDict(extra="forbid")

    steps: list[ToolTraceStep] = Field(default_factory=list)
    total_rounds: int = 0
    token_count_is_estimate: bool = False


class BBox(BaseModel):
    """Bounding box in PDF-point space (from MinerU middle.json)."""

    model_config = ConfigDict(extra="forbid")

    x0: float
    y0: float
    x1: float
    y1: float
    page: int | None = None


class NodeMetadata(BaseModel):
    """Extensible metadata bag for a DocumentNode.

    Phase 4 (MinerU mapping) will add more fields; until then this is a
    minimal placeholder.
    """

    model_config = ConfigDict(extra="allow")

    font_size: float | None = None
    font_name: str | None = None
    column_index: int | None = None
    confidence: float | None = None
    reading_order: int | None = None
    image_path: str | None = None


# ---------------------------------------------------------------------------
# Enumerations (as Literal types)
# ---------------------------------------------------------------------------

ProviderKind = Literal["chat", "embedding", "reranker"]
ProviderType = Literal["openai", "gemini_native", "gemini_compat", "openai_compat", "vllm"]
MessageRole = Literal["user", "assistant", "system", "tool"]
DocumentSourceType = Literal["upload", "arxiv", "url"]
DocumentStatus = Literal["uploaded", "parsing", "parsed", "enriching", "indexed", "failed"]
NodeType = Literal[
    "document", "section", "subsection", "paragraph",
    "figure", "table", "equation", "reference",
]
SummaryKind = Literal[
    "section_detailed", "section_compact",
    "chapter_summary", "compact_chapter_summary",
    "document_overview", "technology_card",
]
FactKind = Literal["metric", "benchmark", "dataset", "hyperparameter", "ablation", "other"]
IngestionState = Literal["pending", "running", "succeeded", "failed"]


# ---------------------------------------------------------------------------
# ProviderProfile
# ---------------------------------------------------------------------------


class ProviderProfileCreate(BaseModel):
    kind: ProviderKind
    provider_type: ProviderType
    name: str
    base_url: str | None = None
    model: str
    api_key_plaintext: str | None = Field(
        default=None,
        description="Plaintext key — service layer encrypts before persisting.",
    )
    config: dict = Field(default_factory=dict)  # type: ignore[type-arg]
    context_window: int | None = None
    is_default: bool = False


class ProviderProfileUpdate(BaseModel):
    name: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key_plaintext: str | None = None
    config: dict | None = None  # type: ignore[type-arg]
    context_window: int | None = None
    is_default: bool | None = None


class ProviderProfileRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kind: ProviderKind
    provider_type: ProviderType
    name: str
    base_url: str | None
    model: str
    # NOTE: api_key_encrypted is never returned to the frontend.
    config: dict  # type: ignore[type-arg]
    context_window: int | None
    is_default: bool
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class ChatCreate(BaseModel):
    name: str
    description: str | None = None
    default_chat_profile_id: uuid.UUID | None = None
    default_embedding_profile_id: uuid.UUID | None = None
    default_reranker_profile_id: uuid.UUID | None = None


class ChatUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    default_chat_profile_id: uuid.UUID | None = None
    default_embedding_profile_id: uuid.UUID | None = None
    default_reranker_profile_id: uuid.UUID | None = None


class ChatRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    default_chat_profile_id: uuid.UUID | None
    default_embedding_profile_id: uuid.UUID | None
    default_reranker_profile_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class SessionCreate(BaseModel):
    chat_id: uuid.UUID
    name: str | None = None
    chat_profile_id: uuid.UUID | None = None
    selected_document_ids: list[uuid.UUID] | None = None


class SessionUpdate(BaseModel):
    name: str | None = None
    chat_profile_id: uuid.UUID | None = None
    selected_document_ids: list[uuid.UUID] | None = None


class SessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    chat_id: uuid.UUID
    name: str | None
    chat_profile_id: uuid.UUID | None
    selected_document_ids: list[uuid.UUID] | None = None
    document_scope_locked: bool = False
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class MessageCreate(BaseModel):
    session_id: uuid.UUID
    role: MessageRole
    content: str
    citations: list[Citation] | None = None
    tool_trace: ToolTrace | None = None
    token_count: int | None = None


class MessageUpdate(BaseModel):
    content: str | None = None
    citations: list[Citation] | None = None
    tool_trace: ToolTrace | None = None
    token_count: int | None = None


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    role: MessageRole
    content: str
    citations: list[Citation] | None
    tool_trace: ToolTrace | None
    token_count: int | None
    created_at: datetime


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class DocumentCreate(BaseModel):
    chat_id: uuid.UUID
    source_type: DocumentSourceType
    original_filename: str
    storage_path: str
    mime_type: str
    page_count: int | None = None
    status: DocumentStatus = "uploaded"
    checksum_sha256: str


class DocumentUpdate(BaseModel):
    original_filename: str | None = None
    storage_path: str | None = None
    page_count: int | None = None
    status: DocumentStatus | None = None


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    chat_id: uuid.UUID
    source_type: DocumentSourceType
    original_filename: str
    storage_path: str
    mime_type: str
    page_count: int | None
    status: DocumentStatus
    checksum_sha256: str
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# DocumentNode
# ---------------------------------------------------------------------------


class DocumentNodeCreate(BaseModel):
    document_id: uuid.UUID
    chat_id: uuid.UUID
    parent_id: uuid.UUID | None = None
    node_type: NodeType
    title: str | None = None
    content: str
    page_start: int
    page_end: int
    order_index: int
    level: int
    bbox: BBox | None = None
    metadata_: NodeMetadata = Field(default_factory=NodeMetadata)


class DocumentNodeUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    order_index: int | None = None
    level: int | None = None
    bbox: BBox | None = None
    metadata_: NodeMetadata | None = None


class DocumentNodeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    chat_id: uuid.UUID
    parent_id: uuid.UUID | None
    node_type: NodeType
    title: str | None
    content: str
    page_start: int
    page_end: int
    order_index: int
    level: int
    bbox: BBox | None
    metadata_: NodeMetadata
    created_at: datetime


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class SummaryCreate(BaseModel):
    chat_id: uuid.UUID
    document_id: uuid.UUID
    source_node_id: uuid.UUID | None = None
    kind: SummaryKind
    content: str
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    token_count: int = 0


class SummaryUpdate(BaseModel):
    content: str | None = None
    keywords: list[str] | None = None
    entities: list[str] | None = None
    token_count: int | None = None


class SummaryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    chat_id: uuid.UUID
    document_id: uuid.UUID
    source_node_id: uuid.UUID | None
    kind: SummaryKind
    content: str
    keywords: list[str]
    entities: list[str]
    token_count: int
    created_at: datetime


# ---------------------------------------------------------------------------
# StructuredFact
# ---------------------------------------------------------------------------


class FactValue(BaseModel):
    """Typed wrapper for the JSONB ``value`` column in structured_facts."""

    model_config = ConfigDict(extra="allow")

    raw: str | None = None
    numeric: float | None = None
    items: list[str] | None = None


class StructuredFactCreate(BaseModel):
    chat_id: uuid.UUID
    document_id: uuid.UUID
    source_node_id: uuid.UUID | None = None
    kind: FactKind
    key: str
    value: FactValue
    unit: str | None = None
    context_excerpt: str | None = None
    page: int | None = None


class StructuredFactUpdate(BaseModel):
    kind: FactKind | None = None
    key: str | None = None
    value: FactValue | None = None
    unit: str | None = None
    context_excerpt: str | None = None
    page: int | None = None


class StructuredFactRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    chat_id: uuid.UUID
    document_id: uuid.UUID
    source_node_id: uuid.UUID | None
    kind: FactKind
    key: str
    value: FactValue
    unit: str | None
    context_excerpt: str | None
    page: int | None
    created_at: datetime


# ---------------------------------------------------------------------------
# IngestionJob
# ---------------------------------------------------------------------------


class IngestionJobCreate(BaseModel):
    chat_id: uuid.UUID
    document_id: uuid.UUID
    state: IngestionState = "pending"
    attempt: int = 0


class IngestionJobUpdate(BaseModel):
    state: IngestionState | None = None
    attempt: int | None = None
    last_error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class IngestionJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    chat_id: uuid.UUID
    document_id: uuid.UUID
    state: IngestionState
    attempt: int
    last_error: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime


# ---------------------------------------------------------------------------
# ChatDocument (association)
# ---------------------------------------------------------------------------


class ChatDocumentCreate(BaseModel):
    chat_id: uuid.UUID
    document_id: uuid.UUID


class ChatDocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chat_id: uuid.UUID
    document_id: uuid.UUID


# ---------------------------------------------------------------------------
# Agent QA response (CLAUDE.md §13)
# ---------------------------------------------------------------------------


class QAResponse(BaseModel):
    """Full response returned by the agent QA endpoint."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    documents_used: list[uuid.UUID] = Field(default_factory=list)
    coverage: float = Field(ge=0.0, le=1.0, description="0–1 coverage confidence")
    uncertainty: list[str] = Field(default_factory=list)
    session_id: uuid.UUID
    message_id: uuid.UUID
    debug_trace: ToolTrace | None = None


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # sub-structures
    "Citation",
    "ToolTraceStep",
    "ToolTrace",
    "BBox",
    "NodeMetadata",
    "FactValue",
    # enums
    "ProviderKind",
    "ProviderType",
    "MessageRole",
    "DocumentSourceType",
    "DocumentStatus",
    "NodeType",
    "SummaryKind",
    "FactKind",
    "IngestionState",
    # domain models
    "ProviderProfileCreate",
    "ProviderProfileUpdate",
    "ProviderProfileRead",
    "ChatCreate",
    "ChatUpdate",
    "ChatRead",
    "SessionCreate",
    "SessionUpdate",
    "SessionRead",
    "MessageCreate",
    "MessageUpdate",
    "MessageRead",
    "DocumentCreate",
    "DocumentUpdate",
    "DocumentRead",
    "DocumentNodeCreate",
    "DocumentNodeUpdate",
    "DocumentNodeRead",
    "SummaryCreate",
    "SummaryUpdate",
    "SummaryRead",
    "StructuredFactCreate",
    "StructuredFactUpdate",
    "StructuredFactRead",
    "IngestionJobCreate",
    "IngestionJobUpdate",
    "IngestionJobRead",
    "ChatDocumentCreate",
    "ChatDocumentRead",
    "QAResponse",
]
