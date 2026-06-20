"""SQLAlchemy 2.x ORM models for Paper Notebook Agent.

All tables carry ``chat_id`` where documents are scoped, and the column is
always indexed to enforce the CLAUDE.md §2 isolation model.

Design rules
------------
- Declarative 2.x typed API (``Mapped`` / ``mapped_column``).
- All PKs are ``uuid.UUID`` generated client-side via ``uuid.uuid4``.
- Timestamps are timezone-aware ``datetime`` with ``server_default=func.now()``.
- JSONB columns use ``postgresql.JSONB``; fall back to ``JSON`` on non-PG engines.
- No business logic in this file — pure schema definition.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uuid_pk() -> Mapped[uuid.UUID]:
    """Shorthand for a UUID primary key column."""
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _now_col() -> Mapped[datetime]:
    """Shorthand for a created_at / updated_at column with server default."""
    return mapped_column(server_default=func.now(), nullable=False)


def _fk_uuid(target: str, *, index: bool = True, nullable: bool = False) -> Mapped[uuid.UUID]:
    """Shorthand for a non-nullable UUID FK column."""
    return mapped_column(UUID(as_uuid=True), ForeignKey(target), index=index, nullable=nullable)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class ProviderProfile(Base):
    """Chat / Embedding / Reranker provider configuration.

    API keys are stored encrypted (``api_key_encrypted``); the plaintext is
    never persisted and never logged.
    """

    __tablename__ = "provider_profiles"
    __table_args__ = (
        Index("ix_provider_profiles_kind_is_default", "kind", "is_default"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    """'chat' | 'embedding' | 'reranker'"""
    provider_type: Mapped[str] = mapped_column(String(30), nullable=False)
    """'openai' | 'gemini_native' | 'gemini_compat' | 'openai_compat' | 'vllm'"""
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    base_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    api_key_encrypted: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")  # type: ignore[type-arg]
    context_window: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = _now_col()
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # back-references
    chats_as_chat: Mapped[list[Chat]] = relationship(
        "Chat",
        foreign_keys="Chat.default_chat_profile_id",
        back_populates="default_chat_profile",
    )
    chats_as_embedding: Mapped[list[Chat]] = relationship(
        "Chat",
        foreign_keys="Chat.default_embedding_profile_id",
        back_populates="default_embedding_profile",
    )
    chats_as_reranker: Mapped[list[Chat]] = relationship(
        "Chat",
        foreign_keys="Chat.default_reranker_profile_id",
        back_populates="default_reranker_profile",
    )


class Chat(Base):
    """Top-level document isolation boundary."""

    __tablename__ = "chats"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    default_chat_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("provider_profiles.id"),
        nullable=True,
        index=True,
    )
    default_embedding_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("provider_profiles.id"),
        nullable=True,
        index=True,
    )
    default_reranker_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("provider_profiles.id"),
        nullable=True,
        index=True,
    )

    created_at: Mapped[datetime] = _now_col()
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # relationships
    default_chat_profile: Mapped[ProviderProfile | None] = relationship(
        "ProviderProfile",
        foreign_keys=[default_chat_profile_id],
        back_populates="chats_as_chat",
    )
    default_embedding_profile: Mapped[ProviderProfile | None] = relationship(
        "ProviderProfile",
        foreign_keys=[default_embedding_profile_id],
        back_populates="chats_as_embedding",
    )
    default_reranker_profile: Mapped[ProviderProfile | None] = relationship(
        "ProviderProfile",
        foreign_keys=[default_reranker_profile_id],
        back_populates="chats_as_reranker",
    )
    sessions: Mapped[list[Session]] = relationship(
        "Session", back_populates="chat", cascade="all, delete-orphan"
    )
    documents: Mapped[list[Document]] = relationship(
        "Document", back_populates="chat", cascade="all, delete-orphan"
    )
    chat_documents: Mapped[list[ChatDocument]] = relationship(
        "ChatDocument", back_populates="chat", cascade="all, delete-orphan"
    )


class Session(Base):
    """One conversation thread under a Chat.

    A Session shares Chat documents but never shares message history across
    other Sessions.
    """

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    chat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chats.id", ondelete="CASCADE"), index=True, nullable=False
    )
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    chat_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("provider_profiles.id"),
        nullable=True,
        index=True,
    )
    selected_document_ids: Mapped[list | None] = mapped_column(JSONB, nullable=True)  # type: ignore[type-arg]
    """Session-level QA document scope. Null means all chat documents."""
    document_scope_locked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    """Locked after the first QA so the right-rail checkbox scope is stable."""
    created_at: Mapped[datetime] = _now_col()
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # relationships
    chat: Mapped[Chat] = relationship("Chat", back_populates="sessions")
    chat_profile: Mapped[ProviderProfile | None] = relationship("ProviderProfile")
    messages: Mapped[list[Message]] = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan"
    )


class Message(Base):
    """A single turn in a Session conversation."""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    """'user' | 'assistant' | 'system' | 'tool'"""
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list | None] = mapped_column(JSONB, nullable=True)  # type: ignore[type-arg]
    """Serialised list[Citation] — structured in domain.py."""
    tool_trace: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # type: ignore[type-arg]
    """Serialised ToolTrace — structured in domain.py."""
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = _now_col()

    # relationships
    session: Mapped[Session] = relationship("Session", back_populates="messages")


class Document(Base):
    """An uploaded or ingested document, always scoped to a Chat."""

    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_chat_status", "chat_id", "status"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    chat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)
    """'upload' | 'arxiv' | 'url'"""
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="uploaded")
    """'uploaded' | 'parsing' | 'parsed' | 'enriching' | 'indexed' | 'failed'"""
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = _now_col()
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # relationships
    chat: Mapped[Chat] = relationship("Chat", back_populates="documents")
    nodes: Mapped[list[DocumentNode]] = relationship(
        "DocumentNode", back_populates="document", cascade="all, delete-orphan"
    )
    summaries: Mapped[list[Summary]] = relationship(
        "Summary", back_populates="document", cascade="all, delete-orphan"
    )
    structured_facts: Mapped[list[StructuredFact]] = relationship(
        "StructuredFact", back_populates="document", cascade="all, delete-orphan"
    )
    ingestion_jobs: Mapped[list[IngestionJob]] = relationship(
        "IngestionJob", back_populates="document", cascade="all, delete-orphan"
    )
    chat_documents: Mapped[list[ChatDocument]] = relationship(
        "ChatDocument", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentNode(Base):
    """Parsed structural node (section, paragraph, figure, table, etc.)."""

    __tablename__ = "document_nodes"
    __table_args__ = (
        Index("ix_document_nodes_doc_order", "document_id", "order_index"),
        Index("ix_document_nodes_chat_id", "chat_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    chat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_nodes.id"),
        nullable=True,
        index=True,
    )
    node_type: Mapped[str] = mapped_column(String(30), nullable=False)
    """'document' | 'section' | 'subsection' | 'paragraph' |
    'figure' | 'table' | 'equation' | 'reference'"""
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # type: ignore[type-arg]
    metadata_: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        "metadata", JSONB, nullable=False, server_default="{}"
    )
    created_at: Mapped[datetime] = _now_col()

    # relationships
    document: Mapped[Document] = relationship("Document", back_populates="nodes")
    children: Mapped[list[DocumentNode]] = relationship(
        "DocumentNode", back_populates="parent"
    )
    parent: Mapped[DocumentNode | None] = relationship(
        "DocumentNode", back_populates="children", remote_side="DocumentNode.id"
    )
    summaries: Mapped[list[Summary]] = relationship(
        "Summary", back_populates="source_node"
    )
    structured_facts: Mapped[list[StructuredFact]] = relationship(
        "StructuredFact", back_populates="source_node"
    )


class Summary(Base):
    """Generated summary of a document or a node, scoped to a Chat."""

    __tablename__ = "summaries"
    __table_args__ = (
        Index("ix_summaries_chat_document", "chat_id", "document_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    chat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    source_node_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_nodes.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(50), nullable=False)
    """e.g. 'section_detailed' | 'section_compact' | 'document_overview' | ..."""
    content: Mapped[str] = mapped_column(Text, nullable=False)
    keywords: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="[]")  # type: ignore[type-arg]
    entities: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="[]")  # type: ignore[type-arg]
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = _now_col()

    # relationships
    document: Mapped[Document] = relationship("Document", back_populates="summaries")
    source_node: Mapped[DocumentNode | None] = relationship(
        "DocumentNode", back_populates="summaries"
    )


class StructuredFact(Base):
    """A single structured fact extracted from a document (metric, benchmark, etc.)."""

    __tablename__ = "structured_facts"
    __table_args__ = (
        Index("ix_structured_facts_chat_document", "chat_id", "document_id"),
        Index("ix_structured_facts_chat_kind", "chat_id", "kind"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    chat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    source_node_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_nodes.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(50), nullable=False)
    """'metric' | 'benchmark' | 'dataset' | 'hyperparameter' | ..."""
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False)  # type: ignore[type-arg]
    unit: Mapped[str | None] = mapped_column(String(100), nullable=True)
    context_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = _now_col()

    # relationships
    document: Mapped[Document] = relationship("Document", back_populates="structured_facts")
    source_node: Mapped[DocumentNode | None] = relationship(
        "DocumentNode", back_populates="structured_facts"
    )


class IngestionJob(Base):
    """Idempotent, retryable ingestion job for a document."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    chat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    """'pending' | 'running' | 'succeeded' | 'failed'"""
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = _now_col()

    # relationships
    document: Mapped[Document] = relationship("Document", back_populates="ingestion_jobs")


class ChatDocument(Base):
    """Association table allowing cross-chat document reuse.

    Primary key is composite (chat_id, document_id).
    Deletion of a Chat or Document cascades here automatically.
    """

    __tablename__ = "chat_documents"

    chat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        primary_key=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    )

    # relationships
    chat: Mapped[Chat] = relationship("Chat", back_populates="chat_documents")
    document: Mapped[Document] = relationship("Document", back_populates="chat_documents")


# ---------------------------------------------------------------------------
# Convenience: expose all ORM classes at module level for alembic autogenerate
# ---------------------------------------------------------------------------

__all__ = [
    "Base",
    "ProviderProfile",
    "Chat",
    "Session",
    "Message",
    "Document",
    "DocumentNode",
    "Summary",
    "StructuredFact",
    "IngestionJob",
    "ChatDocument",
]
