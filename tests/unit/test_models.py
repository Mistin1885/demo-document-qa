"""Unit tests for ORM and domain models.

These tests do NOT require a running database — they only verify:
1. All ORM classes import cleanly.
2. All domain model classes import cleanly.
3. Pydantic v2 round-trip works (create → dict → read).
4. Citation schema matches the CLAUDE.md §13 contract.
5. get_session is importable without triggering a DB connection.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

# ---------------------------------------------------------------------------
# 1. ORM import test
# ---------------------------------------------------------------------------


def test_orm_imports() -> None:
    """All ORM table classes must be importable at module level."""
    from app.models.orm import (  # noqa: F401
        Base,
        Chat,
        ChatDocument,
        Document,
        DocumentNode,
        IngestionJob,
        Message,
        ProviderProfile,
        Session,
        StructuredFact,
        Summary,
    )

    # Ensure they are all proper SQLAlchemy mapped classes (have __table__)
    for cls in (
        Chat,
        ChatDocument,
        Document,
        DocumentNode,
        IngestionJob,
        Message,
        ProviderProfile,
        Session,
        StructuredFact,
        Summary,
    ):
        assert hasattr(cls, "__table__"), f"{cls.__name__} has no __table__"


# ---------------------------------------------------------------------------
# 2. Domain model import test
# ---------------------------------------------------------------------------


def test_domain_imports() -> None:
    """All domain model classes must be importable at module level."""
    from app.models.domain import (  # noqa: F401
        BBox,
        ChatCreate,
        ChatDocumentCreate,
        ChatDocumentRead,
        ChatRead,
        ChatUpdate,
        Citation,
        DocumentCreate,
        DocumentNodeCreate,
        DocumentNodeRead,
        DocumentNodeUpdate,
        DocumentRead,
        DocumentUpdate,
        FactValue,
        IngestionJobCreate,
        IngestionJobRead,
        IngestionJobUpdate,
        MessageCreate,
        MessageRead,
        MessageUpdate,
        NodeMetadata,
        ProviderProfileCreate,
        ProviderProfileRead,
        ProviderProfileUpdate,
        QAResponse,
        SessionCreate,
        SessionRead,
        SessionUpdate,
        StructuredFactCreate,
        StructuredFactRead,
        StructuredFactUpdate,
        SummaryCreate,
        SummaryRead,
        SummaryUpdate,
        ToolTrace,
        ToolTraceStep,
    )

    assert ChatRead is not None
    assert Citation is not None


# ---------------------------------------------------------------------------
# 3. Pydantic round-trip: Chat
# ---------------------------------------------------------------------------


def test_chat_create_round_trip() -> None:
    """ChatCreate → dict → ChatRead.model_validate should round-trip cleanly."""
    from app.models.domain import ChatCreate, ChatRead

    create_data = ChatCreate(
        name="My Notebook",
        description="Testing chat",
    )
    now = datetime.now(tz=UTC)
    chat_id = uuid.uuid4()

    # Simulate the dict that the DB layer would return (including id/timestamps)
    db_dict = {
        "id": chat_id,
        "name": create_data.name,
        "description": create_data.description,
        "default_chat_profile_id": None,
        "default_embedding_profile_id": None,
        "default_reranker_profile_id": None,
        "created_at": now,
        "updated_at": now,
    }

    read_model = ChatRead.model_validate(db_dict)
    assert read_model.id == chat_id
    assert read_model.name == "My Notebook"
    assert read_model.description == "Testing chat"
    assert read_model.default_chat_profile_id is None


# ---------------------------------------------------------------------------
# 4. Citation schema test (CLAUDE.md §13)
# ---------------------------------------------------------------------------


def test_citation_schema() -> None:
    """Citation model must have all fields mandated by CLAUDE.md §13."""
    from app.models.domain import Citation

    chat_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    node_id = uuid.uuid4()

    citation = Citation(
        chat_id=chat_id,
        document_id=doc_id,
        document_name="attention_is_all_you_need.pdf",
        page_start=3,
        page_end=5,
        section_title="Attention Mechanism",
        source_node_id=node_id,
        excerpt="The Transformer model relies entirely on an attention mechanism.",
    )

    assert citation.chat_id == chat_id
    assert citation.document_id == doc_id
    assert citation.page_start == 3
    assert citation.page_end == 5
    assert citation.source_node_id == node_id
    assert "attention mechanism" in citation.excerpt.lower()

    # Serialise and re-parse
    as_dict = citation.model_dump()
    restored = Citation.model_validate(as_dict)
    assert restored.chat_id == chat_id
    assert restored.excerpt == citation.excerpt


def test_citation_requires_mandatory_fields() -> None:
    """Citation without mandatory fields should raise ValidationError."""
    from pydantic import ValidationError

    from app.models.domain import Citation

    with pytest.raises(ValidationError):
        # Missing chat_id, document_id, document_name, page_start, page_end, excerpt
        Citation()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 5. get_session importable without DB connection
# ---------------------------------------------------------------------------


def test_db_import_no_connection() -> None:
    """Importing get_session must NOT trigger a database connection."""
    from app.db import get_session  # noqa: F401

    # Simply importing should not raise any connection errors.
    assert get_session is not None


# ---------------------------------------------------------------------------
# 6. QAResponse round-trip (CLAUDE.md §13)
# ---------------------------------------------------------------------------


def test_qa_response_round_trip() -> None:
    """QAResponse serialisation / deserialisation must preserve all fields."""
    from app.models.domain import Citation, QAResponse, ToolTrace, ToolTraceStep

    session_id = uuid.uuid4()
    message_id = uuid.uuid4()
    chat_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    response = QAResponse(
        answer="The Transformer architecture relies on self-attention.",
        citations=[
            Citation(
                chat_id=chat_id,
                document_id=doc_id,
                document_name="vaswani2017.pdf",
                page_start=2,
                page_end=4,
                excerpt="We propose a new simple network architecture, the Transformer.",
            )
        ],
        documents_used=[doc_id],
        coverage=0.85,
        uncertainty=["Page 6 not included in retrieved chunks."],
        session_id=session_id,
        message_id=message_id,
        debug_trace=ToolTrace(
            steps=[ToolTraceStep(tool_name="search_hybrid", status="ok", token_estimate=512)],
            total_rounds=1,
        ),
    )

    as_dict = response.model_dump()
    restored = QAResponse.model_validate(as_dict)
    assert restored.answer == response.answer
    assert len(restored.citations) == 1
    assert restored.citations[0].chat_id == chat_id
    assert restored.coverage == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# 7. ORM table names check
# ---------------------------------------------------------------------------


def test_orm_table_names() -> None:
    """Every ORM class must have the expected table name."""
    from app.models.orm import (
        Chat,
        ChatDocument,
        Document,
        DocumentNode,
        IngestionJob,
        Message,
        ProviderProfile,
        Session,
        StructuredFact,
        Summary,
    )

    expected = {
        Chat: "chats",
        Session: "sessions",
        Message: "messages",
        Document: "documents",
        DocumentNode: "document_nodes",
        Summary: "summaries",
        StructuredFact: "structured_facts",
        ProviderProfile: "provider_profiles",
        IngestionJob: "ingestion_jobs",
        ChatDocument: "chat_documents",
    }
    for cls, table_name in expected.items():
        assert cls.__tablename__ == table_name, (
            f"Expected {cls.__name__}.__tablename__ == {table_name!r}, "
            f"got {cls.__tablename__!r}"
        )


# ---------------------------------------------------------------------------
# 8. chat_id indexed on all document-scoped tables
# ---------------------------------------------------------------------------


def test_chat_id_indexed() -> None:
    """chat_id must be indexed on every document-scoped table (CLAUDE.md §5.1)."""
    from sqlalchemy import inspect as sa_inspect

    from app.models.orm import (
        Document,
        DocumentNode,
        IngestionJob,
        Session,
        StructuredFact,
        Summary,
    )

    for cls in (Document, DocumentNode, IngestionJob, Session, StructuredFact, Summary):
        mapper = sa_inspect(cls)
        indexed_cols: set[str] = set()

        # columns with index=True
        for col in mapper.mapper.columns:
            if col.index or col.primary_key:
                indexed_cols.add(col.name)

        # explicit Table.indexes
        for idx in cls.__table__.indexes:
            for col in idx.columns:
                indexed_cols.add(col.name)

        assert "chat_id" in indexed_cols, (
            f"{cls.__name__}.chat_id must be indexed (CLAUDE.md §5.1)"
        )
