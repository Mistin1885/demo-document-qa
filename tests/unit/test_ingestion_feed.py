"""Unit tests for app.services.ingestion_service.feed_document (Phase 6.2).

Uses MockEmbeddingProvider and SpyVespaFeedClient — no network, no DB.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from app.enrichment.models import SectionEnrichment
from app.errors import VespaDimensionMismatch
from app.parsing.models import BBox, BlockType, DocumentNodeOut, NodeType, ParsedBlock
from app.providers.mock import MockEmbeddingProvider
from app.services.ingestion_service import feed_document
from app.vespa.feed import FeedReport, VespaChunk

# ---------------------------------------------------------------------------
# Spy VespaFeedClient
# ---------------------------------------------------------------------------


class SpyVespaFeedClient:
    def __init__(self, embedding_dim: int = 4) -> None:
        self._embedding_dim = embedding_dim
        self.delete_calls: list[tuple[uuid.UUID, uuid.UUID]] = []
        self.fed_chunks: list[VespaChunk] = []

    async def delete_by_document(self, chat_id: uuid.UUID, document_id: uuid.UUID) -> int:
        self.delete_calls.append((chat_id, document_id))
        return 0

    async def feed_chunks(self, chunks: list[VespaChunk]) -> FeedReport:
        from app.errors import VespaDimensionMismatch

        for chunk in chunks:
            got = len(chunk.embedding)
            if got != self._embedding_dim:
                raise VespaDimensionMismatch(expected=self._embedding_dim, got=got)
        self.fed_chunks.extend(chunks)
        return FeedReport(success_count=len(chunks), fail_count=0)

    async def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bbox() -> BBox:
    return BBox(x0=0.0, y0=0.0, x1=100.0, y1=20.0, page_width=612.0, page_height=792.0)


def _make_block(chat_id: uuid.UUID, doc_id: uuid.UUID, text: str, ro: int = 0) -> ParsedBlock:
    return ParsedBlock(
        block_id=uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}:block:{ro}"),
        chat_id=chat_id,
        document_id=doc_id,
        page_number=1,
        block_type=BlockType.paragraph,
        text=text,
        bbox=_bbox(),
        reading_order=ro,
    )


def _make_node(chat_id: uuid.UUID, doc_id: uuid.UUID, content: str, oi: int = 0) -> DocumentNodeOut:
    return DocumentNodeOut(
        id=uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}:node:{oi}"),
        chat_id=chat_id,
        document_id=doc_id,
        parent_id=None,
        node_type=NodeType.section,
        title="Introduction",
        content=content,
        page_start=1,
        page_end=2,
        order_index=oi,
        level=1,
        bbox=None,
        source_block_ids=[],
        metadata_={"source": "heuristic"},
    )


def _make_section_enrichment(chat_id: uuid.UUID, doc_id: uuid.UUID) -> SectionEnrichment:
    return SectionEnrichment(
        node_id=uuid.uuid4(),
        chat_id=chat_id,
        document_id=doc_id,
        node_type="section",
        title="Methods",
        page_start=2,
        page_end=4,
        source_block_ids=[],
        detailed_summary="Detailed methods description.",
        compact_summary="Methods summary.",
        token_count=20,
        token_count_estimate=18,
    )


def _make_doc_enrichment(
    chat_id: uuid.UUID, doc_id: uuid.UUID, abstract_summary: str | None = "Abstract."
) -> Any:
    from app.enrichment.models import DocumentEnrichment

    return DocumentEnrichment(
        chat_id=chat_id,
        document_id=doc_id,
        document_overview="Paper overview.",
        abstract_summary=abstract_summary,
        main_contributions=[],
        main_methods=[],
        main_technologies=[],
        main_findings=[],
        main_limitations=[],
        main_datasets=[],
        main_metrics=[],
        main_experimental_results=[],
        main_conclusions=[],
        source_section_node_ids=[],
        token_count_estimate=30,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFeedDocument:
    @pytest.mark.asyncio
    async def test_delete_called_first_with_correct_ids(self) -> None:
        """delete_by_document must be called before feed_chunks, with correct (chat_id, doc_id)."""
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        spy = SpyVespaFeedClient(embedding_dim=4)
        await feed_document(
            chat_id, doc_id,
            blocks=[_make_block(chat_id, doc_id, "Some text.")],
            hierarchy=[], section_enrichments=[], document_enrichment=None, facts=[],
            embedding_provider=MockEmbeddingProvider(dimension=4),
            vespa_client=spy,  # type: ignore[arg-type]
        )
        assert len(spy.delete_calls) == 1
        assert spy.delete_calls[0] == (chat_id, doc_id)

    @pytest.mark.asyncio
    async def test_chunks_fed_with_correct_embeddings_and_ids(self) -> None:
        """All chunks have correct dim embeddings and carry chat_id + document_id."""
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        dim = 4
        spy = SpyVespaFeedClient(embedding_dim=dim)
        await feed_document(
            chat_id, doc_id,
            blocks=[_make_block(chat_id, doc_id, "Block content one.")],
            hierarchy=[], section_enrichments=[], document_enrichment=None, facts=[],
            embedding_provider=MockEmbeddingProvider(dimension=dim),
            vespa_client=spy,  # type: ignore[arg-type]
        )
        assert len(spy.fed_chunks) >= 1
        for chunk in spy.fed_chunks:
            assert len(chunk.embedding) == dim
            assert chunk.chat_id == str(chat_id)
            assert chunk.document_id == str(doc_id)

    @pytest.mark.asyncio
    async def test_dimension_mismatch_raises(self) -> None:
        """Provider returning wrong dim raises VespaDimensionMismatch."""
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        spy = SpyVespaFeedClient(embedding_dim=4)
        with pytest.raises(VespaDimensionMismatch) as exc_info:
            await feed_document(
                chat_id, doc_id,
                blocks=[_make_block(chat_id, doc_id, "content")],
                hierarchy=[], section_enrichments=[], document_enrichment=None, facts=[],
                embedding_provider=MockEmbeddingProvider(dimension=8),
                vespa_client=spy,  # type: ignore[arg-type]
            )
        assert exc_info.value.expected == 4 and exc_info.value.got == 8

    @pytest.mark.asyncio
    async def test_source_types_from_enrichments(self) -> None:
        """section_enrichment → section_summary/compact_section_summary;
        doc_enrichment → document_overview/chapter_summary;
        hierarchy node → chunk."""
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        dim = 4
        spy = SpyVespaFeedClient(embedding_dim=dim)
        await feed_document(
            chat_id, doc_id,
            blocks=[],
            hierarchy=[_make_node(chat_id, doc_id, "Node content text here.")],
            section_enrichments=[_make_section_enrichment(chat_id, doc_id)],
            document_enrichment=_make_doc_enrichment(chat_id, doc_id, "Abstract text."),
            facts=[],
            embedding_provider=MockEmbeddingProvider(dimension=dim),
            vespa_client=spy,  # type: ignore[arg-type]
        )
        source_types = {c.source_type for c in spy.fed_chunks}
        assert "section_summary" in source_types
        assert "compact_section_summary" in source_types
        assert "document_overview" in source_types
        assert "chapter_summary" in source_types
        assert "chunk" in source_types

    @pytest.mark.asyncio
    async def test_empty_input_and_no_doc_enrichment(self) -> None:
        """Empty input → delete called, zero chunks; no doc_enrichment → no document_overview."""
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        dim = 4
        spy = SpyVespaFeedClient(embedding_dim=dim)
        report = await feed_document(
            chat_id, doc_id,
            blocks=[], hierarchy=[], section_enrichments=[],
            document_enrichment=None, facts=[],
            embedding_provider=MockEmbeddingProvider(dimension=dim),
            vespa_client=spy,  # type: ignore[arg-type]
        )
        assert len(spy.delete_calls) == 1
        assert report.success_count == 0 and report.fail_count == 0

        # With blocks but no doc enrichment → no document_overview
        spy2 = SpyVespaFeedClient(embedding_dim=dim)
        await feed_document(
            chat_id, doc_id,
            blocks=[_make_block(chat_id, doc_id, "text")],
            hierarchy=[], section_enrichments=[], document_enrichment=None, facts=[],
            embedding_provider=MockEmbeddingProvider(dimension=dim),
            vespa_client=spy2,  # type: ignore[arg-type]
        )
        assert "document_overview" not in {c.source_type for c in spy2.fed_chunks}
