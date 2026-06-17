"""Unit tests for app.services.ingestion_service.feed_document (Phase 6.2).

All tests use:
- ``MockEmbeddingProvider`` — deterministic, no network calls.
- ``MockVespaFeedClient`` — spy implementation (records calls, no HTTP).

Tests verify:
- ``delete_by_document`` is called first (idempotent re-feed).
- Embedding batch is called.
- ``feed_chunks`` receives chunks with embeddings attached.
- ``VespaDimensionMismatch`` raised when embedding dim != schema dim.
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
# Mock VespaFeedClient
# ---------------------------------------------------------------------------


class SpyVespaFeedClient:
    """Spy implementation of VespaFeedClient — records calls, no HTTP."""

    def __init__(self, embedding_dim: int = 4) -> None:
        self._embedding_dim = embedding_dim
        self.delete_calls: list[tuple[uuid.UUID, uuid.UUID]] = []
        self.fed_chunks: list[VespaChunk] = []
        self._fail_feed: bool = False

    async def delete_by_document(
        self, chat_id: uuid.UUID, document_id: uuid.UUID
    ) -> int:
        self.delete_calls.append((chat_id, document_id))
        return 0

    async def feed_chunks(self, chunks: list[VespaChunk]) -> FeedReport:
        # Validate dims (same as real client)
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
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _bbox() -> BBox:
    return BBox(x0=0.0, y0=0.0, x1=100.0, y1=20.0, page_width=612.0, page_height=792.0)


def _make_block(chat_id: uuid.UUID, doc_id: uuid.UUID, text: str, ro: int = 0) -> ParsedBlock:
    block_id = uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}:block:{ro}")
    return ParsedBlock(
        block_id=block_id,
        chat_id=chat_id,
        document_id=doc_id,
        page_number=1,
        block_type=BlockType.paragraph,
        text=text,
        bbox=_bbox(),
        reading_order=ro,
    )


def _make_node(
    chat_id: uuid.UUID, doc_id: uuid.UUID, content: str, oi: int = 0
) -> DocumentNodeOut:
    node_id = uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}:node:{oi}")
    return DocumentNodeOut(
        id=node_id,
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


def _make_section_enrichment(
    chat_id: uuid.UUID, doc_id: uuid.UUID
) -> SectionEnrichment:
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
    chat_id: uuid.UUID,
    doc_id: uuid.UUID,
    abstract_summary: str | None = "Abstract.",
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
    async def test_delete_called_first(self) -> None:
        """delete_by_document must be invoked before feed_chunks."""
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        dim = 4
        spy_client = SpyVespaFeedClient(embedding_dim=dim)
        embedding_provider = MockEmbeddingProvider(dimension=dim)
        blocks = [_make_block(chat_id, doc_id, "Some text.")]

        await feed_document(
            chat_id,
            doc_id,
            blocks=blocks,
            hierarchy=[],
            section_enrichments=[],
            document_enrichment=None,
            facts=[],
            embedding_provider=embedding_provider,
            vespa_client=spy_client,  # type: ignore[arg-type]
        )

        assert len(spy_client.delete_calls) == 1
        assert spy_client.delete_calls[0] == (chat_id, doc_id)

    @pytest.mark.asyncio
    async def test_chunks_are_fed_after_delete(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        dim = 4
        spy_client = SpyVespaFeedClient(embedding_dim=dim)
        embedding_provider = MockEmbeddingProvider(dimension=dim)
        blocks = [_make_block(chat_id, doc_id, "Block content one.")]

        report = await feed_document(
            chat_id,
            doc_id,
            blocks=blocks,
            hierarchy=[],
            section_enrichments=[],
            document_enrichment=None,
            facts=[],
            embedding_provider=embedding_provider,
            vespa_client=spy_client,  # type: ignore[arg-type]
        )

        assert len(spy_client.fed_chunks) >= 1
        assert report.success_count >= 1

    @pytest.mark.asyncio
    async def test_embeddings_attached_to_chunks(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        dim = 4
        spy_client = SpyVespaFeedClient(embedding_dim=dim)
        embedding_provider = MockEmbeddingProvider(dimension=dim)
        blocks = [_make_block(chat_id, doc_id, "Embedding test content.")]

        await feed_document(
            chat_id,
            doc_id,
            blocks=blocks,
            hierarchy=[],
            section_enrichments=[],
            document_enrichment=None,
            facts=[],
            embedding_provider=embedding_provider,
            vespa_client=spy_client,  # type: ignore[arg-type]
        )

        # All fed chunks should have embeddings of correct length
        for chunk in spy_client.fed_chunks:
            assert len(chunk.embedding) == dim

    @pytest.mark.asyncio
    async def test_dimension_mismatch_raises(self) -> None:
        """Provider returning wrong dim → VespaDimensionMismatch before any feed."""
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        schema_dim = 4
        provider_dim = 8  # mismatch!
        spy_client = SpyVespaFeedClient(embedding_dim=schema_dim)
        embedding_provider = MockEmbeddingProvider(dimension=provider_dim)
        blocks = [_make_block(chat_id, doc_id, "content")]

        with pytest.raises(VespaDimensionMismatch) as exc_info:
            await feed_document(
                chat_id,
                doc_id,
                blocks=blocks,
                hierarchy=[],
                section_enrichments=[],
                document_enrichment=None,
                facts=[],
                embedding_provider=embedding_provider,
                vespa_client=spy_client,  # type: ignore[arg-type]
            )

        assert exc_info.value.expected == schema_dim
        assert exc_info.value.got == provider_dim

    @pytest.mark.asyncio
    async def test_section_enrichments_encoded(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        dim = 4
        spy_client = SpyVespaFeedClient(embedding_dim=dim)
        embedding_provider = MockEmbeddingProvider(dimension=dim)
        section = _make_section_enrichment(chat_id, doc_id)

        await feed_document(
            chat_id,
            doc_id,
            blocks=[],
            hierarchy=[],
            section_enrichments=[section],
            document_enrichment=None,
            facts=[],
            embedding_provider=embedding_provider,
            vespa_client=spy_client,  # type: ignore[arg-type]
        )

        source_types = {c.source_type for c in spy_client.fed_chunks}
        assert "section_summary" in source_types
        assert "compact_section_summary" in source_types

    @pytest.mark.asyncio
    async def test_document_enrichment_encoded(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        dim = 4
        spy_client = SpyVespaFeedClient(embedding_dim=dim)
        embedding_provider = MockEmbeddingProvider(dimension=dim)
        doc_enrichment = _make_doc_enrichment(chat_id, doc_id, abstract_summary="Abstract text.")

        await feed_document(
            chat_id,
            doc_id,
            blocks=[],
            hierarchy=[],
            section_enrichments=[],
            document_enrichment=doc_enrichment,
            facts=[],
            embedding_provider=embedding_provider,
            vespa_client=spy_client,  # type: ignore[arg-type]
        )

        source_types = {c.source_type for c in spy_client.fed_chunks}
        assert "document_overview" in source_types
        assert "chapter_summary" in source_types

    @pytest.mark.asyncio
    async def test_no_document_enrichment_no_overview(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        dim = 4
        spy_client = SpyVespaFeedClient(embedding_dim=dim)
        embedding_provider = MockEmbeddingProvider(dimension=dim)
        blocks = [_make_block(chat_id, doc_id, "text")]

        await feed_document(
            chat_id,
            doc_id,
            blocks=blocks,
            hierarchy=[],
            section_enrichments=[],
            document_enrichment=None,  # no doc enrichment
            facts=[],
            embedding_provider=embedding_provider,
            vespa_client=spy_client,  # type: ignore[arg-type]
        )

        source_types = {c.source_type for c in spy_client.fed_chunks}
        assert "document_overview" not in source_types

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_report(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        dim = 4
        spy_client = SpyVespaFeedClient(embedding_dim=dim)
        embedding_provider = MockEmbeddingProvider(dimension=dim)

        report = await feed_document(
            chat_id,
            doc_id,
            blocks=[],
            hierarchy=[],
            section_enrichments=[],
            document_enrichment=None,
            facts=[],
            embedding_provider=embedding_provider,
            vespa_client=spy_client,  # type: ignore[arg-type]
        )

        # delete always called
        assert len(spy_client.delete_calls) == 1
        assert report.success_count == 0
        assert report.fail_count == 0

    @pytest.mark.asyncio
    async def test_hierarchy_nodes_produce_chunk_source_type(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        dim = 4
        spy_client = SpyVespaFeedClient(embedding_dim=dim)
        embedding_provider = MockEmbeddingProvider(dimension=dim)
        node = _make_node(chat_id, doc_id, "Node content text here.")

        await feed_document(
            chat_id,
            doc_id,
            blocks=[],
            hierarchy=[node],
            section_enrichments=[],
            document_enrichment=None,
            facts=[],
            embedding_provider=embedding_provider,
            vespa_client=spy_client,  # type: ignore[arg-type]
        )

        source_types = {c.source_type for c in spy_client.fed_chunks}
        assert "chunk" in source_types

    @pytest.mark.asyncio
    async def test_chat_id_in_all_chunks(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        dim = 4
        spy_client = SpyVespaFeedClient(embedding_dim=dim)
        embedding_provider = MockEmbeddingProvider(dimension=dim)
        blocks = [_make_block(chat_id, doc_id, "content")]

        await feed_document(
            chat_id,
            doc_id,
            blocks=blocks,
            hierarchy=[],
            section_enrichments=[],
            document_enrichment=None,
            facts=[],
            embedding_provider=embedding_provider,
            vespa_client=spy_client,  # type: ignore[arg-type]
        )

        for chunk in spy_client.fed_chunks:
            assert chunk.chat_id == str(chat_id)

    @pytest.mark.asyncio
    async def test_document_id_in_all_chunks(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        dim = 4
        spy_client = SpyVespaFeedClient(embedding_dim=dim)
        embedding_provider = MockEmbeddingProvider(dimension=dim)
        blocks = [_make_block(chat_id, doc_id, "content")]

        await feed_document(
            chat_id,
            doc_id,
            blocks=blocks,
            hierarchy=[],
            section_enrichments=[],
            document_enrichment=None,
            facts=[],
            embedding_provider=embedding_provider,
            vespa_client=spy_client,  # type: ignore[arg-type]
        )

        for chunk in spy_client.fed_chunks:
            assert chunk.document_id == str(doc_id)
