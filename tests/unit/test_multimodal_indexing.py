"""Loop validation for image/table/formula indexing (≤10 tests)."""

from __future__ import annotations

import uuid

import pytest

from app.parsing.models import BBox, BlockType, ImageRef, ParsedBlock, TableRef
from app.providers.base import EmbeddingProvider, ProviderTestResult
from app.services.ingestion_service import feed_document
from app.vespa.feed import FeedReport, VespaChunk

_CHAT_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_DOC_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def _bbox() -> BBox:
    return BBox(x0=0, y0=0, x1=10, y1=10, page_width=100, page_height=100)


def _block(block_type: BlockType, text: str, order: int) -> ParsedBlock:
    return ParsedBlock(
        block_id=uuid.uuid5(uuid.NAMESPACE_OID, f"{_DOC_ID}:{block_type}:{order}"),
        chat_id=_CHAT_ID,
        document_id=_DOC_ID,
        page_number=order + 1,
        block_type=block_type,
        text=text,
        bbox=_bbox(),
        reading_order=order,
        image=ImageRef(image_path="fig.png", caption=text) if block_type == BlockType.image else None,
        table=TableRef(html_body="<table><tr><td>95.1</td></tr></table>", caption=text)
        if block_type == BlockType.table
        else None,
        equation_latex=text if block_type == BlockType.equation else None,
    )


class _Embedding(EmbeddingProvider):
    def __init__(self) -> None:
        self.texts: list[str] = []

    @property
    def dimension(self) -> int:
        return 3

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts = texts
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def test_connection(self) -> ProviderTestResult:
        return ProviderTestResult(ok=True, model="fake-embedding")


class _Vespa:
    def __init__(self) -> None:
        self._embedding_dim = 3
        self.chunks: list[VespaChunk] = []
        self.deleted: tuple[uuid.UUID, uuid.UUID] | None = None

    async def delete_by_document(self, chat_id: uuid.UUID, document_id: uuid.UUID) -> FeedReport:
        self.deleted = (chat_id, document_id)
        return FeedReport()

    async def feed_chunks(self, chunks: list[VespaChunk]) -> FeedReport:
        self.chunks = chunks
        return FeedReport(success_count=len(chunks))


@pytest.mark.asyncio
async def test_image_table_formula_blocks_are_chunked_and_embedded() -> None:
    blocks = [
        _block(BlockType.image, "Figure 2 shows the latency curve.", 0),
        _block(BlockType.table, "Table 1 reports F1 scores.", 1),
        _block(BlockType.equation, r"F1 = 2PR / (P + R)", 2),
    ]
    embedding = _Embedding()
    vespa = _Vespa()

    report = await feed_document(
        _CHAT_ID,
        _DOC_ID,
        blocks=blocks,
        hierarchy=[],
        section_enrichments=[],
        document_enrichment=None,
        facts=[],
        embedding_provider=embedding,
        vespa_client=vespa,  # type: ignore[arg-type]
    )

    assert report.success_count == 3
    contents = "\n".join(c.content for c in vespa.chunks)
    assert "latency curve" in contents
    assert "F1 scores" in contents
    assert "F1 = 2PR" in contents
    assert embedding.texts == [c.content for c in vespa.chunks]
    assert all(c.embedding == [1.0, 0.0, 0.0] for c in vespa.chunks)
