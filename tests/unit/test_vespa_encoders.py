"""Unit tests for app.vespa.encoders (Phase 6.2).

All tests are deterministic — no network calls, no DB, no embeddings.
Tests cover:
- encode_raw_blocks: mixed block types, heading tracking, skips discarded.
- encode_chunks_from_section: long text → multiple chunks, order_index.
- encode_section_summary: always 2 chunks, correct source_types.
- encode_document_overview: 1 or 2 chunks (abstract_summary=None → 1).
- encode_structured_facts: fact.kind → source_type mapping.
- Deterministic vespa_document_id: same inputs → same IDs.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from app.enrichment.models import (
    DocumentEnrichment,
    SectionEnrichment,
)
from app.parsing.models import BBox, BlockType, DocumentNodeOut, NodeType, ParsedBlock
from app.vespa.encoders import (
    encode_chunks_from_section,
    encode_document_overview,
    encode_raw_blocks,
    encode_section_summary,
    encode_structured_facts,
)
from app.vespa.feed import make_vespa_id

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 4


def _bbox() -> BBox:
    return BBox(x0=0.0, y0=0.0, x1=100.0, y1=20.0, page_width=612.0, page_height=792.0)


def _make_text_block(
    chat_id: uuid.UUID,
    doc_id: uuid.UUID,
    text: str,
    block_type: BlockType = BlockType.paragraph,
    reading_order: int = 0,
    page_number: int = 1,
    title_level: int | None = None,
) -> ParsedBlock:
    block_id = uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}:test:{reading_order}:{block_type}")
    return ParsedBlock(
        block_id=block_id,
        chat_id=chat_id,
        document_id=doc_id,
        page_number=page_number,
        block_type=block_type,
        text=text,
        title_level=title_level,
        bbox=_bbox(),
        reading_order=reading_order,
        image=None,
        table=None,
        equation_latex=None,
        confidence=None,
    )


def _make_node(
    chat_id: uuid.UUID,
    doc_id: uuid.UUID,
    content: str,
    node_type: NodeType = NodeType.section,
    order_index: int = 0,
    title: str | None = "Introduction",
) -> DocumentNodeOut:
    node_id = uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}:node:{order_index}")
    return DocumentNodeOut(
        id=node_id,
        chat_id=chat_id,
        document_id=doc_id,
        parent_id=None,
        node_type=node_type,
        title=title,
        content=content,
        page_start=1,
        page_end=2,
        order_index=order_index,
        level=1,
        bbox=None,
        source_block_ids=[],
        metadata_={"source": "heuristic"},
    )


def _make_section_enrichment(
    chat_id: uuid.UUID,
    doc_id: uuid.UUID,
    node_id: uuid.UUID | None = None,
) -> SectionEnrichment:
    node_id = node_id or uuid.uuid4()
    return SectionEnrichment(
        node_id=node_id,
        chat_id=chat_id,
        document_id=doc_id,
        node_type="section",
        title="Results",
        page_start=3,
        page_end=5,
        source_block_ids=[],
        detailed_summary="This section presents the main experimental results.",
        compact_summary="Experimental results showing improvement.",
        keywords=["results", "experiment"],
        technical_keywords=["F1", "BLEU"],
        entities=["ModelX"],
        token_count=25,
        token_count_estimate=22,
    )


def _make_doc_enrichment(
    chat_id: uuid.UUID,
    doc_id: uuid.UUID,
    abstract_summary: str | None = "A brief abstract.",
) -> DocumentEnrichment:
    return DocumentEnrichment(
        chat_id=chat_id,
        document_id=doc_id,
        document_overview="This paper proposes a new method.",
        abstract_summary=abstract_summary,
        main_contributions=["contribution 1"],
        main_methods=["method 1"],
        main_technologies=["tech 1"],
        main_findings=["finding 1"],
        main_limitations=["limitation 1"],
        main_datasets=["dataset 1"],
        main_metrics=["BLEU"],
        main_experimental_results=["result 1"],
        main_conclusions=["conclusion 1"],
        source_section_node_ids=[],
        token_count_estimate=50,
    )


def _make_structured_fact(
    chat_id: uuid.UUID,
    doc_id: uuid.UUID,
    kind: str = "metric",
    key: str = "BLEU",
    value: object = 42.5,
    unit: str | None = "%",
    page: int | None = 7,
) -> MagicMock:
    """Return a mock StructuredFact row (avoids DB dependency)."""
    fact = MagicMock()
    fact.id = uuid.uuid4()
    fact.chat_id = chat_id
    fact.document_id = doc_id
    fact.source_node_id = None
    fact.kind = kind
    fact.key = key
    fact.value = value
    fact.unit = unit
    fact.context_excerpt = "From Table 3"
    fact.page = page
    return fact


# ---------------------------------------------------------------------------
# encode_raw_blocks
# ---------------------------------------------------------------------------


class TestEncodeRawBlocks:
    def test_text_block_produces_chunk(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        blocks = [_make_text_block(chat_id, doc_id, "Hello world", reading_order=0)]
        chunks = encode_raw_blocks(blocks, chat_id, doc_id)
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world"
        assert chunks[0].source_type == "raw_block"

    def test_title_block_source_type(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        blocks = [
            _make_text_block(
                chat_id, doc_id, "Section 1", block_type=BlockType.title, reading_order=0
            )
        ]
        chunks = encode_raw_blocks(blocks, chat_id, doc_id)
        assert len(chunks) == 1
        assert chunks[0].source_type == "raw_block"
        assert chunks[0].title == "Section 1"

    def test_discarded_block_skipped(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        blocks = [
            _make_text_block(
                chat_id, doc_id, "page 1", block_type=BlockType.discarded, reading_order=0
            )
        ]
        chunks = encode_raw_blocks(blocks, chat_id, doc_id)
        assert len(chunks) == 0

    def test_image_block_uses_caption(self) -> None:
        from app.parsing.models import ImageRef

        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        block_id = uuid.uuid4()
        block = ParsedBlock(
            block_id=block_id,
            chat_id=chat_id,
            document_id=doc_id,
            page_number=1,
            block_type=BlockType.image,
            text="",
            bbox=_bbox(),
            reading_order=0,
            image=ImageRef(image_path="fig1.png", caption="Figure 1 caption"),
        )
        chunks = encode_raw_blocks([block], chat_id, doc_id)
        assert len(chunks) == 1
        assert chunks[0].content == "Figure 1 caption"

    def test_table_block_uses_caption(self) -> None:
        from app.parsing.models import TableRef

        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        block_id = uuid.uuid4()
        block = ParsedBlock(
            block_id=block_id,
            chat_id=chat_id,
            document_id=doc_id,
            page_number=1,
            block_type=BlockType.table,
            text="",
            bbox=_bbox(),
            reading_order=0,
            table=TableRef(html_body="<table></table>", caption="Table 1 caption"),
        )
        chunks = encode_raw_blocks([block], chat_id, doc_id)
        assert len(chunks) == 1
        assert chunks[0].content == "Table 1 caption"

    def test_heading_path_updated_by_title_block(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        blocks = [
            _make_text_block(
                chat_id, doc_id, "Section 2", block_type=BlockType.title, reading_order=0
            ),
            _make_text_block(chat_id, doc_id, "Body text.", reading_order=1),
        ]
        chunks = encode_raw_blocks(blocks, chat_id, doc_id)
        # Both should exist (title + paragraph)
        assert len(chunks) == 2
        # The paragraph's heading_path should be the section title
        paragraph_chunk = next(c for c in chunks if c.content == "Body text.")
        assert paragraph_chunk.heading_path == "Section 2"

    def test_page_range_from_block(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        blocks = [_make_text_block(chat_id, doc_id, "p3 text", page_number=3)]
        chunks = encode_raw_blocks(blocks, chat_id, doc_id)
        assert chunks[0].page_start == 3
        assert chunks[0].page_end == 3

    def test_chat_id_document_id_set_correctly(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        blocks = [_make_text_block(chat_id, doc_id, "text")]
        chunks = encode_raw_blocks(blocks, chat_id, doc_id)
        assert chunks[0].chat_id == str(chat_id)
        assert chunks[0].document_id == str(doc_id)

    def test_empty_content_block_skipped(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        blocks = [_make_text_block(chat_id, doc_id, "   ")]
        chunks = encode_raw_blocks(blocks, chat_id, doc_id)
        assert len(chunks) == 0


# ---------------------------------------------------------------------------
# encode_chunks_from_section
# ---------------------------------------------------------------------------


class TestEncodeChunksFromSection:
    def test_short_content_single_chunk(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        node = _make_node(chat_id, doc_id, "Short text here.", order_index=1)
        chunks = encode_chunks_from_section(node, chat_id, doc_id)
        assert len(chunks) >= 1
        assert chunks[0].source_type == "chunk"

    def test_long_content_multiple_chunks(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        # Generate text that exceeds 256 tokens (~342 words * 0.75 = ~256 tokens)
        long_sentence = "This is an important sentence about neural networks. "
        long_content = (long_sentence * 200).strip()  # ~200 sentences
        node = _make_node(chat_id, doc_id, long_content, order_index=0)
        chunks = encode_chunks_from_section(node, chat_id, doc_id, target_tokens=256)
        assert len(chunks) >= 2

    def test_order_index_increases_across_chunks(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        long_sentence = "This is a sentence about deep learning methods. "
        long_content = (long_sentence * 200).strip()
        node = _make_node(chat_id, doc_id, long_content, order_index=5)
        chunks = encode_chunks_from_section(node, chat_id, doc_id, target_tokens=256)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            assert chunks[i].order_index < chunks[i + 1].order_index

    def test_heading_path_from_node_title(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        node = _make_node(chat_id, doc_id, "Some content.", title="Methods")
        chunks = encode_chunks_from_section(node, chat_id, doc_id)
        assert all(c.heading_path == "Methods" for c in chunks)

    def test_empty_content_returns_empty(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        node = _make_node(chat_id, doc_id, "")
        chunks = encode_chunks_from_section(node, chat_id, doc_id)
        assert len(chunks) == 0

    def test_source_type_is_chunk(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        node = _make_node(chat_id, doc_id, "Some relevant section content.")
        chunks = encode_chunks_from_section(node, chat_id, doc_id)
        assert all(c.source_type == "chunk" for c in chunks)


# ---------------------------------------------------------------------------
# encode_section_summary
# ---------------------------------------------------------------------------


class TestEncodeSectionSummary:
    def test_returns_exactly_two_chunks(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        section = _make_section_enrichment(chat_id, doc_id)
        chunks = encode_section_summary(section, chat_id, doc_id)
        assert len(chunks) == 2

    def test_first_chunk_is_section_summary(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        section = _make_section_enrichment(chat_id, doc_id)
        chunks = encode_section_summary(section, chat_id, doc_id)
        assert chunks[0].source_type == "section_summary"
        assert chunks[0].content == section.detailed_summary

    def test_second_chunk_is_compact(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        section = _make_section_enrichment(chat_id, doc_id)
        chunks = encode_section_summary(section, chat_id, doc_id)
        assert chunks[1].source_type == "compact_section_summary"
        assert chunks[1].content == section.compact_summary

    def test_keywords_propagated(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        section = _make_section_enrichment(chat_id, doc_id)
        chunks = encode_section_summary(section, chat_id, doc_id)
        assert chunks[0].keywords == section.keywords
        assert chunks[0].technical_keywords == section.technical_keywords

    def test_page_range_from_section(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        section = _make_section_enrichment(chat_id, doc_id)
        chunks = encode_section_summary(section, chat_id, doc_id)
        for c in chunks:
            assert c.page_start == section.page_start
            assert c.page_end == section.page_end


# ---------------------------------------------------------------------------
# encode_document_overview
# ---------------------------------------------------------------------------


class TestEncodeDocumentOverview:
    def test_with_abstract_returns_two_chunks(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        doc = _make_doc_enrichment(chat_id, doc_id, abstract_summary="Abstract text.")
        chunks = encode_document_overview(doc, chat_id, doc_id)
        assert len(chunks) == 2

    def test_without_abstract_returns_one_chunk(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        doc = _make_doc_enrichment(chat_id, doc_id, abstract_summary=None)
        chunks = encode_document_overview(doc, chat_id, doc_id)
        assert len(chunks) == 1

    def test_first_chunk_is_document_overview(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        doc = _make_doc_enrichment(chat_id, doc_id)
        chunks = encode_document_overview(doc, chat_id, doc_id)
        assert chunks[0].source_type == "document_overview"
        assert chunks[0].content == doc.document_overview

    def test_second_chunk_is_chapter_summary(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        doc = _make_doc_enrichment(chat_id, doc_id, abstract_summary="Abstract.")
        chunks = encode_document_overview(doc, chat_id, doc_id)
        assert chunks[1].source_type == "chapter_summary"
        assert chunks[1].content == "Abstract."

    def test_chat_id_set_on_all_chunks(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        doc = _make_doc_enrichment(chat_id, doc_id)
        chunks = encode_document_overview(doc, chat_id, doc_id)
        assert all(c.chat_id == str(chat_id) for c in chunks)


# ---------------------------------------------------------------------------
# encode_structured_facts
# ---------------------------------------------------------------------------


class TestEncodeStructuredFacts:
    def test_metric_kind_maps_to_performance_fact(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        fact = _make_structured_fact(chat_id, doc_id, kind="metric")
        chunks = encode_structured_facts([fact])
        assert chunks[0].source_type == "performance_fact"

    def test_benchmark_kind_maps_to_performance_fact(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        fact = _make_structured_fact(chat_id, doc_id, kind="benchmark")
        chunks = encode_structured_facts([fact])
        assert chunks[0].source_type == "performance_fact"

    def test_dataset_kind_maps_to_performance_fact(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        fact = _make_structured_fact(chat_id, doc_id, kind="dataset")
        chunks = encode_structured_facts([fact])
        assert chunks[0].source_type == "performance_fact"

    def test_claim_kind_maps_to_claim(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        fact = _make_structured_fact(chat_id, doc_id, kind="claim")
        chunks = encode_structured_facts([fact])
        assert chunks[0].source_type == "claim"

    def test_definition_kind_maps_to_definition(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        fact = _make_structured_fact(chat_id, doc_id, kind="definition")
        chunks = encode_structured_facts([fact])
        assert chunks[0].source_type == "definition"

    def test_technology_kind_maps_to_technology_card(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        fact = _make_structured_fact(chat_id, doc_id, kind="technology")
        chunks = encode_structured_facts([fact])
        assert chunks[0].source_type == "technology_card"

    def test_other_kind_maps_to_claim(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        fact = _make_structured_fact(chat_id, doc_id, kind="other")
        chunks = encode_structured_facts([fact])
        assert chunks[0].source_type == "claim"

    def test_content_contains_key(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        fact = _make_structured_fact(chat_id, doc_id, key="BLEU", value=42.5)
        chunks = encode_structured_facts([fact])
        assert "BLEU" in chunks[0].content

    def test_empty_facts_returns_empty(self) -> None:
        chunks = encode_structured_facts([])
        assert len(chunks) == 0


# ---------------------------------------------------------------------------
# Deterministic vespa_document_id
# ---------------------------------------------------------------------------


class TestDeterministicVespaId:
    def test_raw_blocks_same_input_same_id(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        # Call encode_raw_blocks twice with the same block
        block = _make_text_block(chat_id, doc_id, "hello", reading_order=0)
        chunks_a = encode_raw_blocks([block], chat_id, doc_id)
        chunks_b = encode_raw_blocks([block], chat_id, doc_id)
        assert chunks_a[0].vespa_document_id == chunks_b[0].vespa_document_id

    def test_section_summary_same_input_same_ids(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        node_id = uuid.uuid4()
        section = _make_section_enrichment(chat_id, doc_id, node_id=node_id)
        chunks_a = encode_section_summary(section, chat_id, doc_id)
        chunks_b = encode_section_summary(section, chat_id, doc_id)
        assert chunks_a[0].vespa_document_id == chunks_b[0].vespa_document_id
        assert chunks_a[1].vespa_document_id == chunks_b[1].vespa_document_id

    def test_doc_overview_same_input_same_ids(self) -> None:
        chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
        doc = _make_doc_enrichment(chat_id, doc_id)
        chunks_a = encode_document_overview(doc, chat_id, doc_id)
        chunks_b = encode_document_overview(doc, chat_id, doc_id)
        assert chunks_a[0].vespa_document_id == chunks_b[0].vespa_document_id

    def test_make_vespa_id_same_input_multiple_times(self) -> None:
        doc_id = uuid.uuid4()
        node_id = uuid.uuid4()
        ids = [make_vespa_id(doc_id, "chunk", node_id, 0) for _ in range(10)]
        assert len(set(ids)) == 1  # all identical
