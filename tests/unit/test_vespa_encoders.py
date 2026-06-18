"""Unit tests for app.vespa.encoders (Phase 6.2).

Deterministic — no network, no DB, no embeddings.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from app.enrichment.models import DocumentEnrichment, SectionEnrichment
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


def _bbox() -> BBox:
    return BBox(x0=0.0, y0=0.0, x1=100.0, y1=20.0, page_width=612.0, page_height=792.0)


def _make_text_block(
    chat_id: uuid.UUID,
    doc_id: uuid.UUID,
    text: str,
    block_type: BlockType = BlockType.paragraph,
    reading_order: int = 0,
    page_number: int = 1,
) -> ParsedBlock:
    block_id = uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}:test:{reading_order}:{block_type}")
    return ParsedBlock(
        block_id=block_id,
        chat_id=chat_id,
        document_id=doc_id,
        page_number=page_number,
        block_type=block_type,
        text=text,
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
    order_index: int = 0,
    title: str | None = "Introduction",
) -> DocumentNodeOut:
    node_id = uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}:node:{order_index}")
    return DocumentNodeOut(
        id=node_id,
        chat_id=chat_id,
        document_id=doc_id,
        parent_id=None,
        node_type=NodeType.section,
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
) -> MagicMock:
    fact = MagicMock()
    fact.id = uuid.uuid4()
    fact.chat_id = chat_id
    fact.document_id = doc_id
    fact.source_node_id = None
    fact.kind = kind
    fact.key = key
    fact.value = value
    fact.unit = "%"
    fact.context_excerpt = "From Table 3"
    fact.page = 7
    return fact


# ---------------------------------------------------------------------------
# encode_raw_blocks
# ---------------------------------------------------------------------------


def test_raw_blocks_basic_and_skips() -> None:
    """paragraph → raw_block; discarded → skipped; empty → skipped; page range correct."""
    chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
    blocks = [
        _make_text_block(chat_id, doc_id, "Hello world", page_number=3),
        _make_text_block(
            chat_id, doc_id, "noise", block_type=BlockType.discarded, reading_order=1
        ),
        _make_text_block(chat_id, doc_id, "   ", reading_order=2),
    ]
    chunks = encode_raw_blocks(blocks, chat_id, doc_id)
    assert len(chunks) == 1
    assert chunks[0].source_type == "raw_block"
    assert chunks[0].content == "Hello world"
    assert chunks[0].page_start == 3 and chunks[0].page_end == 3
    assert chunks[0].chat_id == str(chat_id)
    assert chunks[0].document_id == str(doc_id)


def test_raw_blocks_heading_path_propagated() -> None:
    """Title block updates heading_path for subsequent paragraph."""
    chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
    blocks = [
        _make_text_block(
            chat_id, doc_id, "Section 2", block_type=BlockType.title, reading_order=0
        ),
        _make_text_block(chat_id, doc_id, "Body text.", reading_order=1),
    ]
    chunks = encode_raw_blocks(blocks, chat_id, doc_id)
    paragraph = next(c for c in chunks if c.content == "Body text.")
    assert paragraph.heading_path == "Section 2"


def test_raw_blocks_image_and_table_use_caption() -> None:
    """Image and table blocks expose their captions as content."""
    from app.parsing.models import ImageRef, TableRef

    chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
    img_block = ParsedBlock(
        block_id=uuid.uuid4(),
        chat_id=chat_id,
        document_id=doc_id,
        page_number=1,
        block_type=BlockType.image,
        text="",
        bbox=_bbox(),
        reading_order=0,
        image=ImageRef(image_path="fig1.png", caption="Figure 1 caption"),
    )
    tbl_block = ParsedBlock(
        block_id=uuid.uuid4(),
        chat_id=chat_id,
        document_id=doc_id,
        page_number=1,
        block_type=BlockType.table,
        text="",
        bbox=_bbox(),
        reading_order=1,
        table=TableRef(html_body="<table></table>", caption="Table 1 caption"),
    )
    chunks = encode_raw_blocks([img_block, tbl_block], chat_id, doc_id)
    assert len(chunks) == 2
    assert chunks[0].content == "Figure 1 caption"
    assert chunks[1].content == "Table 1 caption"


# ---------------------------------------------------------------------------
# encode_chunks_from_section
# ---------------------------------------------------------------------------


def test_encode_chunks_from_section() -> None:
    """Short content → single chunk; long content → multiple chunks with increasing order_index."""
    chat_id, doc_id = uuid.uuid4(), uuid.uuid4()

    # Short
    node_short = _make_node(chat_id, doc_id, "Short text here.", title="Methods")
    short_chunks = encode_chunks_from_section(node_short, chat_id, doc_id)
    assert len(short_chunks) >= 1
    assert all(c.source_type == "chunk" for c in short_chunks)
    assert all(c.heading_path == "Methods" for c in short_chunks)

    # Long (force multiple chunks)
    long_content = ("Important sentence about neural networks. " * 200).strip()
    node_long = _make_node(chat_id, doc_id, long_content, order_index=5)
    long_chunks = encode_chunks_from_section(node_long, chat_id, doc_id, target_tokens=256)
    assert len(long_chunks) >= 2
    for i in range(len(long_chunks) - 1):
        assert long_chunks[i].order_index < long_chunks[i + 1].order_index

    # Empty → no chunks
    node_empty = _make_node(chat_id, doc_id, "")
    assert encode_chunks_from_section(node_empty, chat_id, doc_id) == []


# ---------------------------------------------------------------------------
# encode_section_summary
# ---------------------------------------------------------------------------


def test_encode_section_summary() -> None:
    """Always 2 chunks: section_summary (detailed) + compact_section_summary."""
    chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
    section = _make_section_enrichment(chat_id, doc_id)
    chunks = encode_section_summary(section, chat_id, doc_id)
    assert len(chunks) == 2
    assert chunks[0].source_type == "section_summary"
    assert chunks[0].content == section.detailed_summary
    assert chunks[1].source_type == "compact_section_summary"
    assert chunks[1].content == section.compact_summary
    assert chunks[0].keywords == section.keywords
    for c in chunks:
        assert c.page_start == section.page_start
        assert c.page_end == section.page_end


# ---------------------------------------------------------------------------
# encode_document_overview
# ---------------------------------------------------------------------------


def test_encode_document_overview() -> None:
    """With abstract → 2 chunks (document_overview + chapter_summary); without → 1."""
    chat_id, doc_id = uuid.uuid4(), uuid.uuid4()

    doc_with = _make_doc_enrichment(chat_id, doc_id, abstract_summary="Abstract.")
    chunks_with = encode_document_overview(doc_with, chat_id, doc_id)
    assert len(chunks_with) == 2
    assert chunks_with[0].source_type == "document_overview"
    assert chunks_with[1].source_type == "chapter_summary"
    assert chunks_with[1].content == "Abstract."
    assert all(c.chat_id == str(chat_id) for c in chunks_with)

    doc_without = _make_doc_enrichment(chat_id, doc_id, abstract_summary=None)
    chunks_without = encode_document_overview(doc_without, chat_id, doc_id)
    assert len(chunks_without) == 1
    assert chunks_without[0].source_type == "document_overview"


# ---------------------------------------------------------------------------
# encode_structured_facts — kind → source_type mapping (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,expected_source_type",
    [
        ("metric", "performance_fact"),
        ("technology", "technology_card"),
    ],
)
def test_structured_fact_kind_mapping(kind: str, expected_source_type: str) -> None:
    chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
    fact = _make_structured_fact(chat_id, doc_id, kind=kind)
    chunks = encode_structured_facts([fact])
    assert chunks[0].source_type == expected_source_type


def test_structured_facts_content_and_empty() -> None:
    """Content includes the key; empty list returns empty."""
    chat_id, doc_id = uuid.uuid4(), uuid.uuid4()
    fact = _make_structured_fact(chat_id, doc_id, key="BLEU", value=42.5)
    chunks = encode_structured_facts([fact])
    assert "BLEU" in chunks[0].content
    assert encode_structured_facts([]) == []


# ---------------------------------------------------------------------------
# Deterministic vespa_document_id (uuid5)
# ---------------------------------------------------------------------------


def test_deterministic_vespa_id() -> None:
    """Same inputs always produce the same vespa_document_id."""
    doc_id, node_id = uuid.uuid4(), uuid.uuid4()
    # make_vespa_id is deterministic
    ids = [make_vespa_id(doc_id, "chunk", node_id, 0) for _ in range(5)]
    assert len(set(ids)) == 1
    # Different order_index or source_type must differ
    assert make_vespa_id(doc_id, "chunk", node_id, 0) != make_vespa_id(doc_id, "chunk", node_id, 1)
    assert make_vespa_id(doc_id, "chunk", node_id, 0) != make_vespa_id(doc_id, "raw_block", node_id, 0)

    # encode_raw_blocks produces stable IDs across calls
    chat_id = uuid.uuid4()
    block = _make_text_block(chat_id, doc_id, "hello", reading_order=0)
    a = encode_raw_blocks([block], chat_id, doc_id)
    b = encode_raw_blocks([block], chat_id, doc_id)
    assert a[0].vespa_document_id == b[0].vespa_document_id
