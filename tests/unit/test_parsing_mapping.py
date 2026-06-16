"""Unit tests for app.parsing.mapping and app.parsing.models.

Coverage
--------
- happy-path: map synthetic middle.json → correct block count, type distribution,
  reading_order continuity.
- deterministic block_id: two calls with same inputs produce identical block_id lists.
- page 1-indexed: page_idx=0 → ParsedBlock.page_number=1.
- image caption fold: nested image_caption is NOT emitted as a separate block;
  the outer image block's image.caption is populated.
- equation_latex: interline_equation block populates equation_latex correctly.
- discarded blocks: discarded_blocks are emitted with block_type=discarded.
- title_level: levels 1 and 2 are mapped correctly.
- real middle.json smoke test (skipped when file absent).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.parsing.mapping import (
    ParsingError,
    extract_text_from_block,
    load_middle_json,
    map_middle_to_parsed_blocks,
)
from app.parsing.models import BlockType, ParsedBlock

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "mineru_sample"
SYNTHETIC_MIDDLE = FIXTURE_DIR / "middle.json"

REAL_MIDDLE = (
    Path(__file__).parent.parent.parent
    / "data"
    / "parsed"
    / "2410.05779v3"
    / "hybrid_auto"
    / "2410.05779v3_middle.json"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_middle() -> dict:
    return json.loads(SYNTHETIC_MIDDLE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fixed_ids() -> tuple[UUID, UUID]:
    """Stable UUIDs used across determinism tests."""
    chat_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    document_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    return chat_id, document_id


@pytest.fixture(scope="module")
def blocks(synthetic_middle: dict, fixed_ids: tuple[UUID, UUID]) -> list[ParsedBlock]:
    chat_id, document_id = fixed_ids
    return map_middle_to_parsed_blocks(
        synthetic_middle, chat_id=chat_id, document_id=document_id
    )


# ---------------------------------------------------------------------------
# Happy-path: block count and type distribution
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_total_block_count(self, blocks: list[ParsedBlock]) -> None:
        """Synthetic fixture has 4 preproc + 1 discarded on page 1,
        3 preproc on page 2 = 8 total."""
        # page 1: title(1) + paragraph(1) + equation(1) + image(1) + discarded(1) = 5
        # page 2: title(1) + ref_text(1) + table(1) = 3
        assert len(blocks) == 8

    def test_type_distribution(self, blocks: list[ParsedBlock]) -> None:
        type_counts = Counter(b.block_type for b in blocks)
        assert type_counts[BlockType.title] == 2
        assert type_counts[BlockType.paragraph] == 1
        assert type_counts[BlockType.equation] == 1
        assert type_counts[BlockType.image] == 1
        assert type_counts[BlockType.ref_text] == 1
        assert type_counts[BlockType.table] == 1
        assert type_counts[BlockType.discarded] == 1

    def test_reading_order_unique_within_page(self, blocks: list[ParsedBlock]) -> None:
        """reading_order values within each page (for non-discarded) must be unique."""
        from collections import defaultdict

        by_page: dict[int, list[int]] = defaultdict(list)
        for b in blocks:
            if b.block_type != BlockType.discarded:
                by_page[b.page_number].append(b.reading_order)
        for page_num, orders in by_page.items():
            assert len(orders) == len(set(orders)), (
                f"Duplicate reading_order on page {page_num}: {sorted(orders)}"
            )

    def test_no_image_caption_block_emitted(self, blocks: list[ParsedBlock]) -> None:
        """image_caption sub-block must be folded into parent; not emitted separately."""
        types = [b.block_type for b in blocks]
        assert BlockType.image_caption not in types

    def test_no_table_caption_block_emitted(self, blocks: list[ParsedBlock]) -> None:
        """table_caption sub-block must be folded into parent; not emitted separately."""
        types = [b.block_type for b in blocks]
        assert BlockType.table_caption not in types


# ---------------------------------------------------------------------------
# Deterministic block_id
# ---------------------------------------------------------------------------


class TestDeterministicBlockId:
    def test_two_calls_same_ids(self, synthetic_middle: dict, fixed_ids: tuple[UUID, UUID]) -> None:
        chat_id, document_id = fixed_ids
        blocks_a = map_middle_to_parsed_blocks(
            synthetic_middle, chat_id=chat_id, document_id=document_id
        )
        blocks_b = map_middle_to_parsed_blocks(
            synthetic_middle, chat_id=chat_id, document_id=document_id
        )
        ids_a = [b.block_id for b in blocks_a]
        ids_b = [b.block_id for b in blocks_b]
        assert ids_a == ids_b

    def test_different_document_id_gives_different_block_ids(
        self, synthetic_middle: dict, fixed_ids: tuple[UUID, UUID]
    ) -> None:
        chat_id, document_id = fixed_ids
        other_doc = uuid4()
        blocks_a = map_middle_to_parsed_blocks(
            synthetic_middle, chat_id=chat_id, document_id=document_id
        )
        blocks_b = map_middle_to_parsed_blocks(
            synthetic_middle, chat_id=chat_id, document_id=other_doc
        )
        ids_a = set(b.block_id for b in blocks_a)
        ids_b = set(b.block_id for b in blocks_b)
        # Different document_id → no shared block_ids
        assert ids_a.isdisjoint(ids_b)

    def test_all_block_ids_unique(self, blocks: list[ParsedBlock]) -> None:
        ids = [b.block_id for b in blocks]
        assert len(ids) == len(set(ids)), "Duplicate block_id values found"


# ---------------------------------------------------------------------------
# Page 1-indexed
# ---------------------------------------------------------------------------


class TestPageNumbering:
    def test_page_idx_0_maps_to_page_number_1(self, blocks: list[ParsedBlock]) -> None:
        page1_blocks = [b for b in blocks if b.page_number == 1]
        assert len(page1_blocks) > 0, "No blocks on page 1"

    def test_page_idx_1_maps_to_page_number_2(self, blocks: list[ParsedBlock]) -> None:
        page2_blocks = [b for b in blocks if b.page_number == 2]
        assert len(page2_blocks) > 0, "No blocks on page 2"

    def test_no_page_number_zero(self, blocks: list[ParsedBlock]) -> None:
        assert all(b.page_number >= 1 for b in blocks)

    def test_page_numbers_are_1_and_2(self, blocks: list[ParsedBlock]) -> None:
        pages = {b.page_number for b in blocks}
        assert pages == {1, 2}


# ---------------------------------------------------------------------------
# Image caption fold-in
# ---------------------------------------------------------------------------


class TestImageCaptionFolding:
    def test_image_block_has_caption(self, blocks: list[ParsedBlock]) -> None:
        image_blocks = [b for b in blocks if b.block_type == BlockType.image]
        assert len(image_blocks) == 1
        img = image_blocks[0]
        assert img.image is not None
        assert img.image.caption is not None
        assert "Figure 1" in img.image.caption

    def test_image_block_has_image_path(self, blocks: list[ParsedBlock]) -> None:
        image_blocks = [b for b in blocks if b.block_type == BlockType.image]
        img = image_blocks[0]
        assert img.image is not None
        assert img.image.image_path == "synthetic_p1_ccdd5678.jpg"

    def test_image_text_field_is_empty(self, blocks: list[ParsedBlock]) -> None:
        image_blocks = [b for b in blocks if b.block_type == BlockType.image]
        assert image_blocks[0].text == ""


# ---------------------------------------------------------------------------
# Table caption fold-in
# ---------------------------------------------------------------------------


class TestTableCaptionFolding:
    def test_table_block_has_caption(self, blocks: list[ParsedBlock]) -> None:
        table_blocks = [b for b in blocks if b.block_type == BlockType.table]
        assert len(table_blocks) == 1
        tbl = table_blocks[0]
        assert tbl.table is not None
        assert tbl.table.caption is not None
        assert "Table 1" in tbl.table.caption

    def test_table_block_has_html_body(self, blocks: list[ParsedBlock]) -> None:
        table_blocks = [b for b in blocks if b.block_type == BlockType.table]
        tbl = table_blocks[0]
        assert tbl.table is not None
        assert "<table>" in tbl.table.html_body
        assert "Accuracy" in tbl.table.html_body

    def test_table_text_field_is_empty(self, blocks: list[ParsedBlock]) -> None:
        table_blocks = [b for b in blocks if b.block_type == BlockType.table]
        assert table_blocks[0].text == ""


# ---------------------------------------------------------------------------
# Equation LaTeX
# ---------------------------------------------------------------------------


class TestEquation:
    def test_equation_latex_field(self, blocks: list[ParsedBlock]) -> None:
        eq_blocks = [b for b in blocks if b.block_type == BlockType.equation]
        assert len(eq_blocks) == 1
        eq = eq_blocks[0]
        assert eq.equation_latex == "E = mc^{2}"

    def test_equation_text_field_has_display_fences(self, blocks: list[ParsedBlock]) -> None:
        eq_blocks = [b for b in blocks if b.block_type == BlockType.equation]
        eq = eq_blocks[0]
        assert eq.text.startswith("$$")
        assert "E = mc^{2}" in eq.text


# ---------------------------------------------------------------------------
# Discarded blocks
# ---------------------------------------------------------------------------


class TestDiscardedBlocks:
    def test_discarded_block_emitted(self, blocks: list[ParsedBlock]) -> None:
        discarded = [b for b in blocks if b.block_type == BlockType.discarded]
        assert len(discarded) == 1

    def test_discarded_block_page_number(self, blocks: list[ParsedBlock]) -> None:
        discarded = [b for b in blocks if b.block_type == BlockType.discarded]
        assert discarded[0].page_number == 1

    def test_discarded_block_has_text(self, blocks: list[ParsedBlock]) -> None:
        discarded = [b for b in blocks if b.block_type == BlockType.discarded]
        assert "Footnote" in discarded[0].text


# ---------------------------------------------------------------------------
# Title level
# ---------------------------------------------------------------------------


class TestTitleLevel:
    def test_doc_title_has_level_1(self, blocks: list[ParsedBlock]) -> None:
        title_blocks = [b for b in blocks if b.block_type == BlockType.title]
        level1 = [b for b in title_blocks if b.title_level == 1]
        assert len(level1) == 1
        assert "Synthetic Test Paper" in level1[0].text

    def test_section_heading_has_level_2(self, blocks: list[ParsedBlock]) -> None:
        title_blocks = [b for b in blocks if b.block_type == BlockType.title]
        level2 = [b for b in title_blocks if b.title_level == 2]
        assert len(level2) == 1
        assert "Introduction" in level2[0].text

    def test_non_title_blocks_have_no_title_level(self, blocks: list[ParsedBlock]) -> None:
        non_title = [b for b in blocks if b.block_type != BlockType.title]
        assert all(b.title_level is None for b in non_title)


# ---------------------------------------------------------------------------
# BBox
# ---------------------------------------------------------------------------


class TestBBox:
    def test_bbox_page_dimensions_populated(self, blocks: list[ParsedBlock]) -> None:
        for b in blocks:
            assert b.bbox.page_width == 612.0
            assert b.bbox.page_height == 792.0

    def test_bbox_coordinates_reasonable(self, blocks: list[ParsedBlock]) -> None:
        for b in blocks:
            assert b.bbox.x0 >= 0
            assert b.bbox.y0 >= 0
            assert b.bbox.x1 > b.bbox.x0
            assert b.bbox.y1 > b.bbox.y0


# ---------------------------------------------------------------------------
# load_middle_json helper
# ---------------------------------------------------------------------------


class TestLoadMiddleJson:
    def test_loads_valid_file(self) -> None:
        result = load_middle_json(SYNTHETIC_MIDDLE)
        assert "pdf_info" in result

    def test_raises_parsing_error_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ParsingError, match="not found"):
            load_middle_json(tmp_path / "nonexistent.json")

    def test_raises_parsing_error_for_invalid_json(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{ not valid json }", encoding="utf-8")
        with pytest.raises(ParsingError, match="not valid JSON"):
            load_middle_json(bad)


# ---------------------------------------------------------------------------
# extract_text_from_block helper
# ---------------------------------------------------------------------------


class TestExtractTextFromBlock:
    def test_joins_span_contents(self) -> None:
        block = {
            "lines": [
                {"spans": [{"type": "text", "content": "Hello"}]},
                {"spans": [{"type": "text", "content": "World"}]},
            ]
        }
        result = extract_text_from_block(block)
        assert result == "Hello World"

    def test_skips_spans_without_content(self) -> None:
        block = {
            "lines": [
                {
                    "spans": [
                        {"type": "image", "image_path": "foo.jpg"},
                        {"type": "text", "content": "Caption"},
                    ]
                }
            ]
        }
        result = extract_text_from_block(block)
        assert result == "Caption"

    def test_empty_block_returns_empty_string(self) -> None:
        assert extract_text_from_block({}) == ""
        assert extract_text_from_block({"lines": []}) == ""


# ---------------------------------------------------------------------------
# Smoke test against real middle.json (skipped when absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not REAL_MIDDLE.exists(),
    reason=f"Real middle.json not available at {REAL_MIDDLE}",
)
class TestRealMiddleJsonSmoke:
    def test_does_not_raise(self) -> None:
        middle = json.loads(REAL_MIDDLE.read_text(encoding="utf-8"))
        blocks = map_middle_to_parsed_blocks(
            middle, chat_id=uuid4(), document_id=uuid4()
        )
        assert blocks  # non-empty

    def test_block_count_above_threshold(self) -> None:
        middle = json.loads(REAL_MIDDLE.read_text(encoding="utf-8"))
        blocks = map_middle_to_parsed_blocks(
            middle, chat_id=uuid4(), document_id=uuid4()
        )
        assert len(blocks) > 100, f"Expected >100 blocks, got {len(blocks)}"

    def test_has_title_blocks(self) -> None:
        middle = json.loads(REAL_MIDDLE.read_text(encoding="utf-8"))
        blocks = map_middle_to_parsed_blocks(
            middle, chat_id=uuid4(), document_id=uuid4()
        )
        titles = [b for b in blocks if b.block_type == BlockType.title]
        assert len(titles) >= 1, "Expected at least 1 title block"

    def test_has_ref_text_blocks(self) -> None:
        middle = json.loads(REAL_MIDDLE.read_text(encoding="utf-8"))
        blocks = map_middle_to_parsed_blocks(
            middle, chat_id=uuid4(), document_id=uuid4()
        )
        refs = [b for b in blocks if b.block_type == BlockType.ref_text]
        assert len(refs) >= 5, f"Expected >=5 ref_text blocks, got {len(refs)}"

    def test_has_equation_blocks(self) -> None:
        middle = json.loads(REAL_MIDDLE.read_text(encoding="utf-8"))
        blocks = map_middle_to_parsed_blocks(
            middle, chat_id=uuid4(), document_id=uuid4()
        )
        equations = [b for b in blocks if b.block_type == BlockType.equation]
        assert len(equations) >= 1, "Expected at least 1 equation block"

    def test_doc_title_level_1_exists(self) -> None:
        middle = json.loads(REAL_MIDDLE.read_text(encoding="utf-8"))
        blocks = map_middle_to_parsed_blocks(
            middle, chat_id=uuid4(), document_id=uuid4()
        )
        doc_titles = [
            b for b in blocks if b.block_type == BlockType.title and b.title_level == 1
        ]
        assert len(doc_titles) >= 1, "Expected at least 1 doc-title (title_level=1)"
