"""Unit tests for app.parsing.mapping and app.parsing.models.

Kept invariants
---------------
- happy-path: block count, type distribution, reading_order uniqueness per page,
  no stray caption blocks, title_level mapping (levels 1 and 2).
- image/table caption fold-in: parent fields populated, text field empty.
- equation_latex field and display fences.
- discarded blocks emitted with correct type/page/text.
- deterministic uuid5 block_ids + page numbers 1-indexed.
- load_middle_json: valid load and two error paths.
- extract_text_from_block helper.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_middle() -> dict:
    return json.loads(SYNTHETIC_MIDDLE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fixed_ids() -> tuple[UUID, UUID]:
    return (
        UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
    )


@pytest.fixture(scope="module")
def blocks(synthetic_middle: dict, fixed_ids: tuple[UUID, UUID]) -> list[ParsedBlock]:
    chat_id, document_id = fixed_ids
    return map_middle_to_parsed_blocks(synthetic_middle, chat_id=chat_id, document_id=document_id)


# ---------------------------------------------------------------------------
# 1. Happy-path: count, type distribution, reading_order, no stray caption blocks,
#    title_level mapping
# ---------------------------------------------------------------------------


def test_happy_path(blocks: list[ParsedBlock]) -> None:
    """Total block count, type distribution, reading_order unique per page,
    no image_caption/table_caption blocks; title levels 1 and 2 map correctly."""
    assert len(blocks) == 8

    tc = Counter(b.block_type for b in blocks)
    assert tc[BlockType.title] == 2
    assert tc[BlockType.paragraph] == 1
    assert tc[BlockType.equation] == 1
    assert tc[BlockType.image] == 1
    assert tc[BlockType.ref_text] == 1
    assert tc[BlockType.table] == 1
    assert tc[BlockType.discarded] == 1

    by_page: dict[int, list[int]] = defaultdict(list)
    for b in blocks:
        if b.block_type != BlockType.discarded:
            by_page[b.page_number].append(b.reading_order)
    for page_num, orders in by_page.items():
        assert len(orders) == len(set(orders)), f"Duplicate reading_order on page {page_num}"

    emitted_types = {b.block_type for b in blocks}
    assert BlockType.image_caption not in emitted_types
    assert BlockType.table_caption not in emitted_types

    # Title level mapping
    title_blocks = [b for b in blocks if b.block_type == BlockType.title]
    level1 = [b for b in title_blocks if b.title_level == 1]
    level2 = [b for b in title_blocks if b.title_level == 2]
    assert any("Synthetic Test Paper" in b.text for b in level1)
    assert any("Introduction" in b.text for b in level2)
    assert all(b.title_level is None for b in blocks if b.block_type != BlockType.title)


# ---------------------------------------------------------------------------
# 2. Caption fold-in: image and table (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block_type, caption_text, body_check",
    [
        (BlockType.image, "Figure 1", None),
        (BlockType.table, "Table 1", "<table>"),
    ],
)
def test_caption_fold_in(
    blocks: list[ParsedBlock],
    block_type: BlockType,
    caption_text: str,
    body_check: str | None,
) -> None:
    """Caption sub-block is folded into parent; text field is empty."""
    matching = [b for b in blocks if b.block_type == block_type]
    assert len(matching) == 1
    blk = matching[0]
    assert blk.text == ""

    if block_type == BlockType.image:
        assert blk.image is not None
        assert blk.image.caption is not None
        assert caption_text in blk.image.caption
        assert blk.image.image_path == "synthetic_p1_ccdd5678.jpg"
    else:
        assert blk.table is not None
        assert blk.table.caption is not None
        assert caption_text in blk.table.caption
        assert body_check is not None and body_check in blk.table.html_body


# ---------------------------------------------------------------------------
# 3. Equation LaTeX + display fences; discarded block
# ---------------------------------------------------------------------------


def test_equation_and_discarded(blocks: list[ParsedBlock]) -> None:
    """Equation carries latex + display fences; discarded block on page 1 with text."""
    eq_blocks = [b for b in blocks if b.block_type == BlockType.equation]
    assert len(eq_blocks) == 1
    eq = eq_blocks[0]
    assert eq.equation_latex == "E = mc^{2}"
    assert eq.text.startswith("$$") and "E = mc^{2}" in eq.text

    discarded = [b for b in blocks if b.block_type == BlockType.discarded]
    assert len(discarded) == 1
    assert discarded[0].page_number == 1
    assert "Footnote" in discarded[0].text


# ---------------------------------------------------------------------------
# 4. Deterministic uuid5 block_ids + page 1-indexed (no zero)
# ---------------------------------------------------------------------------


def test_deterministic_ids_and_page_numbering(
    synthetic_middle: dict, fixed_ids: tuple[UUID, UUID], blocks: list[ParsedBlock]
) -> None:
    """Same inputs → same block_ids; different document_id → disjoint; pages ≥ 1."""
    chat_id, document_id = fixed_ids
    blocks_b = map_middle_to_parsed_blocks(synthetic_middle, chat_id=chat_id, document_id=document_id)
    assert [b.block_id for b in blocks] == [b.block_id for b in blocks_b]

    blocks_c = map_middle_to_parsed_blocks(
        synthetic_middle, chat_id=chat_id, document_id=uuid4()
    )
    assert {b.block_id for b in blocks}.isdisjoint({b.block_id for b in blocks_c})

    assert all(b.page_number >= 1 for b in blocks)
    assert {b.page_number for b in blocks} == {1, 2}


# ---------------------------------------------------------------------------
# 5. load_middle_json: valid load + two error paths
# ---------------------------------------------------------------------------


def test_load_middle_json_valid() -> None:
    result = load_middle_json(SYNTHETIC_MIDDLE)
    assert "pdf_info" in result


@pytest.mark.parametrize(
    "scenario, match",
    [("missing_file", "not found"), ("invalid_json", "not valid JSON")],
)
def test_load_middle_json_errors(tmp_path: Path, scenario: str, match: str) -> None:
    if scenario == "missing_file":
        with pytest.raises(ParsingError, match=match):
            load_middle_json(tmp_path / "nonexistent.json")
    else:
        bad = tmp_path / "bad.json"
        bad.write_text("{ not valid json }", encoding="utf-8")
        with pytest.raises(ParsingError, match=match):
            load_middle_json(bad)


# ---------------------------------------------------------------------------
# 6. extract_text_from_block helper
# ---------------------------------------------------------------------------


def test_extract_text_from_block() -> None:
    """Joins span contents; skips non-content spans; empty block returns ''."""
    block = {
        "lines": [
            {"spans": [{"type": "text", "content": "Hello"}]},
            {"spans": [{"type": "image", "image_path": "foo.jpg"}, {"type": "text", "content": "World"}]},
        ]
    }
    assert extract_text_from_block(block) == "Hello World"
    assert extract_text_from_block({}) == ""
    assert extract_text_from_block({"lines": []}) == ""
