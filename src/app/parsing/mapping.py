"""Map a post-processed ``<doc>_middle.json`` dict to a list of ParsedBlock.

Public API
----------
- ``map_middle_to_parsed_blocks(middle, *, chat_id, document_id)``
  Pure function — no I/O, no database writes.
- ``load_middle_json(path)``
  Thin helper: read + JSON-parse; raise ``ParsingError`` on failure.
- ``extract_text_from_block(block)``
  Join ``lines[].spans[].content``; exported for Phase 4.3 reuse.

All three are re-exported from ``app.parsing.__init__``.

Mapping rules
-------------
- Walk ``pdf_info[i].preproc_blocks`` in ``index`` order (fall back to
  enumerate order if ``index`` is absent).
- ``image`` / ``table`` blocks fold their sub-block text (caption, footnote,
  body html) into the parent ``ParsedBlock.image`` / ``.table``; sub-blocks
  are NOT emitted as separate ``ParsedBlock`` instances.
- ``discarded_blocks`` are also emitted (``block_type=discarded``) so that
  Phase 4.3 / enrichment layers may optionally use them.
- ``block_id`` is deterministic: ``uuid5(NAMESPACE_OID,
  "<document_id>:<page_idx>:<reading_order>:<type>")``.

See: CLAUDE.md §6.3, §6.4; ``deploy/mineru/output-schema.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_OID, UUID, uuid5

from app.errors import AppError
from app.parsing.models import (
    BBox,
    BlockType,
    ImageRef,
    ParsedBlock,
    TableRef,
)

# ---------------------------------------------------------------------------
# Domain error
# ---------------------------------------------------------------------------


class ParsingError(AppError):
    """Raised when middle.json cannot be loaded or is structurally invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_middle_json(path: Path) -> dict[str, Any]:
    """Read and JSON-parse *path*; raise ``ParsingError`` on any failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ParsingError(f"middle.json not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ParsingError(f"middle.json is not valid JSON ({path}): {exc}") from exc


def extract_text_from_block(block: dict[str, Any]) -> str:
    """Return plain text by joining ``lines[].spans[].content``.

    Behaviour:
    - Concatenate *all* span ``content`` values with a single space.
    - ``inline_equation`` and ``interline_equation`` spans are included as-is
      (they carry LaTeX wrapped in ``$...$`` or plain LaTeX; we keep them so
      that the text field contains a readable representation).
    - Spans without a ``content`` key (e.g. ``image`` spans in
      ``image_body``) are skipped.

    This function is exported for Phase 4.3 reuse.
    """
    parts: list[str] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            content = span.get("content")
            if isinstance(content, str) and content:
                parts.append(content)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Internal per-block-type helpers
# ---------------------------------------------------------------------------


def _make_block_id(document_id: UUID, page_idx: int, reading_order: int, raw_type: str) -> UUID:
    """Return a deterministic UUID-5 for this block."""
    name = f"{document_id}:{page_idx}:{reading_order}:{raw_type}"
    return uuid5(NAMESPACE_OID, name)


def _make_bbox(block: dict[str, Any], page_size: list[float]) -> BBox:
    bbox = block.get("bbox", [0.0, 0.0, 0.0, 0.0])
    pw = float(page_size[0]) if len(page_size) > 1 else 612.0
    ph = float(page_size[1]) if len(page_size) > 1 else 792.0
    return BBox(
        x0=float(bbox[0]),
        y0=float(bbox[1]),
        x1=float(bbox[2]),
        y1=float(bbox[3]),
        page_width=pw,
        page_height=ph,
    )


def _min_span_score(block: dict[str, Any]) -> float | None:
    """Return the minimum ``score`` across all spans in a block, or ``None``."""
    scores: list[float] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            s = span.get("score")
            if isinstance(s, (int, float)):
                scores.append(float(s))
    return min(scores) if scores else None


def _extract_sub_text(sub_block: dict[str, Any]) -> str:
    """Extract text from a sub-block (image_caption, table_caption, etc.)."""
    return extract_text_from_block(sub_block)


def _map_text_or_title_block(
    block: dict[str, Any],
    raw_type: str,
    page_idx: int,
    reading_order: int,
    page_size: list[float],
    chat_id: UUID,
    document_id: UUID,
) -> ParsedBlock:
    """Map a ``text`` or ``title`` block (including ``ref_text``)."""
    # Determine our BlockType and title_level
    title_level: int | None = None

    if raw_type == "title":
        # MinerU "title" blocks always carry a "level" key
        lvl = block.get("level")
        if lvl == 1:
            block_type = BlockType.title
            title_level = 1
        else:
            # level 2 (or any other value) → section/subsection heading
            block_type = BlockType.title
            title_level = 2
    elif raw_type == "text":
        # MinerU "text" blocks may optionally have "text_level"
        # (the output-schema.md calls this "level" too; real data uses "level"
        # only on "title" type; "text" blocks may have neither)
        lvl = block.get("level") or block.get("text_level")
        if lvl == 1:
            block_type = BlockType.title
            title_level = 1
        elif lvl == 2:
            block_type = BlockType.title
            title_level = 2
        else:
            block_type = BlockType.paragraph
    elif raw_type == "ref_text":
        block_type = BlockType.ref_text
    else:
        # Fallback — shouldn't reach here via normal code path
        block_type = BlockType.paragraph

    text = extract_text_from_block(block)
    confidence = _min_span_score(block)

    return ParsedBlock(
        block_id=_make_block_id(document_id, page_idx, reading_order, raw_type),
        chat_id=chat_id,
        document_id=document_id,
        page_number=page_idx + 1,
        block_type=block_type,
        text=text,
        title_level=title_level,
        bbox=_make_bbox(block, page_size),
        reading_order=reading_order,
        confidence=confidence,
    )


def _map_equation_block(
    block: dict[str, Any],
    page_idx: int,
    reading_order: int,
    page_size: list[float],
    chat_id: UUID,
    document_id: UUID,
) -> ParsedBlock:
    """Map a MinerU ``interline_equation`` block."""
    # The LaTeX is in the first span's ``content``
    latex: str = ""
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            content = span.get("content")
            if isinstance(content, str) and content:
                latex = content
                break
        if latex:
            break

    # Render text field as display-math fences (mirrors PoC markdown)
    text = f"$$\n{latex}\n$$" if latex else ""
    confidence = _min_span_score(block)

    return ParsedBlock(
        block_id=_make_block_id(document_id, page_idx, reading_order, "interline_equation"),
        chat_id=chat_id,
        document_id=document_id,
        page_number=page_idx + 1,
        block_type=BlockType.equation,
        text=text,
        equation_latex=latex or None,
        bbox=_make_bbox(block, page_size),
        reading_order=reading_order,
        confidence=confidence,
    )


def _map_image_block(
    block: dict[str, Any],
    page_idx: int,
    reading_order: int,
    page_size: list[float],
    chat_id: UUID,
    document_id: UUID,
) -> ParsedBlock:
    """Map a MinerU ``image`` block, folding caption/footnote sub-blocks."""
    image_path: str = ""
    caption: str | None = None
    footnote: str | None = None

    for sub in block.get("blocks", []):
        sub_type = sub.get("type", "")
        if sub_type == "image_body":
            # image_path lives in the span
            for line in sub.get("lines", []):
                for span in line.get("spans", []):
                    ip = span.get("image_path")
                    if isinstance(ip, str) and ip:
                        image_path = ip
                        break
                if image_path:
                    break
        elif sub_type == "image_caption":
            caption = _extract_sub_text(sub) or None
        elif sub_type == "image_footnote":
            footnote = _extract_sub_text(sub) or None

    image_ref = ImageRef(
        image_path=image_path,
        caption=caption,
        footnote=footnote,
    )

    return ParsedBlock(
        block_id=_make_block_id(document_id, page_idx, reading_order, "image"),
        chat_id=chat_id,
        document_id=document_id,
        page_number=page_idx + 1,
        block_type=BlockType.image,
        text="",
        bbox=_make_bbox(block, page_size),
        reading_order=reading_order,
        image=image_ref,
        confidence=None,  # image_body spans carry no "score" field
    )


def _map_table_block(
    block: dict[str, Any],
    page_idx: int,
    reading_order: int,
    page_size: list[float],
    chat_id: UUID,
    document_id: UUID,
) -> ParsedBlock:
    """Map a MinerU ``table`` block, folding caption/footnote sub-blocks."""
    html_body: str = ""
    image_path: str = ""
    caption: str | None = None
    footnote: str | None = None

    for sub in block.get("blocks", []):
        sub_type = sub.get("type", "")
        if sub_type == "table_body":
            # HTML lives in the "html" field of the table-type span; many
            # MinerU builds also expose the table crop's image_path on the
            # same span so the VLM can reconstruct rows the html misses.
            for line in sub.get("lines", []):
                for span in line.get("spans", []):
                    html = span.get("html")
                    if isinstance(html, str) and html and not html_body:
                        html_body = html
                    ip = span.get("image_path")
                    if isinstance(ip, str) and ip and not image_path:
                        image_path = ip
        elif sub_type == "table_caption":
            caption = _extract_sub_text(sub) or None
        elif sub_type == "table_footnote":
            footnote = _extract_sub_text(sub) or None

    table_ref = TableRef(
        html_body=html_body,
        image_path=image_path,
        caption=caption,
        footnote=footnote,
    )

    return ParsedBlock(
        block_id=_make_block_id(document_id, page_idx, reading_order, "table"),
        chat_id=chat_id,
        document_id=document_id,
        page_number=page_idx + 1,
        block_type=BlockType.table,
        text="",
        bbox=_make_bbox(block, page_size),
        reading_order=reading_order,
        table=table_ref,
        confidence=None,  # table spans carry no "score" field
    )


def _map_generic_block(
    block: dict[str, Any],
    raw_type: str,
    page_idx: int,
    reading_order: int,
    page_size: list[float],
    chat_id: UUID,
    document_id: UUID,
) -> ParsedBlock:
    """Map a top-level caption/footnote or unknown block to its BlockType."""
    # Top-level image_caption / image_footnote / table_caption / table_footnote
    # that appear outside their parent container are mapped to their own type.
    _type_map: dict[str, BlockType] = {
        "image_caption": BlockType.image_caption,
        "image_footnote": BlockType.image_footnote,
        "table_caption": BlockType.table_caption,
        "table_footnote": BlockType.table_footnote,
    }
    block_type = _type_map.get(raw_type, BlockType.discarded)
    text = extract_text_from_block(block)
    confidence = _min_span_score(block)

    return ParsedBlock(
        block_id=_make_block_id(document_id, page_idx, reading_order, raw_type),
        chat_id=chat_id,
        document_id=document_id,
        page_number=page_idx + 1,
        block_type=block_type,
        text=text,
        bbox=_make_bbox(block, page_size),
        reading_order=reading_order,
        confidence=confidence,
    )


def _map_discarded_block(
    block: dict[str, Any],
    page_idx: int,
    reading_order: int,
    page_size: list[float],
    chat_id: UUID,
    document_id: UUID,
) -> ParsedBlock:
    """Map a MinerU discarded block (page_footnote / aside_text / page_number)."""
    text = extract_text_from_block(block)
    confidence = _min_span_score(block)
    raw_type = block.get("type", "unknown")

    return ParsedBlock(
        block_id=_make_block_id(document_id, page_idx, reading_order, f"discarded:{raw_type}"),
        chat_id=chat_id,
        document_id=document_id,
        page_number=page_idx + 1,
        block_type=BlockType.discarded,
        text=text,
        bbox=_make_bbox(block, page_size),
        reading_order=reading_order,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Per-block dispatcher
# ---------------------------------------------------------------------------

_TEXT_LIKE_TYPES = frozenset({"text", "title", "ref_text"})


def _map_preproc_block(
    block: dict[str, Any],
    page_idx: int,
    reading_order: int,
    page_size: list[float],
    chat_id: UUID,
    document_id: UUID,
) -> ParsedBlock:
    """Dispatch a single ``preproc_blocks`` entry to the correct mapper."""
    raw_type: str = block.get("type", "unknown")

    if raw_type in _TEXT_LIKE_TYPES:
        return _map_text_or_title_block(
            block, raw_type, page_idx, reading_order, page_size, chat_id, document_id
        )
    if raw_type == "interline_equation":
        return _map_equation_block(block, page_idx, reading_order, page_size, chat_id, document_id)
    if raw_type == "image":
        return _map_image_block(block, page_idx, reading_order, page_size, chat_id, document_id)
    if raw_type == "table":
        return _map_table_block(block, page_idx, reading_order, page_size, chat_id, document_id)
    if raw_type in {"image_caption", "image_footnote", "table_caption", "table_footnote"}:
        # Top-level occurrence (not nested inside image/table parent)
        return _map_generic_block(
            block, raw_type, page_idx, reading_order, page_size, chat_id, document_id
        )
    # Unknown type → discarded with original type in block_id namespace
    return _map_generic_block(
        block, raw_type, page_idx, reading_order, page_size, chat_id, document_id
    )


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def map_middle_to_parsed_blocks(
    middle: dict[str, Any],
    *,
    chat_id: UUID,
    document_id: UUID,
) -> list[ParsedBlock]:
    """Convert a post-processed middle.json dict to an ordered list of ParsedBlock.

    Parameters
    ----------
    middle:
        The parsed content of ``<doc>_middle.json``.  Must contain a
        ``pdf_info`` list at the top level.
    chat_id:
        Injected by the ingestion service; never derived from MinerU output.
    document_id:
        Injected by the ingestion service; never derived from MinerU output.

    Returns
    -------
    list[ParsedBlock]
        Blocks ordered by (page_number, reading_order).  Discarded blocks
        from each page are appended after that page's main blocks.

    Notes
    -----
    - ``block_id`` is deterministic (UUID-5) — repeated calls with the same
      ``middle`` + ``document_id`` produce identical IDs.
    - The function is a pure function: no I/O, no database writes, no side
      effects.
    - ``image`` / ``table`` sub-blocks are folded into the parent block; they
      are not emitted as separate ``ParsedBlock`` instances.
    """
    result: list[ParsedBlock] = []

    pdf_info: list[dict[str, Any]] = middle.get("pdf_info", [])

    for page in pdf_info:
        page_idx: int = int(page.get("page_idx", 0))
        page_size: list[float] = [float(x) for x in page.get("page_size", [612.0, 792.0])]

        # ----------------------------------------------------------------
        # preproc_blocks — sort by MinerU "index" field (reading order)
        # ----------------------------------------------------------------
        preproc: list[dict[str, Any]] = page.get("preproc_blocks", [])

        # Sort by "index" if present on the block; fall back to the list
        # position for blocks that lack the field.
        def _sort_key(item: tuple[int, dict[str, Any]]) -> int:
            enum_pos, blk = item
            idx = blk.get("index")
            return int(idx) if isinstance(idx, int) else enum_pos

        sorted_blocks = sorted(enumerate(preproc), key=_sort_key)

        for enum_pos, blk in sorted_blocks:
            reading_order = blk.get("index", enum_pos)
            parsed = _map_preproc_block(
                blk, page_idx, reading_order, page_size, chat_id, document_id
            )
            result.append(parsed)

        # ----------------------------------------------------------------
        # discarded_blocks — appended after main blocks for this page
        # ----------------------------------------------------------------
        discarded: list[dict[str, Any]] = page.get("discarded_blocks", [])
        # reading_order for discarded: use a high offset so they don't collide
        # with preproc indices (preproc indices are per-page but we still want
        # unique block_ids within the document_id namespace)
        discarded_offset = 10_000
        for d_pos, dblk in enumerate(discarded):
            d_order = discarded_offset + d_pos
            parsed = _map_discarded_block(dblk, page_idx, d_order, page_size, chat_id, document_id)
            result.append(parsed)

    return result


__all__ = [
    "ParsingError",
    "load_middle_json",
    "extract_text_from_block",
    "map_middle_to_parsed_blocks",
]
