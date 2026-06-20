"""Pure encoder functions: parsing/enrichment outputs → VespaChunk list.

Every function in this module is a pure transformation with no I/O.
Embeddings are NOT produced here — callers inject them after encoding.

Source types produced (CLAUDE.md §5.2):
    raw_block, chunk, section_summary, compact_section_summary,
    chapter_summary, document_overview, technology_card, claim,
    definition, performance_fact.

Design rules (CLAUDE.md §12)
------------------------------
- No ``dict[str, Any]``.
- No database I/O.
- No FastAPI imports.
- ``chat_id`` is injected by the caller; never derived here.
- Deterministic ``vespa_document_id`` via ``make_vespa_id`` (UUID-5).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from app.enrichment.models import DocumentEnrichment, SectionEnrichment
from app.models.orm import StructuredFact
from app.parsing.figure_narrator import FigureNarration
from app.parsing.models import BlockType, DocumentNodeOut, ParsedBlock
from app.vespa.feed import VespaChunk, make_vespa_id

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_WORDS_PER_TOKEN = 0.75
"""Rough heuristic: 1 token ≈ 0.75 English words (used for chunk splitting)."""

CHUNK_TARGET_TOKENS: int = 256
"""Target token count for ``encode_chunks_from_section`` chunks."""


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: word_count / 0.75 (no tiktoken dependency)."""
    return max(1, int(len(text.split()) / _WORDS_PER_TOKEN))


def _now_epoch_ms() -> int:
    return int(datetime.now(tz=UTC).timestamp() * 1000)


def _uuid_str(v: uuid.UUID | str) -> str:
    return str(v)


# ---------------------------------------------------------------------------
# encode_raw_blocks
# ---------------------------------------------------------------------------


def encode_raw_blocks(
    blocks: list[ParsedBlock],
    chat_id: uuid.UUID,
    document_id: uuid.UUID,
    *,
    heading_path: str = "",
    created_at: int | None = None,
) -> list[VespaChunk]:
    """Encode a list of ``ParsedBlock`` objects as ``raw_block`` chunks.

    - ``text`` / ``title`` / ``paragraph`` / ``equation`` / ``ref_text``
      blocks use their ``text`` field as ``content``.
    - ``image`` blocks use the image caption (or image path) as ``content``.
    - ``table`` blocks use the table caption as ``content``.
    - ``discarded`` blocks are skipped.

    The ``heading_path`` parameter is set by the caller when the blocks
    belong to a known section (title block changes the path for subsequent
    blocks).

    Parameters
    ----------
    blocks:
        ``ParsedBlock`` list from ``mapping.py``; must all share
        ``chat_id`` / ``document_id``.
    chat_id:
        Chat isolation context (injected by service layer).
    document_id:
        Parent document UUID.
    heading_path:
        Current heading context when these blocks are emitted (optional).
    created_at:
        Unix epoch milliseconds; defaults to ``now``.
    """
    ts = created_at if created_at is not None else _now_epoch_ms()
    chunks: list[VespaChunk] = []
    current_heading = heading_path

    for block in blocks:
        if block.block_type == BlockType.discarded:
            continue

        # Determine content
        if block.block_type == BlockType.title:
            content = block.text
            if content.strip():
                current_heading = content.strip()
        elif block.block_type in (
            BlockType.text,
            BlockType.paragraph,
            BlockType.equation,
            BlockType.ref_text,
        ):
            content = block.text
        elif block.block_type == BlockType.image:
            # Use caption, fallback to image path
            img = block.image
            if img and img.caption:
                content = img.caption
            elif img:
                content = img.image_path
            else:
                content = ""
        elif block.block_type in (BlockType.image_caption, BlockType.image_footnote):
            content = block.text
        elif block.block_type == BlockType.table:
            tbl = block.table
            if tbl and tbl.caption:
                content = tbl.caption
            elif tbl and tbl.html_body:
                content = tbl.html_body[:500]
            else:
                content = ""
        elif block.block_type in (BlockType.table_caption, BlockType.table_footnote):
            content = block.text
        else:
            content = block.text

        if not content.strip():
            continue

        chunk = VespaChunk(
            vespa_document_id=make_vespa_id(
                document_id,
                "raw_block",
                block.block_id,
                block.reading_order,
            ),
            chat_id=_uuid_str(chat_id),
            document_id=_uuid_str(document_id),
            source_node_id=_uuid_str(block.block_id),
            parent_node_id=None,
            source_type="raw_block",
            title=block.text[:200] if block.block_type == BlockType.title else "",
            heading_path=current_heading,
            content=content,
            page_start=block.page_number,
            page_end=block.page_number,
            order_index=block.reading_order,
            token_count=_estimate_tokens(content),
            embedding=[],
            created_at=ts,
        )
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# encode_chunks_from_section
# ---------------------------------------------------------------------------


def encode_chunks_from_section(
    node: DocumentNodeOut,
    chat_id: uuid.UUID,
    document_id: uuid.UUID,
    *,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    created_at: int | None = None,
) -> list[VespaChunk]:
    """Split a section node's content into ~``target_tokens``-token chunks.

    The split uses sentence boundaries (split on ``". "`` / ``"\\n"``).
    Long single sentences are kept as-is (no sub-sentence splitting).

    Each produced chunk gets:
    - ``source_type`` = ``"chunk"``
    - ``order_index`` = global ``node.order_index * 1000 + chunk_position``
      (preserves inter-chunk ordering while remaining unique).
    - Monotonically increasing ``order_index`` across chunks within the node.
    - ``heading_path`` from ``node.title``.

    Parameters
    ----------
    node:
        A ``DocumentNodeOut`` with non-empty ``content`` (paragraph / section
        text joined by the hierarchy layer).
    chat_id:
        Chat isolation context.
    document_id:
        Parent document UUID.
    target_tokens:
        Approximate token count per chunk.
    created_at:
        Unix epoch ms; defaults to ``now``.
    """
    ts = created_at if created_at is not None else _now_epoch_ms()
    content = node.content.strip()
    if not content:
        return []

    # Sentence-based split: split on ". " or "\\n" keeping sentences together
    sentences: list[str] = []
    raw_sentences = content.replace("\n", ". ").split(". ")
    for s in raw_sentences:
        s = s.strip()
        if s:
            sentences.append(s)

    chunks: list[VespaChunk] = []
    current_words: list[str] = []
    chunk_index = 0

    def _flush() -> None:
        nonlocal chunk_index
        text = ". ".join(current_words).strip()
        if not text:
            return
        order_idx = node.order_index * 1000 + chunk_index
        chunk = VespaChunk(
            vespa_document_id=make_vespa_id(
                document_id,
                "chunk",
                node.id,
                order_idx,
            ),
            chat_id=_uuid_str(chat_id),
            document_id=_uuid_str(document_id),
            source_node_id=_uuid_str(node.id),
            parent_node_id=_uuid_str(node.parent_id) if node.parent_id else None,
            source_type="chunk",
            title=node.title or "",
            heading_path=node.title or "",
            content=text,
            page_start=node.page_start,
            page_end=node.page_end,
            order_index=order_idx,
            token_count=_estimate_tokens(text),
            embedding=[],
            created_at=ts,
        )
        chunks.append(chunk)
        chunk_index += 1
        current_words.clear()

    for sentence in sentences:
        current_words.append(sentence)
        estimated = _estimate_tokens(". ".join(current_words))
        if estimated >= target_tokens:
            _flush()

    # Flush remaining sentences
    _flush()

    return chunks


# ---------------------------------------------------------------------------
# encode_section_summary
# ---------------------------------------------------------------------------


def encode_section_summary(
    section: SectionEnrichment,
    chat_id: uuid.UUID,
    document_id: uuid.UUID,
    *,
    created_at: int | None = None,
) -> list[VespaChunk]:
    """Encode a ``SectionEnrichment`` as two Vespa chunks.

    Returns
    -------
    list[VespaChunk]
        Always two entries:
        1. ``source_type="section_summary"``  — ``detailed_summary``
        2. ``source_type="compact_section_summary"`` — ``compact_summary``
    """
    ts = created_at if created_at is not None else _now_epoch_ms()
    node_id = section.node_id

    def _make(st: str, text: str, order_offset: int) -> VespaChunk:
        return VespaChunk(
            vespa_document_id=make_vespa_id(document_id, st, node_id, order_offset),
            chat_id=_uuid_str(chat_id),
            document_id=_uuid_str(document_id),
            source_node_id=_uuid_str(node_id),
            parent_node_id=None,
            source_type=st,
            title=section.title or "",
            heading_path=section.title or "",
            content=text,
            keywords=list(section.keywords),
            technical_keywords=list(section.technical_keywords),
            entities=list(section.entities),
            page_start=section.page_start,
            page_end=section.page_end,
            order_index=order_offset,
            token_count=_estimate_tokens(text),
            embedding=[],
            created_at=ts,
        )

    return [
        _make("section_summary", section.detailed_summary, 0),
        _make("compact_section_summary", section.compact_summary, 1),
    ]


# ---------------------------------------------------------------------------
# encode_document_overview
# ---------------------------------------------------------------------------


def encode_document_overview(
    doc: DocumentEnrichment,
    chat_id: uuid.UUID,
    document_id: uuid.UUID,
    *,
    abstract_node_id: uuid.UUID | None = None,
    created_at: int | None = None,
) -> list[VespaChunk]:
    """Encode a ``DocumentEnrichment`` as 1 or 2 Vespa chunks.

    Produces:
    - ``source_type="document_overview"`` — ``document_overview`` text.
    - ``source_type="chapter_summary"``   — ``abstract_summary`` text.
      Only emitted when ``doc.abstract_summary`` is not ``None``.

    Parameters
    ----------
    doc:
        Document-level enrichment produced by Phase 5.2.
    chat_id:
        Chat isolation context.
    document_id:
        Parent document UUID.
    abstract_node_id:
        UUID of the abstract ``DocumentNode`` (used as ``source_node_id`` for
        the chapter_summary chunk).  Defaults to a deterministic synthetic UUID
        when ``None``.
    created_at:
        Unix epoch ms; defaults to ``now``.
    """
    ts = created_at if created_at is not None else _now_epoch_ms()
    # Use a synthetic source_node_id for the document-level overview
    synthetic_node_id = uuid.uuid5(
        uuid.NAMESPACE_OID, f"{document_id}:document_overview"
    )

    chunks: list[VespaChunk] = [
        VespaChunk(
            vespa_document_id=make_vespa_id(document_id, "document_overview", synthetic_node_id, 0),
            chat_id=_uuid_str(chat_id),
            document_id=_uuid_str(document_id),
            source_node_id=_uuid_str(synthetic_node_id),
            parent_node_id=None,
            source_type="document_overview",
            title="",
            heading_path="",
            content=doc.document_overview,
            page_start=1,
            page_end=1,
            order_index=0,
            token_count=_estimate_tokens(doc.document_overview),
            embedding=[],
            created_at=ts,
        )
    ]

    if doc.abstract_summary is not None:
        abs_node_id = abstract_node_id or uuid.uuid5(
            uuid.NAMESPACE_OID, f"{document_id}:abstract"
        )
        chunks.append(
            VespaChunk(
                vespa_document_id=make_vespa_id(
                    document_id, "chapter_summary", abs_node_id, 0
                ),
                chat_id=_uuid_str(chat_id),
                document_id=_uuid_str(document_id),
                source_node_id=_uuid_str(abs_node_id),
                parent_node_id=None,
                source_type="chapter_summary",
                title="Abstract",
                heading_path="Abstract",
                content=doc.abstract_summary,
                page_start=1,
                page_end=1,
                order_index=1,
                token_count=_estimate_tokens(doc.abstract_summary),
                embedding=[],
                created_at=ts,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# encode_structured_facts
# ---------------------------------------------------------------------------

_FACT_KIND_TO_SOURCE_TYPE: dict[str, str] = {
    "metric": "performance_fact",
    "benchmark": "performance_fact",
    "dataset": "performance_fact",
    "hyperparameter": "performance_fact",
    "claim": "claim",
    "definition": "definition",
    "technology": "technology_card",
    "technology_card": "technology_card",
    "other": "claim",
}


def encode_structured_facts(
    facts: list[StructuredFact],
    *,
    created_at: int | None = None,
) -> list[VespaChunk]:
    """Encode ORM ``StructuredFact`` rows as Vespa chunks.

    Maps ``fact.kind`` to a Vespa ``source_type``:
    - ``"metric"`` / ``"benchmark"`` / ``"dataset"`` / ``"hyperparameter"``
      → ``"performance_fact"``
    - ``"claim"`` / ``"other"`` → ``"claim"``
    - ``"definition"`` → ``"definition"``
    - ``"technology"`` / ``"technology_card"`` → ``"technology_card"``

    Chunk ``content`` is built from:
    ``"{key}: {value_text}  (context: {context_excerpt})"``

    Parameters
    ----------
    facts:
        ORM ``StructuredFact`` rows — must all share the same ``chat_id`` /
        ``document_id`` (the encoder trusts the caller to scope correctly).
    created_at:
        Unix epoch ms; defaults to ``now``.
    """
    ts = created_at if created_at is not None else _now_epoch_ms()
    chunks: list[VespaChunk] = []

    for idx, fact in enumerate(facts):
        source_type = _FACT_KIND_TO_SOURCE_TYPE.get(fact.kind, "claim")

        # Build content
        value_text = str(fact.value) if not isinstance(fact.value, str) else fact.value
        content = f"{fact.key}: {value_text}"
        if fact.context_excerpt:
            content += f"  (context: {fact.context_excerpt})"
        if fact.unit:
            content = f"{fact.key}: {value_text} {fact.unit}"

        source_node_id = (
            str(fact.source_node_id) if fact.source_node_id else str(fact.id)
        )
        order_idx = idx

        chunk = VespaChunk(
            vespa_document_id=make_vespa_id(
                fact.document_id,
                source_type,
                fact.id,
                order_idx,
            ),
            chat_id=str(fact.chat_id),
            document_id=str(fact.document_id),
            source_node_id=source_node_id,
            parent_node_id=None,
            source_type=source_type,
            title=fact.key,
            heading_path="",
            content=content,
            page_start=fact.page if fact.page is not None else 1,
            page_end=fact.page if fact.page is not None else 1,
            order_index=order_idx,
            token_count=_estimate_tokens(content),
            embedding=[],
            created_at=ts,
        )
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# encode_figure_narrations  (multimodal narration chunks)
# ---------------------------------------------------------------------------


_NARRATION_TEXT_LIMIT = 4000
"""Cap the embedded narration text size so a runaway VLM response doesn't blow
out the per-chunk token budget."""


def _compose_narration_content(narration: FigureNarration) -> str:
    """Build the chunk text that gets embedded for one figure / table.

    The reference design (see ``ref/src/app/modules/document_parser/infra/vision/image_narrator.py``)
    co-locates the page text with the narration so a single embedding captures
    both the visual semantics and the textual context.  We preserve that by
    laying out:

        [caption]
        [VLM narration]
        Same-page context:
        [page text excerpt]

    Empty sections are omitted so short captions don't dilute the signal.
    """
    parts: list[str] = []
    if narration.caption:
        parts.append(narration.caption.strip())
    if narration.narrative_text:
        parts.append(narration.narrative_text.strip())
    if narration.footnote:
        parts.append(f"Footnote: {narration.footnote.strip()}")
    if narration.page_text_excerpt.strip():
        parts.append("Same-page context:\n" + narration.page_text_excerpt.strip())
    content = "\n\n".join(p for p in parts if p)
    if len(content) > _NARRATION_TEXT_LIMIT:
        content = content[:_NARRATION_TEXT_LIMIT] + " …"
    return content


def encode_figure_narrations(
    narrations: list[FigureNarration],
    *,
    created_at: int | None = None,
) -> list[VespaChunk]:
    """Encode VLM-produced figure / table narrations as Vespa chunks.

    Source type mapping:
    - ``BlockType.table`` → ``"table_record"``
    - ``BlockType.image`` → ``"figure_caption"``

    Each narration produces exactly one chunk whose ``content`` blends the
    caption, the narration, and the same-page text excerpt so retrieval can
    surface the chunk from either a visual query (e.g. "the figure that
    compares latency on COCO") or a textual one.
    """
    ts = created_at if created_at is not None else _now_epoch_ms()
    chunks: list[VespaChunk] = []

    for narration in narrations:
        if narration.error or not narration.narrative_text.strip():
            # Failed narrations are skipped — raw_block encoder still indexes the
            # caption text, so we don't lose retrievability entirely.
            continue

        source_type = (
            "table_record"
            if narration.block_type == BlockType.table
            else "figure_caption"
        )
        content = _compose_narration_content(narration)
        if not content:
            continue

        chunks.append(
            VespaChunk(
                vespa_document_id=make_vespa_id(
                    narration.document_id,
                    source_type,
                    narration.block_id,
                    narration.reading_order,
                ),
                chat_id=_uuid_str(narration.chat_id),
                document_id=_uuid_str(narration.document_id),
                source_node_id=_uuid_str(narration.block_id),
                parent_node_id=None,
                source_type=source_type,
                title=(narration.caption or narration.image_path or "")[:200],
                heading_path=narration.caption or "",
                content=content,
                page_start=narration.page_number,
                page_end=narration.page_number,
                order_index=narration.reading_order,
                token_count=_estimate_tokens(content),
                embedding=[],
                created_at=ts,
            )
        )

    return chunks


__all__ = [
    "CHUNK_TARGET_TOKENS",
    "encode_chunks_from_section",
    "encode_document_overview",
    "encode_figure_narrations",
    "encode_raw_blocks",
    "encode_section_summary",
    "encode_structured_facts",
]
