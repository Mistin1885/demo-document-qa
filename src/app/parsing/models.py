"""Domain models for MinerU-parsed document blocks.

These are pure Pydantic v2 data-transfer objects.  No database I/O, no
FastAPI imports.  Phase 4.3 (hierarchy) and Phase 5 (enrichment) consume
these to produce ``DocumentNode`` rows and Vespa chunks respectively.

Mapping source: ``<doc>_middle.json`` produced by ``scripts/mineru_poc.py``
(post-processed MinerU 3.3+ hybrid output).

See: CLAUDE.md §6.3, §6.4; ``deploy/mineru/output-schema.md``.
"""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# BlockType
# ---------------------------------------------------------------------------


class BlockType(StrEnum):
    """All possible block types emitted by the mapping layer.

    Values are lower-case strings so they survive JSON round-trips without
    conversion.  The ``discarded`` type captures MinerU ``aside_text``,
    ``page_footnote``, and ``page_number`` blocks as well as any unknown
    ``type`` values we encounter.
    """

    title = "title"  # type: ignore[assignment]
    text = "text"  # retained for parity; mapping layer uses paragraph
    paragraph = "paragraph"  # text block without text_level
    image = "image"
    image_caption = "image_caption"
    image_footnote = "image_footnote"
    table = "table"
    table_caption = "table_caption"
    table_footnote = "table_footnote"
    equation = "equation"  # MinerU interline_equation
    ref_text = "ref_text"
    discarded = "discarded"  # page_footnote / aside_text / page_number + unknowns


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class BBox(BaseModel):
    """Bounding box in PDF point space, augmented with page dimensions.

    ``page_width`` and ``page_height`` come from ``pdf_info[i].page_size`` so
    callers can normalise coordinates to [0, 1] without an extra lookup.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float


class ImageRef(BaseModel):
    """Metadata about an image extracted from a MinerU ``image`` block.

    ``image_path`` is the post-processed renamed basename (already on disk
    under ``images/``); consumers must not re-derive the name.
    """

    image_path: str  # renamed basename, e.g. "2410.05779v3_p3_2fa1818e.jpg"
    caption: str | None = None
    footnote: str | None = None


class TableRef(BaseModel):
    """Metadata about a table extracted from a MinerU ``table`` block.

    ``html_body`` is the raw ``<table>…</table>`` HTML fragment produced by
    MinerU (rowspan/colspan preserved).
    """

    html_body: str  # <table>...</table> HTML; empty string if not found
    caption: str | None = None
    footnote: str | None = None


# ---------------------------------------------------------------------------
# ParsedBlock
# ---------------------------------------------------------------------------


class ParsedBlock(BaseModel):
    """Flat representation of one MinerU block after mapping.

    One ``ParsedBlock`` corresponds to one entry in
    ``pdf_info[i].preproc_blocks`` (or ``discarded_blocks``).  Image and table
    sub-blocks (``image_body``, ``image_caption``, ``table_body``,
    ``table_caption``, …) are *folded into* the parent block's ``image`` /
    ``table`` field rather than emitted as separate ``ParsedBlock`` instances.

    Fields
    ------
    block_id
        Deterministic UUID-5 derived from ``(document_id, page_idx,
        reading_order, type)`` so repeated ingestion runs produce the same IDs
        (Vespa upsert is idempotent).
    chat_id / document_id
        Injected by the service layer (ingestion pipeline); the mapping
        function never mints these.
    page_number
        1-indexed page number (``pdf_info[i].page_idx + 1``).
    block_type
        See ``BlockType``.
    text
        Plain-text content for ``title``, ``paragraph``, ``ref_text``,
        ``equation``.  Empty string for ``image`` / ``table`` (structured data
        lives in ``image`` / ``table``).
    title_level
        ``1`` = document title, ``2`` = section / subsection heading.
        ``None`` for non-title blocks.
    bbox
        Block bounding box plus page dimensions.
    reading_order
        Per-page reading order index from MinerU (``block["index"]``); if the
        field is absent, the block's enumerate position within the page is used.
    image
        Populated for ``block_type == image``; ``None`` otherwise.
    table
        Populated for ``block_type == table``; ``None`` otherwise.
    equation_latex
        Raw LaTeX string for ``block_type == equation``; ``None`` otherwise.
    confidence
        Minimum span ``score`` seen in this block.  ``None`` if no spans carry
        a ``score`` field (e.g. ``image_body`` spans do not).
    """

    block_id: UUID
    chat_id: UUID
    document_id: UUID
    page_number: int = Field(ge=1)
    block_type: BlockType
    text: str
    title_level: int | None = None
    bbox: BBox
    reading_order: int
    image: ImageRef | None = None
    table: TableRef | None = None
    equation_latex: str | None = None
    confidence: float | None = None


# ---------------------------------------------------------------------------
# Phase 4.3 — Hierarchy models
# ---------------------------------------------------------------------------


class NodeType(StrEnum):
    """Document-tree node types produced by ``derive_hierarchy``.

    Mirrors the ORM ``DocumentNode.node_type`` enum defined in Phase 2.2.
    Phase 4.3 extends the ORM list with ``abstract`` and ``authors``; Phase 5
    (enrichment) will use all values here.
    """

    document = "document"
    section = "section"
    subsection = "subsection"
    paragraph = "paragraph"
    abstract = "abstract"
    authors = "authors"
    figure = "figure"
    table = "table"
    equation = "equation"
    reference = "reference"
    appendix = "appendix"


class DocumentNodeOut(BaseModel):
    """Output node produced by ``derive_hierarchy``.

    Design rules (CLAUDE.md §12)
    ----------------------------
    - Pure data-transfer object — no DB I/O, no FastAPI imports.
    - ``metadata_`` uses a typed union; never ``dict[str, Any]``.
    - ``id`` is deterministic (UUID-5) so repeated ingestion is idempotent.
    - ``source_block_ids`` enables upstream traceability back to raw blocks.
    """

    id: UUID
    """Deterministic UUID-5: ``uuid5(NAMESPACE_OID, f"{document_id}:node:{ordinal}")``.

    ``ordinal`` is the global ``order_index`` of this node within the document.
    """

    chat_id: UUID
    document_id: UUID
    parent_id: UUID | None
    """``None`` for the document-root node; populated for all others."""

    node_type: NodeType
    title: str | None
    content: str
    """Joined inner text.

    - ``paragraph`` / ``abstract`` / ``authors``: same as block text.
    - ``section`` / ``subsection`` / ``appendix`` / ``document``: empty string
      (children carry the text).
    - ``figure``: ``caption + "\\n" + footnote`` (empty string if neither).
    - ``table``: same convention as figure.
    - ``equation``: raw LaTeX string.
    - ``reference``: full reference text.
    """

    page_start: int = Field(ge=1)
    """1-indexed first page covered by this node."""

    page_end: int = Field(ge=1)
    """1-indexed last page covered by this node."""

    order_index: int = Field(ge=0)
    """Global 0-based ordering within the document (matches reading order)."""

    level: int = Field(ge=0)
    """Tree depth: ``document`` = 0, ``section`` / ``abstract`` / ``authors``
    / ``appendix`` = 1, ``subsection`` / ``paragraph`` under section = 2,
    paragraph under subsection = 3, etc."""

    bbox: BBox | None
    """Block bounding box when this node corresponds to a single ParsedBlock;
    ``None`` for structural container nodes that span multiple blocks."""

    source_block_ids: list[UUID]
    """IDs of ``ParsedBlock`` instances that belong to this node.

    Invariant: each ``ParsedBlock`` appears in at most one node's
    ``source_block_ids`` across the entire ``HierarchyResult``.
    """

    metadata_: dict[str, str | int | float | bool | None]
    """Extensible metadata bag.

    Mandatory keys injected by ``derive_hierarchy``:
    - ``"source"``: always ``"heuristic"`` (Phase 4.3); Phase 5 enrichment
      will add ``"llm"`` values.

    Optional keys (set when applicable):
    - ``"heuristic_name"``: name of the heuristic that produced this node.
    - ``"image_path"``: for ``NodeType.figure``.
    - ``"html_body"``: full ``<table>…</table>`` HTML for ``NodeType.table``.
    - ``"boundary"``: ``"references"`` or ``"appendix"`` for boundary sections.
    """


class HierarchyResult(BaseModel):
    """Full hierarchy for one document returned by ``derive_hierarchy``.

    Consumers (Phase 4.4 evaluation, Phase 5 enrichment, ingestion service)
    read ``nodes`` as an ordered flat list; tree structure is recovered via
    ``parent_id`` links.
    """

    document_id: UUID
    chat_id: UUID
    nodes: list[DocumentNodeOut]
    references_start_index: int | None
    """``order_index`` of the first node whose ``node_type == NodeType.reference``
    (i.e. the first individual reference entry, *not* the references-section
    heading).  ``None`` if no references were found."""

    appendix_start_index: int | None
    """``order_index`` of the ``NodeType.appendix`` boundary node.
    ``None`` if no appendix was found."""

    heuristics_applied: list[str]
    """Sorted, deduplicated list of heuristic names applied during derivation.

    Populated by ``derive_hierarchy``; used by Phase 4.4 evaluator to audit
    which heuristics fired on a given document.
    """


__all__ = [
    "BlockType",
    "BBox",
    "ImageRef",
    "TableRef",
    "ParsedBlock",
    # Phase 4.3
    "NodeType",
    "DocumentNodeOut",
    "HierarchyResult",
]
