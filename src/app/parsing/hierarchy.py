"""Derive document hierarchy from a flat list of ParsedBlock instances.

Public API
----------
- :func:`derive_hierarchy` — pure function: ``list[ParsedBlock] → HierarchyResult``.

Algorithm (Phase 4.3)
---------------------
The derivation proceeds in eight sequential steps (A–H) that are each
implemented as a private helper.  All heuristics are recorded in
``HierarchyResult.heuristics_applied`` so Phase 4.4 evaluation can audit them.

Design rules (CLAUDE.md §12)
------------------------------
- Pure function: no I/O, no database writes, no FastAPI imports.
- Full type hints on every public and private helper.
- ``metadata_`` uses a typed union — never ``dict[str, Any]``.
- Each ``ParsedBlock`` ends up in *at most one* node's ``source_block_ids``.
- ``discarded`` blocks are never owned by a node.

See: CLAUDE.md §6.4, §5.1.
"""

from __future__ import annotations

import re
from uuid import NAMESPACE_OID, UUID, uuid5

from app.parsing.models import (
    BBox,
    BlockType,
    DocumentNodeOut,
    HierarchyResult,
    NodeType,
    ParsedBlock,
)

# ---------------------------------------------------------------------------
# Typed metadata alias (mirrors DocumentNodeOut.metadata_ annotation)
# ---------------------------------------------------------------------------

_MetaDict = dict[str, str | int | float | bool | None]

# ---------------------------------------------------------------------------
# Heuristic name constants
# ---------------------------------------------------------------------------

_HEURISTIC_AUTHORS = "authors-by-sup-numeric-or-star"
_HEURISTIC_ABSTRACT_HEADING = "abstract-by-heading"
_HEURISTIC_ABSTRACT_PREFIX = "abstract-by-prefix"
_HEURISTIC_SUBSECTION_NUMBERED = "subsection-by-numbered-heading"
_HEURISTIC_REFS_BY_KEYWORD = "refs-by-keyword"
_HEURISTIC_APPENDIX_BY_KEYWORD = "appendix-by-keyword"

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Subsection patterns: "3.1", "3.1.1", "A.1", "A.1.2", etc.
_RE_SUBSECTION = re.compile(
    r"^(?:[A-Z]|\d+)"  # top-level number or letter
    r"(?:\.\d+)+"  # one or more .N segments → subsection
    r"(?:\s|$)",  # followed by space or end
)

# References heading keywords (case-insensitive exact match after strip)
_RE_REFS_HEADING = re.compile(r"^(references|bibliography)$", re.IGNORECASE)

# Appendix heading: starts with "appendix" (word), or "A APPENDIX", or
# matches "Appendix N" / "Appendix A" / "A.N ..." type headings, or
# patterns like "7 APPENDIX" (number followed by word APPENDIX)
_RE_APPENDIX_HEADING = re.compile(
    r"^appendix\b"  # "Appendix ...", "APPENDIX ..."
    r"|^\d+\s+appendix\b"  # "7 APPENDIX", "1 Appendix"
    r"|^[A-Z]\s+APPENDIX\b",  # "A APPENDIX"
    re.IGNORECASE,
)

# Authors heuristic: contains <sup>digit or <sup>* — suggests affiliation markers
_RE_SUP_NUMERIC = re.compile(r"<sup>[\d\*†‡§¶]+</sup>", re.IGNORECASE)
_RE_AT_SIGN = re.compile(r"@")

# Abstract prefix heuristic
_RE_ABSTRACT_PREFIX = re.compile(r"^abstract\s*[—\-–:]\s*", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Deterministic node id
# ---------------------------------------------------------------------------


def _node_id(document_id: UUID, ordinal: int) -> UUID:
    """Return ``uuid5(NAMESPACE_OID, f"{document_id}:node:{ordinal}")``.

    ``ordinal`` is the ``order_index`` of the node.
    """
    return uuid5(NAMESPACE_OID, f"{document_id}:node:{ordinal}")


# ---------------------------------------------------------------------------
# Node builder helper
# ---------------------------------------------------------------------------


def _make_node(
    *,
    document_id: UUID,
    chat_id: UUID,
    node_type: NodeType,
    title: str | None,
    content: str,
    page_start: int,
    page_end: int,
    order_index: int,
    level: int,
    parent_id: UUID | None,
    bbox: BBox | None,
    source_block_ids: list[UUID],
    metadata_: _MetaDict,
) -> DocumentNodeOut:
    return DocumentNodeOut(
        id=_node_id(document_id, order_index),
        chat_id=chat_id,
        document_id=document_id,
        parent_id=parent_id,
        node_type=node_type,
        title=title,
        content=content,
        page_start=page_start,
        page_end=page_end,
        order_index=order_index,
        level=level,
        bbox=bbox,
        source_block_ids=source_block_ids,
        metadata_=metadata_,
    )


# ---------------------------------------------------------------------------
# Heuristic predicates
# ---------------------------------------------------------------------------


def _looks_like_authors(text: str) -> bool:
    """Return True when *text* looks like an author list.

    Heuristic rules (all must hold):
    1. Fewer than 400 characters.
    2. Contains a ``<sup>`` numeric/star/dagger marker OR an ``@`` sign.
    3. Does not contain three or more complete sentences (rough proxy for a
       paragraph of prose rather than a comma-separated name list).

    Deliberate false-negative bias: if unsure, return False so we don't
    misclassify a paragraph as authors.
    """
    if len(text) >= 400:
        return False
    # Must have affiliation markers or email
    if not (_RE_SUP_NUMERIC.search(text) or _RE_AT_SIGN.search(text)):
        return False
    # Rough sentence count: count sentences ending with period + capital or end
    # Avoid matching "et al." or abbreviations by requiring space or end after.
    sentence_endings = re.findall(r"\.\s+[A-Z]", text)
    if len(sentence_endings) >= 3:
        return False
    return True


def _is_refs_heading(text: str) -> bool:
    """Return True for 'References', 'REFERENCES', 'Bibliography' headings."""
    return bool(_RE_REFS_HEADING.match(text.strip()))


def _is_appendix_heading(text: str) -> bool:
    """Return True for 'Appendix…', 'APPENDIX…', 'A APPENDIX' headings."""
    stripped = text.strip()
    return bool(_RE_APPENDIX_HEADING.match(stripped))


def _is_subsection_heading(text: str) -> bool:
    """Return True when the heading text starts with a sub-numbered prefix.

    Examples that return True: '3.1 Methods', '3.1.1 Details', 'A.1 Setup'.
    Examples that return False: '3 Methods', 'Introduction', 'A Background'.
    """
    return bool(_RE_SUBSECTION.match(text.strip()))


# ---------------------------------------------------------------------------
# Main pure function
# ---------------------------------------------------------------------------


def derive_hierarchy(
    parsed_blocks: list[ParsedBlock],
    *,
    document_id: UUID,
    chat_id: UUID,
) -> HierarchyResult:
    """Derive a document hierarchy from a flat ``ParsedBlock`` list.

    Parameters
    ----------
    parsed_blocks:
        Ordered list produced by ``map_middle_to_parsed_blocks``.  The function
        treats them as already sorted by ``(page_number, reading_order)``.
    document_id:
        Stable identifier for the document (injected by the service layer).
    chat_id:
        Chat isolation identifier (injected by the service layer).

    Returns
    -------
    HierarchyResult
        Flat ordered list of nodes plus structural metadata.
    """
    nodes: list[DocumentNodeOut] = []
    heuristics_applied: set[str] = set()

    # Track which block_ids have been claimed to enforce the one-owner invariant.
    claimed_block_ids: set[UUID] = set()

    def _claim(block_ids: list[UUID]) -> list[UUID]:
        """Verify no block is claimed twice; record them and return the list."""
        for bid in block_ids:
            if bid in claimed_block_ids:
                raise RuntimeError(
                    f"Block {bid} claimed by multiple nodes — hierarchy invariant violated"
                )
            claimed_block_ids.add(bid)
        return block_ids

    # Running ordinal counter — document root gets 0.
    ordinal: int = 0

    # ------------------------------------------------------------------ #
    # Step (A): doc title → document root node (level=0)
    # ------------------------------------------------------------------ #
    doc_title_block: ParsedBlock | None = None
    for blk in parsed_blocks:
        if blk.block_type == BlockType.title and blk.title_level == 1:
            doc_title_block = blk
            break

    doc_root = _make_node(
        document_id=document_id,
        chat_id=chat_id,
        node_type=NodeType.document,
        title=doc_title_block.text if doc_title_block is not None else None,
        content="",
        page_start=doc_title_block.page_number if doc_title_block is not None else 1,
        page_end=doc_title_block.page_number if doc_title_block is not None else 1,
        order_index=ordinal,
        level=0,
        parent_id=None,
        bbox=doc_title_block.bbox if doc_title_block is not None else None,
        source_block_ids=_claim([doc_title_block.block_id]) if doc_title_block is not None else [],
        metadata_={"source": "heuristic"},
    )
    nodes.append(doc_root)
    ordinal += 1

    # Convenience: the document root's UUID used as parent for top-level nodes.
    root_id: UUID = doc_root.id

    # ------------------------------------------------------------------ #
    # Collect all non-discarded, non-doc-title blocks for further processing.
    # ------------------------------------------------------------------ #
    # Skip the doc-title block itself (already claimed).
    remaining: list[ParsedBlock] = [
        b
        for b in parsed_blocks
        if b.block_type != BlockType.discarded
        and not (b.block_type == BlockType.title and b.title_level == 1)
    ]

    # ------------------------------------------------------------------ #
    # Step (B): authors heuristic
    # ------------------------------------------------------------------ #
    # Find the index of the first section heading (title_level=2) in remaining.
    first_section_idx = next(
        (
            i
            for i, b in enumerate(remaining)
            if b.block_type == BlockType.title and b.title_level == 2
        ),
        len(remaining),
    )

    # Candidate paragraphs before the first section heading
    pre_section: list[ParsedBlock] = remaining[:first_section_idx]
    author_block_ids: list[UUID] = []

    for blk in pre_section:
        if blk.block_type in (BlockType.paragraph, BlockType.text):
            if _looks_like_authors(blk.text):
                author_block_ids.append(blk.block_id)

    if author_block_ids:
        heuristics_applied.add(_HEURISTIC_AUTHORS)
        # Collect all author blocks to join into one node
        author_blocks = [b for b in pre_section if b.block_id in set(author_block_ids)]
        author_text = " ".join(b.text for b in author_blocks)
        author_node = _make_node(
            document_id=document_id,
            chat_id=chat_id,
            node_type=NodeType.authors,
            title=None,
            content=author_text,
            page_start=min(b.page_number for b in author_blocks),
            page_end=max(b.page_number for b in author_blocks),
            order_index=ordinal,
            level=1,
            parent_id=root_id,
            bbox=author_blocks[0].bbox if len(author_blocks) == 1 else None,
            source_block_ids=_claim(author_block_ids),
            metadata_={"source": "heuristic", "heuristic_name": _HEURISTIC_AUTHORS},
        )
        nodes.append(author_node)
        ordinal += 1

    # Remove claimed blocks from remaining.
    remaining = [b for b in remaining if b.block_id not in claimed_block_ids]

    # ------------------------------------------------------------------ #
    # Step (C): abstract
    # ------------------------------------------------------------------ #
    abstract_heading_idx: int | None = None
    for i, blk in enumerate(remaining):
        if blk.block_type == BlockType.title and blk.title_level == 2:
            if blk.text.strip().lower() == "abstract":
                abstract_heading_idx = i
                break

    if abstract_heading_idx is not None:
        heuristics_applied.add(_HEURISTIC_ABSTRACT_HEADING)
        abs_heading_blk = remaining[abstract_heading_idx]
        # Collect paragraph blocks immediately after the heading until next heading
        abs_paragraphs: list[ParsedBlock] = []
        j = abstract_heading_idx + 1
        while j < len(remaining):
            blk = remaining[j]
            if blk.block_type == BlockType.title and blk.title_level == 2:
                break
            if blk.block_type in (BlockType.paragraph, BlockType.text):
                abs_paragraphs.append(blk)
            j += 1

        all_abs_blocks = [abs_heading_blk] + abs_paragraphs
        abs_content = " ".join(b.text for b in abs_paragraphs)
        abs_ids = [b.block_id for b in all_abs_blocks]

        abstract_node = _make_node(
            document_id=document_id,
            chat_id=chat_id,
            node_type=NodeType.abstract,
            title="Abstract",
            content=abs_content,
            page_start=min(b.page_number for b in all_abs_blocks),
            page_end=max(b.page_number for b in all_abs_blocks),
            order_index=ordinal,
            level=1,
            parent_id=root_id,
            bbox=None,
            source_block_ids=_claim(abs_ids),
            metadata_={"source": "heuristic", "heuristic_name": _HEURISTIC_ABSTRACT_HEADING},
        )
        nodes.append(abstract_node)
        ordinal += 1
    else:
        # Fallback: abstract-by-prefix heuristic — page 1 paragraph starting
        # with "Abstract —" or "Abstract:" etc.
        for blk in remaining:
            if blk.page_number == 1 and blk.block_type in (BlockType.paragraph, BlockType.text):
                if _RE_ABSTRACT_PREFIX.match(blk.text):
                    heuristics_applied.add(_HEURISTIC_ABSTRACT_PREFIX)
                    abstract_node = _make_node(
                        document_id=document_id,
                        chat_id=chat_id,
                        node_type=NodeType.abstract,
                        title="Abstract",
                        content=blk.text,
                        page_start=blk.page_number,
                        page_end=blk.page_number,
                        order_index=ordinal,
                        level=1,
                        parent_id=root_id,
                        bbox=blk.bbox,
                        source_block_ids=_claim([blk.block_id]),
                        metadata_={
                            "source": "heuristic",
                            "heuristic_name": _HEURISTIC_ABSTRACT_PREFIX,
                        },
                    )
                    nodes.append(abstract_node)
                    ordinal += 1
                    break

    # Remove claimed blocks from remaining
    remaining = [b for b in remaining if b.block_id not in claimed_block_ids]

    # ------------------------------------------------------------------ #
    # Steps (D), (E), (F), (G): main pass over remaining blocks
    # ------------------------------------------------------------------ #
    # State machine: current section / subsection / appendix node ids + levels
    current_section_id: UUID | None = None  # most recent section node id
    current_subsection_id: UUID | None = None  # most recent subsection node id
    current_appendix_id: UUID | None = None  # most recent appendix node id
    in_appendix: bool = False  # True once appendix started

    references_start_index: int | None = None
    appendix_start_index: int | None = None

    def _current_container_id() -> UUID:
        """Return the id of the deepest active container for attaching children."""
        if current_subsection_id is not None:
            return current_subsection_id
        if current_section_id is not None:
            return current_section_id
        if current_appendix_id is not None:
            return current_appendix_id
        return root_id

    def _paragraph_level() -> int:
        """Return tree level for a paragraph under the current container."""
        if current_subsection_id is not None:
            return 3
        if current_section_id is not None:
            return 2
        if current_appendix_id is not None:
            return 2
        return 1

    for blk in remaining:
        if blk.block_id in claimed_block_ids:
            continue  # already claimed (should not happen at this point, but safety)

        # ---- heading block -----------------------------------------------
        if blk.block_type == BlockType.title and blk.title_level == 2:
            heading_text = blk.text.strip()

            # Step (E): References boundary
            if _is_refs_heading(heading_text):
                heuristics_applied.add(_HEURISTIC_REFS_BY_KEYWORD)
                in_appendix = False
                current_subsection_id = None
                sec_node = _make_node(
                    document_id=document_id,
                    chat_id=chat_id,
                    node_type=NodeType.section,
                    title=heading_text,
                    content="",
                    page_start=blk.page_number,
                    page_end=blk.page_number,
                    order_index=ordinal,
                    level=1,
                    parent_id=root_id,
                    bbox=blk.bbox,
                    source_block_ids=_claim([blk.block_id]),
                    metadata_={
                        "source": "heuristic",
                        "heuristic_name": _HEURISTIC_REFS_BY_KEYWORD,
                        "boundary": "references",
                    },
                )
                current_section_id = sec_node.id
                nodes.append(sec_node)
                ordinal += 1
                continue

            # Step (F): Appendix boundary
            if _is_appendix_heading(heading_text):
                heuristics_applied.add(_HEURISTIC_APPENDIX_BY_KEYWORD)
                in_appendix = True
                current_section_id = None
                current_subsection_id = None
                app_node = _make_node(
                    document_id=document_id,
                    chat_id=chat_id,
                    node_type=NodeType.appendix,
                    title=heading_text,
                    content="",
                    page_start=blk.page_number,
                    page_end=blk.page_number,
                    order_index=ordinal,
                    level=1,
                    parent_id=root_id,
                    bbox=blk.bbox,
                    source_block_ids=_claim([blk.block_id]),
                    metadata_={
                        "source": "heuristic",
                        "heuristic_name": _HEURISTIC_APPENDIX_BY_KEYWORD,
                        "boundary": "appendix",
                    },
                )
                if appendix_start_index is None:
                    appendix_start_index = ordinal
                current_appendix_id = app_node.id
                nodes.append(app_node)
                ordinal += 1
                continue

            # Step (D): subsection heuristic
            if _is_subsection_heading(heading_text):
                heuristics_applied.add(_HEURISTIC_SUBSECTION_NUMBERED)
                # Parent is current section / appendix, or root if none
                if in_appendix and current_appendix_id is not None:
                    parent = current_appendix_id
                    sub_level = 2
                elif current_section_id is not None:
                    parent = current_section_id
                    sub_level = 2
                else:
                    parent = root_id
                    sub_level = 2
                sub_node = _make_node(
                    document_id=document_id,
                    chat_id=chat_id,
                    node_type=NodeType.subsection,
                    title=heading_text,
                    content="",
                    page_start=blk.page_number,
                    page_end=blk.page_number,
                    order_index=ordinal,
                    level=sub_level,
                    parent_id=parent,
                    bbox=blk.bbox,
                    source_block_ids=_claim([blk.block_id]),
                    metadata_={
                        "source": "heuristic",
                        "heuristic_name": _HEURISTIC_SUBSECTION_NUMBERED,
                    },
                )
                current_subsection_id = sub_node.id
                nodes.append(sub_node)
                ordinal += 1
                continue

            # Step (D): plain section heading
            current_subsection_id = None
            if in_appendix:
                # A non-appendix heading after appendix started means we've
                # moved back to body — this shouldn't happen in well-formed PDFs
                # but handle gracefully: treat as regular section
                in_appendix = False
                current_appendix_id = None
            sec_node = _make_node(
                document_id=document_id,
                chat_id=chat_id,
                node_type=NodeType.section,
                title=heading_text,
                content="",
                page_start=blk.page_number,
                page_end=blk.page_number,
                order_index=ordinal,
                level=1,
                parent_id=root_id,
                bbox=blk.bbox,
                source_block_ids=_claim([blk.block_id]),
                metadata_={"source": "heuristic"},
            )
            current_section_id = sec_node.id
            nodes.append(sec_node)
            ordinal += 1
            continue

        # ---- ref_text block (Step E) ------------------------------------
        if blk.block_type == BlockType.ref_text:
            # Track first reference node order_index
            if references_start_index is None:
                references_start_index = ordinal
            heuristics_applied.add(_HEURISTIC_REFS_BY_KEYWORD)
            ref_node = _make_node(
                document_id=document_id,
                chat_id=chat_id,
                node_type=NodeType.reference,
                title=None,
                content=blk.text,
                page_start=blk.page_number,
                page_end=blk.page_number,
                order_index=ordinal,
                level=2,
                parent_id=current_section_id if current_section_id is not None else root_id,
                bbox=blk.bbox,
                source_block_ids=_claim([blk.block_id]),
                metadata_={"source": "heuristic", "heuristic_name": _HEURISTIC_REFS_BY_KEYWORD},
            )
            nodes.append(ref_node)
            ordinal += 1
            continue

        # ---- image block (Step G) ----------------------------------------
        if blk.block_type == BlockType.image:
            caption = blk.image.caption if blk.image else None
            footnote = blk.image.footnote if blk.image else None
            image_path = blk.image.image_path if blk.image else None
            parts = [p for p in [caption, footnote] if p]
            content = "\n".join(parts)
            meta: _MetaDict = {"source": "heuristic"}
            if image_path:
                meta["image_path"] = image_path
            fig_node = _make_node(
                document_id=document_id,
                chat_id=chat_id,
                node_type=NodeType.figure,
                title=caption,
                content=content,
                page_start=blk.page_number,
                page_end=blk.page_number,
                order_index=ordinal,
                level=_paragraph_level(),
                parent_id=_current_container_id(),
                bbox=blk.bbox,
                source_block_ids=_claim([blk.block_id]),
                metadata_=meta,
            )
            nodes.append(fig_node)
            ordinal += 1
            continue

        # ---- table block (Step G) ----------------------------------------
        if blk.block_type == BlockType.table:
            caption = blk.table.caption if blk.table else None
            footnote = blk.table.footnote if blk.table else None
            html_body = blk.table.html_body if blk.table else ""
            parts = [p for p in [caption, footnote] if p]
            content = "\n".join(parts)
            meta = {"source": "heuristic"}
            if html_body:
                meta["html_body"] = html_body
            tbl_node = _make_node(
                document_id=document_id,
                chat_id=chat_id,
                node_type=NodeType.table,
                title=caption,
                content=content,
                page_start=blk.page_number,
                page_end=blk.page_number,
                order_index=ordinal,
                level=_paragraph_level(),
                parent_id=_current_container_id(),
                bbox=blk.bbox,
                source_block_ids=_claim([blk.block_id]),
                metadata_=meta,
            )
            nodes.append(tbl_node)
            ordinal += 1
            continue

        # ---- equation block (Step G) -------------------------------------
        if blk.block_type == BlockType.equation:
            latex = blk.equation_latex or blk.text
            eq_node = _make_node(
                document_id=document_id,
                chat_id=chat_id,
                node_type=NodeType.equation,
                title=None,
                content=latex,
                page_start=blk.page_number,
                page_end=blk.page_number,
                order_index=ordinal,
                level=_paragraph_level(),
                parent_id=_current_container_id(),
                bbox=blk.bbox,
                source_block_ids=_claim([blk.block_id]),
                metadata_={"source": "heuristic"},
            )
            nodes.append(eq_node)
            ordinal += 1
            continue

        # ---- paragraph / text blocks (Step D) ----------------------------
        if blk.block_type in (BlockType.paragraph, BlockType.text):
            para_node = _make_node(
                document_id=document_id,
                chat_id=chat_id,
                node_type=NodeType.paragraph,
                title=None,
                content=blk.text,
                page_start=blk.page_number,
                page_end=blk.page_number,
                order_index=ordinal,
                level=_paragraph_level(),
                parent_id=_current_container_id(),
                bbox=blk.bbox,
                source_block_ids=_claim([blk.block_id]),
                metadata_={"source": "heuristic"},
            )
            nodes.append(para_node)
            ordinal += 1
            continue

        # ---- unclaimed non-discarded blocks (image_caption etc.) ---------
        # Top-level image_caption / image_footnote / table_caption /
        # table_footnote that appear outside their parent container are rare
        # but possible.  Fold them into a paragraph node so they aren't lost.
        if blk.block_type in (
            BlockType.image_caption,
            BlockType.image_footnote,
            BlockType.table_caption,
            BlockType.table_footnote,
        ):
            para_node = _make_node(
                document_id=document_id,
                chat_id=chat_id,
                node_type=NodeType.paragraph,
                title=None,
                content=blk.text,
                page_start=blk.page_number,
                page_end=blk.page_number,
                order_index=ordinal,
                level=_paragraph_level(),
                parent_id=_current_container_id(),
                bbox=blk.bbox,
                source_block_ids=_claim([blk.block_id]),
                metadata_={"source": "heuristic", "original_block_type": blk.block_type.value},
            )
            nodes.append(para_node)
            ordinal += 1
            continue

        # All other block types (e.g. discarded — shouldn't reach here because
        # they were filtered out, but guard anyway) are silently skipped.

    # ------------------------------------------------------------------ #
    # Step (H): update page_start / page_end for container nodes
    # ------------------------------------------------------------------ #
    # Build a mapping from node id → node (for mutation)
    node_by_id: dict[UUID, DocumentNodeOut] = {n.id: n for n in nodes}

    def _propagate_page_range() -> None:
        """Bottom-up propagation of page_start / page_end.

        Repeats until no changes occur (handles trees of any depth).
        """
        # Iterate until stable (handles arbitrary depth; typically 3-4 passes)
        for _pass in range(10):
            changed = False
            # Collect children ranges for each parent in this pass
            children_pages: dict[UUID, list[tuple[int, int]]] = {}
            for n in node_by_id.values():
                if n.parent_id is not None and n.parent_id in node_by_id:
                    if n.parent_id not in children_pages:
                        children_pages[n.parent_id] = []
                    children_pages[n.parent_id].append((n.page_start, n.page_end))

            for nid, pages in children_pages.items():
                node = node_by_id[nid]
                new_start = min(p[0] for p in pages)
                new_end = max(p[1] for p in pages)
                # Also honour the node's own initial page range (e.g. doc title)
                new_start = min(new_start, node.page_start)
                new_end = max(new_end, node.page_end)
                if node.page_start != new_start or node.page_end != new_end:
                    updated = node.model_copy(update={"page_start": new_start, "page_end": new_end})
                    node_by_id[nid] = updated
                    changed = True

            if not changed:
                break

        # Rebuild nodes list preserving order
        nodes[:] = sorted(node_by_id.values(), key=lambda n: n.order_index)

    _propagate_page_range()

    return HierarchyResult(
        document_id=document_id,
        chat_id=chat_id,
        nodes=nodes,
        references_start_index=references_start_index,
        appendix_start_index=appendix_start_index,
        heuristics_applied=sorted(heuristics_applied),
    )


__all__ = ["derive_hierarchy"]
