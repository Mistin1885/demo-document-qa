"""Figure / table narration for ingestion-time multimodal enrichment.

What this module does
---------------------
For every ``ParsedBlock`` whose ``block_type`` is ``image`` or ``table``, we:

1. Resolve the renamed image basename to an absolute path under the
   MinerU output ``images/`` directory.
2. Compose a *same-page text excerpt* by concatenating every text-like block
   on the same page (paragraph / title / ref_text / equation + the figure's
   own caption + footnote).
3. Send (image, system prompt, user message) to the VLM:
   - For tables: ask for a clean HTML table reconstruction.
   - For images: ask for two natural-language paragraphs describing the image
     and its semantic role w.r.t. the same-page text.
4. Return a list of ``FigureNarration`` items.  Downstream encoders embed
   ``narrative_text`` together with the page text so retrieval can match
   either modality.

Why same-page text
------------------
Following the reference implementation
(``ref/src/app/modules/document_parser/infra/vision/image_narrator.py``), the
same-page text gives the VLM enough context to describe *function* and not
just surface content, which markedly improves retrievability when users ask
"什麼圖在講 X" style questions whose wording never appears verbatim in OCR.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from app.parsing.models import BlockType, ParsedBlock
from app.parsing.vlm_client import VLMClient, VLMError, encode_path_to_b64

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TABLE_SYSTEM_PROMPT = """You convert table images into clean, structured HTML tables for downstream RAG indexing.

Rules:
- Output ONE <table>...</table> element only — no surrounding prose, no <html>/<body> wrappers.
- Preserve the original column / row order. Merge cells only if the source clearly does.
- If a cell is empty in the source, write an empty <td></td>; do NOT invent text.
- Preserve units, abbreviations and footnote markers exactly as shown.
- If the image is not actually a table, return a single <table><tr><td>NON_TABLE_IMAGE</td></tr></table>.
- Output language: keep cell text exactly as in the image. Do NOT translate.
"""

TABLE_USER_PROMPT = """Reconstruct this table image as clean HTML.
Same-page text (for disambiguation only — do NOT invent rows that aren't visible):

{page_text}
"""

FIGURE_SYSTEM_PROMPT = """You write RAG-ready narrative descriptions for images, figures, charts, and diagrams found in academic / technical documents.

Your output MUST contain exactly two paragraphs separated by a blank line:

Paragraph 1 — describe ONLY what is visible in the image: layout, axes,
labels, symbols, plotted curves, depicted objects, colours, schematic blocks,
arrows, equations, etc. Do not speculate beyond the image.

Paragraph 2 — explain the image's semantic role w.r.t. the same-page text:
what it is meant to show, which concepts / quantities / methods it relates
to, and what retrievable topics it would answer. Stay grounded in the
provided text; if the connection is unclear, keep this paragraph short.

Style: prefer plain English, mention concrete entities by name when visible.
Do not use bullet lists, headers, or markdown.
"""

FIGURE_USER_PROMPT = """Image kind: {figure_kind}

Same-page text (auxiliary context):

{page_text}
"""

# Tables that come back too short / clearly broken get a fall-back narration
TABLE_FALLBACK_KIND = "table"
FIGURE_FALLBACK_KIND = "figure"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FigureNarration:
    """One narration produced from an image / table ``ParsedBlock``."""

    block_id: UUID
    document_id: UUID
    chat_id: UUID
    page_number: int
    reading_order: int
    block_type: BlockType  # image or table
    image_path: str
    """Renamed basename as stored in MinerU's ``images/`` directory."""

    narrative_text: str
    """The actual VLM output: HTML for tables, two paragraphs for figures."""

    caption: str | None = None
    footnote: str | None = None
    page_text_excerpt: str = ""
    """Same-page text used as context for the VLM call (truncated)."""

    error: str | None = None
    """When the VLM call failed, ``narrative_text`` is empty and ``error``
    holds a short explanation (kept so the ingestion logs can audit)."""

    extra: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEXT_LIKE = frozenset(
    {
        BlockType.paragraph,
        BlockType.title,
        BlockType.text,
        BlockType.ref_text,
        BlockType.equation,
        BlockType.image_caption,
        BlockType.image_footnote,
        BlockType.table_caption,
        BlockType.table_footnote,
    }
)

_PAGE_TEXT_LIMIT_CHARS = 4000


def _collect_page_text(blocks: list[ParsedBlock], page_number: int) -> str:
    """Concatenate all text-like blocks on the same page, ordered by reading order."""
    same_page = [b for b in blocks if b.page_number == page_number]
    same_page.sort(key=lambda b: b.reading_order)
    parts: list[str] = []
    for blk in same_page:
        if blk.block_type in _TEXT_LIKE and blk.text.strip():
            parts.append(blk.text.strip())
    text = "\n".join(parts)
    if len(text) > _PAGE_TEXT_LIMIT_CHARS:
        return text[:_PAGE_TEXT_LIMIT_CHARS] + " …"
    return text


def _resolve_image_path(images_dir: Path, basename: str) -> Path | None:
    """Resolve a renamed image basename to an absolute file path under ``images_dir``."""
    if not basename:
        return None
    candidate = images_dir / Path(basename).name
    if candidate.exists():
        return candidate
    # MinerU sometimes stores image_path as "images/foo.jpg"; tolerate that.
    candidate2 = images_dir.parent / basename
    if candidate2.exists():
        return candidate2
    return None


def _figure_kind(block: ParsedBlock) -> str:
    if block.block_type == BlockType.table:
        return "table"
    img = block.image
    cap = (img.caption if img else None) or ""
    cap_low = cap.lower()
    if any(tok in cap_low for tok in ("algorithm", "pseudo", "listing")):
        return "algorithm"
    if any(tok in cap_low for tok in ("eq.", "equation")):
        return "equation"
    if any(tok in cap_low for tok in ("flowchart", "architecture", "pipeline", "framework")):
        return "diagram"
    if any(tok in cap_low for tok in ("table",)):
        return "table"
    if any(tok in cap_low for tok in ("chart", "plot", "curve", "histogram", "graph")):
        return "chart"
    return "figure"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def narrate_blocks(
    blocks: list[ParsedBlock],
    *,
    images_dir: Path,
    client: VLMClient | None = None,
    max_concurrency: int = 2,
) -> list[FigureNarration]:
    """Run figure / table narration over every image/table block in the document.

    Parameters
    ----------
    blocks:
        Full ``ParsedBlock`` list for one document (mixed types).  Only
        ``image`` and ``table`` blocks trigger VLM calls; the rest are used as
        same-page context.
    images_dir:
        ``<output_dir>/images`` from ``MinerUParseResult.output_dir``.
        Renamed basenames in ``ParsedBlock.image.image_path`` resolve against
        this directory.
    client:
        Optional preconfigured ``VLMClient``.  Defaults to one read from
        ``app.config``.
    max_concurrency:
        VLM inference is expensive; default 2 means at most two requests run
        in flight against the endpoint.
    """
    vlm = client or VLMClient()
    sem = asyncio.Semaphore(max_concurrency)

    visual_blocks = [
        b for b in blocks if b.block_type in (BlockType.image, BlockType.table)
    ]
    if not visual_blocks:
        return []

    async def _one(block: ParsedBlock) -> FigureNarration | None:
        if block.block_type == BlockType.image:
            img = block.image
            basename = (img.image_path if img else "") or ""
            caption = img.caption if img else None
            footnote = img.footnote if img else None
        else:  # table
            tbl = block.table
            basename = (tbl.image_path if tbl else "") or ""
            caption = tbl.caption if tbl else None
            footnote = tbl.footnote if tbl else None
            # Tables can be referenced either by their html (already rich) or
            # by a sibling image crop emitted from MinerU's table_body span.
            # When we have a crop we re-run the VLM to get a cleaner HTML
            # reconstruction; otherwise we skip and let the raw_block encoder
            # index the existing html.

        image_path = _resolve_image_path(images_dir, basename) if basename else None
        if image_path is None and block.block_type == BlockType.image:
            return FigureNarration(
                block_id=block.block_id,
                document_id=block.document_id,
                chat_id=block.chat_id,
                page_number=block.page_number,
                reading_order=block.reading_order,
                block_type=block.block_type,
                image_path=basename,
                narrative_text="",
                caption=caption,
                footnote=footnote,
                error=f"image file not found: {basename}",
            )

        page_text = _collect_page_text(blocks, block.page_number)

        # Tables: if we have an image crop, run the table-HTML extraction; if
        # we only have the MinerU-rendered html_body, skip the VLM (the html
        # is already structured and will be indexed via raw_block).
        if block.block_type == BlockType.table:
            if image_path is None:
                return None  # nothing to narrate beyond raw html
            system_prompt = TABLE_SYSTEM_PROMPT
            user_prompt = TABLE_USER_PROMPT.format(page_text=page_text or "(no same-page text)")
            kind = "table"
        else:
            system_prompt = FIGURE_SYSTEM_PROMPT
            kind = _figure_kind(block)
            user_prompt = FIGURE_USER_PROMPT.format(
                figure_kind=kind,
                page_text=page_text or "(no same-page text)",
            )

        if image_path is None:
            return None

        try:
            image_b64 = encode_path_to_b64(image_path, vlm.settings)
        except FileNotFoundError as exc:
            return FigureNarration(
                block_id=block.block_id,
                document_id=block.document_id,
                chat_id=block.chat_id,
                page_number=block.page_number,
                reading_order=block.reading_order,
                block_type=block.block_type,
                image_path=basename,
                narrative_text="",
                caption=caption,
                footnote=footnote,
                page_text_excerpt=page_text,
                error=str(exc),
            )

        async with sem:
            try:
                narration = await vlm.call_with_image(
                    system_prompt=system_prompt,
                    user_text=user_prompt,
                    image_b64=image_b64,
                )
            except VLMError as exc:
                logger.warning(
                    "VLM narration failed for block %s (page %d): %s",
                    block.block_id,
                    block.page_number,
                    exc,
                )
                return FigureNarration(
                    block_id=block.block_id,
                    document_id=block.document_id,
                    chat_id=block.chat_id,
                    page_number=block.page_number,
                    reading_order=block.reading_order,
                    block_type=block.block_type,
                    image_path=basename,
                    narrative_text="",
                    caption=caption,
                    footnote=footnote,
                    page_text_excerpt=page_text,
                    error=str(exc),
                )

        return FigureNarration(
            block_id=block.block_id,
            document_id=block.document_id,
            chat_id=block.chat_id,
            page_number=block.page_number,
            reading_order=block.reading_order,
            block_type=block.block_type,
            image_path=basename,
            narrative_text=narration,
            caption=caption,
            footnote=footnote,
            page_text_excerpt=page_text,
            extra={"figure_kind": kind},
        )

    results = await asyncio.gather(*(_one(b) for b in visual_blocks))
    return [r for r in results if r is not None]


__all__ = [
    "FigureNarration",
    "narrate_blocks",
]
