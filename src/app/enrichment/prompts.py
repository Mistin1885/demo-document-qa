"""Prompt builders for Section-level enrichment (Phase 5.1).

Design rules (CLAUDE.md §12)
-----------------------------
- No FastAPI imports; no SQLAlchemy imports; no LLM-specific imports.
- ``chat_id`` is **never** included in any prompt (LLM has no need for it).
- All helpers are pure functions — no global state.
- Max input per prompt is ``_MAX_CONTENT_CHARS`` characters (content is
  truncated per-paragraph with a ``[truncated]`` marker to prevent token
  overflow).
"""

from __future__ import annotations

from uuid import UUID

from app.parsing.models import BlockType, DocumentNodeOut, ParsedBlock
from app.providers.base import ChatMessage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_CONTENT_CHARS = 8_000
"""Hard limit on total content text sent to the LLM per section prompt."""

_SYSTEM_PROMPT = """\
You are a scientific paper assistant.
Analyse the provided section text and respond with ONLY a JSON object that
matches the following schema (no markdown fences, no extra keys):

{
  "detailed_summary": "<2-6 sentence summary that can be read independently>",
  "compact_summary": "<1-2 sentence summary for use in a chat manifest>",
  "keywords": [{"term": "<str>", "weight": <0.0-1.0>, "source_block_ids": []}],
  "entities": [
    {
      "name": "<str>",
      "type": "<model|dataset|metric|method|organization|person|concept|other>",
      "canonical": "<str or null>",
      "source_block_ids": []
    }
  ],
  "definitions": [{"term": "<str>", "definition": "<str>", "source_block_ids": []}],
  "claims": [{"statement": "<str>", "confidence": "<low|medium|high>", "source_block_ids": []}],
  "methods": [
    {
      "name": "<str>",
      "role": "<proposed|baseline|ablation|prior_work>",
      "summary": "<str>",
      "source_block_ids": []
    }
  ],
  "limitations": [
    {
      "statement": "<str>",
      "scope": "<dataset|method|experiment|scope|other>",
      "source_block_ids": []
    }
  ],
  "performance_facts": [
    {
      "metric": "<str>",
      "value": <float>,
      "unit": "<str or null>",
      "method": "<str or null>",
      "dataset": "<str or null>",
      "baseline": "<str or null>",
      "improvement": "<str or null>",
      "source_block_ids": []
    }
  ]
}

Rules:
- Every list may be empty ([]) if no items are found.
- For source_block_ids, use the block IDs provided in the user message (UUID strings).
  If unsure which block a fact came from, leave source_block_ids as [].
- Do not invent information; only use what is in the provided text.
- Output ONLY the JSON object — no preamble, no markdown.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate *text* to *max_chars* characters, appending ``[truncated]``."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " [truncated]"


def build_section_user_message(
    node: DocumentNodeOut,
    blocks: list[ParsedBlock],
) -> str:
    """Build the user-turn message for section enrichment.

    Parameters
    ----------
    node:
        The section / subsection / appendix node to enrich.
    blocks:
        All ``ParsedBlock`` instances whose ``block_id`` is in
        ``node.source_block_ids``.  Blocks not found in ``node.source_block_ids``
        are silently ignored.

    Returns
    -------
    str
        A formatted text block with the section title, page range, and
        per-block content (block_id + text / caption / LaTeX).
        Total content is capped at ``_MAX_CONTENT_CHARS`` characters.
    """
    title = node.title or "(untitled section)"
    header = (
        f"Section: {title}\n"
        f"Pages: {node.page_start}–{node.page_end}\n\n"
        "Blocks (each prefixed with its block_id):\n"
    )

    node_block_ids: frozenset[UUID] = frozenset(node.source_block_ids)
    # Build block index for fast lookup
    block_map: dict[UUID, ParsedBlock] = {b.block_id: b for b in blocks}

    parts: list[str] = [header]
    total_chars = len(header)

    for bid in node.source_block_ids:
        blk = block_map.get(bid)
        if blk is None or bid not in node_block_ids:
            continue

        # Build the text representation for this block
        if blk.block_type in (
            BlockType.paragraph,
            BlockType.text,
            BlockType.title,
            BlockType.ref_text,
        ):
            raw = blk.text.strip()
        elif blk.block_type == BlockType.equation and blk.equation_latex:
            raw = f"[equation] {blk.equation_latex.strip()}"
        elif blk.block_type == BlockType.image and blk.image:
            raw = f"[figure] {blk.image.caption or '(no caption)'}"
        elif blk.block_type == BlockType.table and blk.table:
            raw = f"[table] {blk.table.caption or '(no caption)'}"
        else:
            continue  # skip empty / discarded blocks

        if not raw:
            continue

        remaining = _MAX_CONTENT_CHARS - total_chars
        if remaining <= 0:
            parts.append(f"\n[block {bid}]: [truncated — content limit reached]")
            break

        block_text = _truncate_text(raw, remaining)
        line = f"\n[block {bid}]: {block_text}"
        parts.append(line)
        total_chars += len(line)

    return "".join(parts)


def build_section_messages(
    node: DocumentNodeOut,
    blocks: list[ParsedBlock],
) -> list[ChatMessage]:
    """Return the full ``[system, user]`` message list for a section enrichment call.

    Parameters
    ----------
    node:
        Section / subsection / appendix node.
    blocks:
        All ``ParsedBlock`` instances associated with *node*.

    Returns
    -------
    list[ChatMessage]
        Two-element list: system prompt + user message.
    """
    return [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user", content=build_section_user_message(node, blocks)),
    ]


_DOCUMENT_SYSTEM_PROMPT = """\
[DOCUMENT_OVERVIEW]
You are a scientific paper assistant.
Analyse the provided paper sections and respond with ONLY a JSON object that
matches the following schema (no markdown fences, no extra keys):

{
  "overview": "<5-10 sentence paper-wide overview>",
  "contributions": [
    {
      "title": "<str>",
      "summary": "<1-2 sentence description>",
      "source_section_ids": ["<uuid>"]
    }
  ],
  "methods": [
    {
      "name": "<str>",
      "description": "<str>",
      "source_section_ids": ["<uuid>"]
    }
  ],
  "findings": [
    {
      "statement": "<str>",
      "evidence": "<str or null>",
      "source_section_ids": ["<uuid>"]
    }
  ],
  "limitations": [
    {
      "text": "<str>",
      "source_section_ids": ["<uuid>"]
    }
  ],
  "datasets": [
    {
      "name": "<str>",
      "role": "<training|evaluation|benchmark|ablation|other>",
      "size_hint": "<str or null>",
      "source_section_ids": ["<uuid>"]
    }
  ],
  "metrics": [
    {
      "name": "<str>",
      "best_value": <float or null>,
      "baseline_value": <float or null>,
      "improvement": "<str or null>",
      "source_section_ids": ["<uuid>"]
    }
  ],
  "conclusions": [
    {
      "statement": "<str>",
      "category": "<finding|future_work|caveat|claim>",
      "source_section_ids": ["<uuid>"]
    }
  ]
}

Rules:
- Every list may be empty ([]) if no items are found.
- For source_section_ids, use the section node IDs provided in the user message.
  If unsure, leave source_section_ids as [].
- Do not invent information; only use what is in the provided text.
- Output ONLY the JSON object — no preamble, no markdown.
"""


def build_document_system_prompt() -> str:
    """Return the system prompt for document-level enrichment.

    The prompt includes the ``[DOCUMENT_OVERVIEW]`` marker used by
    ``FixtureChatProvider`` to route to the ``document_default`` fixture.

    Returns
    -------
    str
        The full system prompt string.
    """
    return _DOCUMENT_SYSTEM_PROMPT


def build_document_user_message(
    sections: list[tuple[str, str, int, int, str]],
    abstract_text: str | None = None,
    entities_top: list[str] | None = None,
    methods_top: list[str] | None = None,
    limitations_all: list[str] | None = None,
    performance_facts_all: list[str] | None = None,
    doc_title: str | None = None,
    authors_excerpt: str | None = None,
) -> str:
    """Build the user-turn message for document-level enrichment.

    Parameters
    ----------
    sections:
        List of ``(section_node_id, title, page_start, page_end, compact_summary)``
        tuples, one per section enrichment.  Uses **compact** summaries to
        stay within token budget.
    abstract_text:
        Optional abstract text to prepend.
    entities_top:
        Top-20 deduplicated entity names across all sections.
    methods_top:
        Top-10 deduplicated method names across all sections.
    limitations_all:
        All limitation texts (with section node IDs in the section list).
    performance_facts_all:
        All performance fact strings.
    doc_title:
        Title of the paper (from hierarchy or caller).
    authors_excerpt:
        First 200 chars of the authors string.

    Returns
    -------
    str
        Formatted user message.
    """
    parts: list[str] = []

    if doc_title:
        parts.append(f"Paper title: {doc_title}")
    if authors_excerpt:
        parts.append(f"Authors: {authors_excerpt[:200]}")

    if abstract_text:
        parts.append(f"\nAbstract:\n{abstract_text[:1500]}")

    parts.append("\nSections (node_id | title | pages | compact_summary):")
    for node_id_str, title, pg_start, pg_end, compact in sections:
        parts.append(f"  [{node_id_str}] {title} (p{pg_start}-{pg_end}): {compact}")

    if entities_top:
        parts.append(f"\nTop entities: {', '.join(entities_top[:20])}")
    if methods_top:
        parts.append(f"Top methods: {', '.join(methods_top[:10])}")
    if limitations_all:
        parts.append(f"Limitations: {'; '.join(limitations_all)}")
    if performance_facts_all:
        parts.append(f"Performance facts: {'; '.join(performance_facts_all)}")

    parts.append("\nGenerate the DocumentOverview JSON object covering the full paper.")

    return "\n".join(parts)


__all__ = [
    "build_section_messages",
    "build_section_user_message",
    "build_document_system_prompt",
    "build_document_user_message",
]
