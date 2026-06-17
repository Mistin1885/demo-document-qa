"""Enrichment service — persist section-level enrichments to the database.

Phase 5.1 — Section summaries are written as ``summaries`` ORM rows.

Design contract (CLAUDE.md §12)
--------------------------------
- No FastAPI imports; no HTTP-specific types.
- All public functions are async and accept a SQLAlchemy ``AsyncSession``.
- ``chat_id`` is taken exclusively from the ``SectionEnrichment`` object
  (which inherits it from ``hierarchy.chat_id``); it is never passed in by
  an external caller.
- Writes are **idempotent**: if a ``(chat_id, document_id, source_node_id,
  kind)`` row already exists, update ``content / keywords / entities /
  token_count`` instead of inserting a duplicate.
- No raw SQL strings — only SQLAlchemy ORM/Core expressions.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.enrichment.models import DocumentEnrichment, SectionEnrichment
from app.models.orm import Summary

# Kinds written per enrichment
_KIND_DETAILED = "section_detailed"
_KIND_COMPACT = "section_compact"


async def _upsert_summary(
    session: AsyncSession,
    *,
    chat_id: uuid.UUID,
    document_id: uuid.UUID,
    source_node_id: uuid.UUID | None,
    kind: str,
    content: str,
    keywords: list[str],
    entities: list[str],
    token_count: int,
) -> uuid.UUID:
    """Insert or update a single Summary row and return its id.

    Idempotent on ``(chat_id, document_id, source_node_id, kind)``.
    Handles ``source_node_id=None`` correctly (uses ``IS NULL`` in SQL).
    """
    from sqlalchemy import ColumnElement, null

    node_filter: ColumnElement[bool]
    if source_node_id is None:
        node_filter = Summary.source_node_id.is_(null())
    else:
        node_filter = Summary.source_node_id == source_node_id

    stmt = select(Summary).where(
        Summary.chat_id == chat_id,
        Summary.document_id == document_id,
        node_filter,
        Summary.kind == kind,
    )
    result = await session.execute(stmt)
    existing: Summary | None = result.scalar_one_or_none()

    if existing is not None:
        existing.content = content
        existing.keywords = keywords  # type: ignore[assignment]
        existing.entities = entities  # type: ignore[assignment]
        existing.token_count = token_count
        await session.flush()
        return existing.id  # type: ignore[return-value]

    row = Summary(
        id=uuid.uuid4(),
        chat_id=chat_id,
        document_id=document_id,
        source_node_id=source_node_id,
        kind=kind,
        content=content,
        keywords=keywords,  # type: ignore[arg-type]
        entities=entities,  # type: ignore[arg-type]
        token_count=token_count,
        created_at=datetime.now(UTC).replace(tzinfo=None),
    )
    session.add(row)
    await session.flush()
    return row.id  # type: ignore[return-value]


async def persist_section_summaries(
    session: AsyncSession,
    enrichments: list[SectionEnrichment],
) -> list[uuid.UUID]:
    """Persist section enrichments to the ``summaries`` table.

    For each ``SectionEnrichment``, two ``Summary`` rows are written:

    - ``kind="section_detailed"`` — ``detailed_summary``
    - ``kind="section_compact"`` — ``compact_summary``

    Parameters
    ----------
    session:
        Active async DB session (caller commits / rolls back).
    enrichments:
        Output of ``enrich_sections()``.

    Returns
    -------
    list[uuid.UUID]
        IDs of all ``Summary`` rows written (2 per enrichment, in order:
        detailed then compact, interleaved per enrichment).

    Notes
    -----
    ``chat_id`` is taken exclusively from each ``SectionEnrichment`` object;
    it is never accepted from an external argument.
    """
    summary_ids: list[uuid.UUID] = []

    for enrichment in enrichments:
        # Detailed summary
        detailed_id = await _upsert_summary(
            session,
            chat_id=enrichment.chat_id,
            document_id=enrichment.document_id,
            source_node_id=enrichment.node_id,
            kind=_KIND_DETAILED,
            content=enrichment.detailed_summary,
            keywords=enrichment.keywords,
            entities=enrichment.entities,
            token_count=enrichment.token_count,
        )
        summary_ids.append(detailed_id)

        # Compact summary
        compact_id = await _upsert_summary(
            session,
            chat_id=enrichment.chat_id,
            document_id=enrichment.document_id,
            source_node_id=enrichment.node_id,
            kind=_KIND_COMPACT,
            content=enrichment.compact_summary,
            keywords=enrichment.keywords,
            entities=enrichment.entities,
            token_count=enrichment.token_count,
        )
        summary_ids.append(compact_id)

    return summary_ids


# ---------------------------------------------------------------------------
# Phase 5.2 — Document-level summary persistence
# ---------------------------------------------------------------------------

_KIND_DOC_OVERVIEW = "document_overview"
_KIND_ABSTRACT = "chapter_summary"


async def persist_document_summaries(
    session: AsyncSession,
    enrichment: DocumentEnrichment,
    *,
    abstract_node_id: uuid.UUID | None,
) -> list[uuid.UUID]:
    """Persist document-level enrichment to the ``summaries`` table.

    Writes at most two ``Summary`` rows:

    - ``kind="document_overview"`` — always written.
    - ``kind="chapter_summary"``   — written only when both
      ``enrichment.abstract_summary`` is not ``None`` and
      ``abstract_node_id`` is not ``None``.

    Parameters
    ----------
    session:
        Active async DB session (caller commits / rolls back).
    enrichment:
        Output of ``enrich_document()``.
    abstract_node_id:
        The ``DocumentNodeOut.id`` of the abstract node.  Pass ``None`` when
        the document has no abstract node.

    Returns
    -------
    list[uuid.UUID]
        IDs of written ``Summary`` rows (1 or 2 entries).

    Notes
    -----
    - Idempotent: ``(chat_id, document_id, source_node_id, kind)`` is the
      unique key; existing rows are updated instead of duplicated.
    - ``chat_id`` is taken from ``enrichment.chat_id`` (which mirrors
      ``hierarchy.chat_id``); it is never accepted from an external argument.
    - ``keywords`` for the document_overview row contains the union of
      ``main_methods`` and ``main_technologies`` (top-20 deduped).
    - ``entities`` for the document_overview row contains the union of
      ``main_datasets`` and ``main_metrics`` (top-20 deduped).
    """
    chat_id = enrichment.chat_id
    document_id = enrichment.document_id

    # Build combined keyword / entity lists for the overview row
    kw_raw = list(enrichment.main_methods) + list(enrichment.main_technologies)
    kw_seen: dict[str, None] = {}
    for kw in kw_raw:
        kw_seen[kw.strip()] = None
    keywords: list[str] = sorted(kw_seen.keys())[:20]

    ent_raw = list(enrichment.main_datasets) + list(enrichment.main_metrics)
    ent_seen: dict[str, None] = {}
    for ent in ent_raw:
        ent_seen[ent.strip()] = None
    entities: list[str] = sorted(ent_seen.keys())[:20]

    token_count = enrichment.token_count_estimate

    summary_ids: list[uuid.UUID] = []

    # Row 1: document_overview (source_node_id = None)
    overview_id = await _upsert_summary(
        session,
        chat_id=chat_id,
        document_id=document_id,
        source_node_id=None,  # type: ignore[arg-type]
        kind=_KIND_DOC_OVERVIEW,
        content=enrichment.document_overview,
        keywords=keywords,
        entities=entities,
        token_count=token_count,
    )
    summary_ids.append(overview_id)

    # Row 2: abstract chapter_summary (only when available)
    if enrichment.abstract_summary is not None and abstract_node_id is not None:
        abstract_id = await _upsert_summary(
            session,
            chat_id=chat_id,
            document_id=document_id,
            source_node_id=abstract_node_id,
            kind=_KIND_ABSTRACT,
            content=enrichment.abstract_summary,
            keywords=[],
            entities=[],
            token_count=max(1, len(enrichment.abstract_summary.split())),
        )
        summary_ids.append(abstract_id)

    return summary_ids
