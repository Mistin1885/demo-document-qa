"""ORM bridge: ``SectionEnrichment`` → ``list[SummaryRow]``.

Phase 5.1 helper that flattens a ``SectionEnrichment`` domain object into a
list of ORM-shaped rows for the ``summaries`` table.  This module does **not**
import SQLAlchemy ORM or any database client — it returns typed Pydantic
``SummaryRow`` objects that the ingestion service can write directly.

Design rules (CLAUDE.md §12)
-----------------------------
- No raw SQL; no SQLAlchemy imports.
- All public functions have full type hints.
- No ``dict[str, Any]``; ``SummaryRow`` is a typed Pydantic model.
"""

from __future__ import annotations

import json
import uuid

from pydantic import BaseModel, ConfigDict

from app.enrichment.models import DocumentOverview, SectionEnrichment

# ---------------------------------------------------------------------------
# SummaryRow
# ---------------------------------------------------------------------------


class SummaryRow(BaseModel):
    """ORM-shaped representation of one row in the ``summaries`` table.

    This is a Pydantic data-transfer object — the ingestion service maps it
    to the SQLAlchemy ``Summary`` ORM model.

    ``kind`` values produced by :func:`to_summary_rows`:

    - ``section_detailed``
    - ``section_compact``
    - ``section_keywords``
    - ``section_entities``
    - ``section_claims``
    - ``section_methods``
    - ``section_limitations``
    - ``section_performance_facts``
    - ``section_definitions``
    """

    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID
    chat_id: uuid.UUID
    document_id: uuid.UUID
    source_node_id: uuid.UUID | None
    kind: str
    content: str
    """Serialized JSON string or plain text, depending on ``kind``."""
    keywords: list[str]
    entities: list[str]
    token_count: int


# ---------------------------------------------------------------------------
# Kind constants (single source of truth)
# ---------------------------------------------------------------------------

KIND_DETAILED = "section_detailed"
KIND_COMPACT = "section_compact"
KIND_KEYWORDS = "section_keywords"
KIND_ENTITIES = "section_entities"
KIND_CLAIMS = "section_claims"
KIND_METHODS = "section_methods"
KIND_LIMITATIONS = "section_limitations"
KIND_PERFORMANCE_FACTS = "section_performance_facts"
KIND_DEFINITIONS = "section_definitions"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_summary_rows(enrichment: SectionEnrichment) -> list[SummaryRow]:
    """Flatten *enrichment* into ``SummaryRow`` instances for the summaries table.

    Always produces the ``section_detailed`` and ``section_compact`` rows.
    Additional rows are produced only when the respective list is non-empty:

    - ``section_keywords``          — when ``enrichment.keywords`` is non-empty
    - ``section_entities``          — when ``enrichment.entities`` is non-empty
    - ``section_claims``            — when ``enrichment.claims`` is non-empty
    - ``section_methods``           — when ``enrichment.methods`` is non-empty
    - ``section_limitations``       — when ``enrichment.limitations`` is non-empty
    - ``section_performance_facts`` — when ``enrichment.performance_facts`` is non-empty
    - ``section_definitions``       — when ``enrichment.definitions`` is non-empty

    Parameters
    ----------
    enrichment:
        Output of :func:`app.enrichment.section.enrich_sections` for a single node.

    Returns
    -------
    list[SummaryRow]
        One row per active field.  Every ``kind`` in the returned list is
        unique (validated by callers in tests).
    """
    chat_id = enrichment.chat_id
    document_id = enrichment.document_id
    source_node_id = enrichment.node_id
    keywords = list(enrichment.keywords)
    entities = list(enrichment.entities)

    rows: list[SummaryRow] = []

    # --- always present ---
    rows.append(
        SummaryRow(
            id=uuid.uuid4(),
            chat_id=chat_id,
            document_id=document_id,
            source_node_id=source_node_id,
            kind=KIND_DETAILED,
            content=enrichment.detailed_summary,
            keywords=keywords,
            entities=entities,
            token_count=enrichment.token_count_estimate or enrichment.token_count,
        )
    )
    rows.append(
        SummaryRow(
            id=uuid.uuid4(),
            chat_id=chat_id,
            document_id=document_id,
            source_node_id=source_node_id,
            kind=KIND_COMPACT,
            content=enrichment.compact_summary,
            keywords=keywords,
            entities=entities,
            token_count=max(1, len(enrichment.compact_summary.split())),
        )
    )

    # --- conditional rows ---
    if enrichment.keywords:
        rows.append(
            SummaryRow(
                id=uuid.uuid4(),
                chat_id=chat_id,
                document_id=document_id,
                source_node_id=source_node_id,
                kind=KIND_KEYWORDS,
                content=json.dumps([{"term": kw, "weight": 1.0} for kw in enrichment.keywords]),
                keywords=keywords,
                entities=[],
                token_count=len(enrichment.keywords),
            )
        )

    if enrichment.entities:
        rows.append(
            SummaryRow(
                id=uuid.uuid4(),
                chat_id=chat_id,
                document_id=document_id,
                source_node_id=source_node_id,
                kind=KIND_ENTITIES,
                content=json.dumps(
                    [{"name": ent, "type": "concept"} for ent in enrichment.entities]
                ),
                keywords=[],
                entities=entities,
                token_count=len(enrichment.entities),
            )
        )

    if enrichment.claims:
        rows.append(
            SummaryRow(
                id=uuid.uuid4(),
                chat_id=chat_id,
                document_id=document_id,
                source_node_id=source_node_id,
                kind=KIND_CLAIMS,
                content=json.dumps(
                    [{"statement": c.text, "confidence": "medium"} for c in enrichment.claims]
                ),
                keywords=[],
                entities=[],
                token_count=sum(len(c.text.split()) for c in enrichment.claims),
            )
        )

    if enrichment.methods:
        rows.append(
            SummaryRow(
                id=uuid.uuid4(),
                chat_id=chat_id,
                document_id=document_id,
                source_node_id=source_node_id,
                kind=KIND_METHODS,
                content=json.dumps(
                    [
                        {"name": m.name, "role": "proposed", "summary": m.description}
                        for m in enrichment.methods
                    ]
                ),
                keywords=[],
                entities=[],
                token_count=sum(len(m.description.split()) for m in enrichment.methods),
            )
        )

    if enrichment.limitations:
        rows.append(
            SummaryRow(
                id=uuid.uuid4(),
                chat_id=chat_id,
                document_id=document_id,
                source_node_id=source_node_id,
                kind=KIND_LIMITATIONS,
                content=json.dumps(
                    [{"statement": lim.text, "scope": "other"} for lim in enrichment.limitations]
                ),
                keywords=[],
                entities=[],
                token_count=sum(len(lim.text.split()) for lim in enrichment.limitations),
            )
        )

    if enrichment.performance_facts:
        rows.append(
            SummaryRow(
                id=uuid.uuid4(),
                chat_id=chat_id,
                document_id=document_id,
                source_node_id=source_node_id,
                kind=KIND_PERFORMANCE_FACTS,
                content=json.dumps(
                    [
                        {
                            "metric": pf.metric,
                            "value": pf.value,
                            "context": pf.context,
                        }
                        for pf in enrichment.performance_facts
                    ]
                ),
                keywords=[],
                entities=[],
                token_count=len(enrichment.performance_facts),
            )
        )

    if enrichment.definitions:
        rows.append(
            SummaryRow(
                id=uuid.uuid4(),
                chat_id=chat_id,
                document_id=document_id,
                source_node_id=source_node_id,
                kind=KIND_DEFINITIONS,
                content=json.dumps(
                    [{"term": d.term, "definition": d.definition} for d in enrichment.definitions]
                ),
                keywords=[],
                entities=[],
                token_count=sum(len(d.definition.split()) for d in enrichment.definitions),
            )
        )

    return rows


# ---------------------------------------------------------------------------
# Document-level kind constants
# ---------------------------------------------------------------------------

KIND_DOC_OVERVIEW = "document_overview"
KIND_DOC_CONTRIBUTIONS = "document_contributions"
KIND_DOC_METHODS = "document_methods"
KIND_DOC_FINDINGS = "document_findings"
KIND_DOC_LIMITATIONS = "document_limitations"
KIND_DOC_DATASETS = "document_datasets"
KIND_DOC_METRICS = "document_metrics"
KIND_DOC_CONCLUSIONS = "document_conclusions"


def to_document_summary_rows(overview: DocumentOverview) -> list[SummaryRow]:
    """Flatten *overview* into ``SummaryRow`` instances for the summaries table.

    Always produces 8 rows (one per structured field):

    - ``document_overview``       — overview text (plain text)
    - ``document_contributions``  — JSON-encoded list
    - ``document_methods``        — JSON-encoded list
    - ``document_findings``       — JSON-encoded list
    - ``document_limitations``    — JSON-encoded list
    - ``document_datasets``       — JSON-encoded list
    - ``document_metrics``        — JSON-encoded list
    - ``document_conclusions``    — JSON-encoded list

    Parameters
    ----------
    overview:
        Output of :func:`app.enrichment.document_overview.enrich_document_overview`.

    Returns
    -------
    list[SummaryRow]
        Exactly 8 rows, one per kind.  Every ``kind`` is unique.

    Notes
    -----
    This function does NOT write to the database — it returns typed Pydantic
    objects for the ingestion service to persist.
    """
    chat_id = overview.chat_id
    document_id = overview.document_id
    token_count = overview.token_count_estimate

    rows: list[SummaryRow] = []

    # 1. overview (plain text)
    rows.append(
        SummaryRow(
            id=uuid.uuid4(),
            chat_id=chat_id,
            document_id=document_id,
            source_node_id=None,
            kind=KIND_DOC_OVERVIEW,
            content=overview.overview,
            keywords=[],
            entities=[],
            token_count=token_count,
        )
    )

    # 2. contributions (JSON list)
    rows.append(
        SummaryRow(
            id=uuid.uuid4(),
            chat_id=chat_id,
            document_id=document_id,
            source_node_id=None,
            kind=KIND_DOC_CONTRIBUTIONS,
            content=json.dumps(
                [
                    {
                        "title": c.title,
                        "summary": c.summary,
                        "source_section_ids": [str(sid) for sid in c.source_section_ids],
                    }
                    for c in overview.contributions
                ]
            ),
            keywords=[],
            entities=[],
            token_count=len(overview.contributions),
        )
    )

    # 3. methods (JSON list)
    rows.append(
        SummaryRow(
            id=uuid.uuid4(),
            chat_id=chat_id,
            document_id=document_id,
            source_node_id=None,
            kind=KIND_DOC_METHODS,
            content=json.dumps(
                [
                    {
                        "name": m.name,
                        "description": m.description,
                        "source_section_ids": [str(sid) for sid in m.source_section_ids],
                    }
                    for m in overview.methods
                ]
            ),
            keywords=[],
            entities=[],
            token_count=len(overview.methods),
        )
    )

    # 4. findings (JSON list)
    rows.append(
        SummaryRow(
            id=uuid.uuid4(),
            chat_id=chat_id,
            document_id=document_id,
            source_node_id=None,
            kind=KIND_DOC_FINDINGS,
            content=json.dumps(
                [
                    {
                        "statement": f.statement,
                        "evidence": f.evidence,
                        "source_section_ids": [str(sid) for sid in f.source_section_ids],
                    }
                    for f in overview.findings
                ]
            ),
            keywords=[],
            entities=[],
            token_count=len(overview.findings),
        )
    )

    # 5. limitations (JSON list)
    rows.append(
        SummaryRow(
            id=uuid.uuid4(),
            chat_id=chat_id,
            document_id=document_id,
            source_node_id=None,
            kind=KIND_DOC_LIMITATIONS,
            content=json.dumps(
                [
                    {
                        "text": lim.text,
                        "source_section_ids": [str(sid) for sid in lim.source_section_ids],
                    }
                    for lim in overview.limitations
                ]
            ),
            keywords=[],
            entities=[],
            token_count=len(overview.limitations),
        )
    )

    # 6. datasets (JSON list)
    rows.append(
        SummaryRow(
            id=uuid.uuid4(),
            chat_id=chat_id,
            document_id=document_id,
            source_node_id=None,
            kind=KIND_DOC_DATASETS,
            content=json.dumps(
                [
                    {
                        "name": ds.name,
                        "role": ds.role,
                        "size_hint": ds.size_hint,
                        "source_section_ids": [str(sid) for sid in ds.source_section_ids],
                    }
                    for ds in overview.datasets
                ]
            ),
            keywords=[],
            entities=[],
            token_count=len(overview.datasets),
        )
    )

    # 7. metrics (JSON list)
    rows.append(
        SummaryRow(
            id=uuid.uuid4(),
            chat_id=chat_id,
            document_id=document_id,
            source_node_id=None,
            kind=KIND_DOC_METRICS,
            content=json.dumps(
                [
                    {
                        "name": met.name,
                        "best_value": met.best_value,
                        "baseline_value": met.baseline_value,
                        "improvement": met.improvement,
                        "source_section_ids": [str(sid) for sid in met.source_section_ids],
                    }
                    for met in overview.metrics
                ]
            ),
            keywords=[],
            entities=[],
            token_count=len(overview.metrics),
        )
    )

    # 8. conclusions (JSON list)
    rows.append(
        SummaryRow(
            id=uuid.uuid4(),
            chat_id=chat_id,
            document_id=document_id,
            source_node_id=None,
            kind=KIND_DOC_CONCLUSIONS,
            content=json.dumps(
                [
                    {
                        "statement": con.statement,
                        "category": con.category,
                        "source_section_ids": [str(sid) for sid in con.source_section_ids],
                    }
                    for con in overview.conclusions
                ]
            ),
            keywords=[],
            entities=[],
            token_count=len(overview.conclusions),
        )
    )

    return rows


__all__ = ["SummaryRow", "to_summary_rows", "to_document_summary_rows"]
