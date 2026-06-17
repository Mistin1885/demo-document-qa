"""Structured facts service — Phase 5.3.

Provides:
- ``FactsFilter``       — Pydantic v2 restricted filter schema (LangGraph tool contract).
- ``query_facts``       — Async, isolation-enforced SELECT.
- ``persist_facts``     — Idempotent upsert with deterministic UUID-5 IDs.

Isolation contract (CLAUDE.md §2, §3)
---------------------------------------
- ``query_facts`` ignores ``filt.chat_id`` entirely; it always uses the
  ``current_chat_id`` keyword argument passed by the service layer.
- ``persist_facts`` always overwrites each fact's ``chat_id`` with
  ``current_chat_id`` before writing.
- The LLM / agent NEVER controls ``chat_id``; it is injected from
  ``AgentState`` by the tool caller.

SQL safety (CLAUDE.md §6, §12)
---------------------------------
- ``query_facts`` uses SQLAlchemy 2.x typed ``select()`` expressions only.
- No ``text()``, no f-string SQL, no dynamic ORDER BY strings.
- ``order_by`` is fixed to ``(StructuredFact.created_at, StructuredFact.id)``.

Idempotency
-----------
``StructuredFact.id`` is ``uuid5(NAMESPACE_OID, f"{document_id}:fact:{kind}:{key}:{page}:{seq}")``.
Running ``persist_facts`` twice with the same data produces the same rows
(ON CONFLICT DO UPDATE).

Design rules (CLAUDE.md §12)
------------------------------
- No FastAPI imports.
- No ``dict[str, Any]``.
- All public functions are async and fully type-annotated.
"""

from __future__ import annotations

import uuid
from typing import Annotated
from uuid import NAMESPACE_OID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain import FactKind, StructuredFactCreate, StructuredFactRead
from app.models.orm import StructuredFact

# ---------------------------------------------------------------------------
# FactsFilter — restricted query schema (LangGraph tool contract)
# ---------------------------------------------------------------------------

# Maximum number of key strings accepted per query
_MAX_KEYS = 50
# Maximum chars per individual key string
_MAX_KEY_LEN = 100
# Maximum number of unit strings accepted per query
_MAX_UNITS = 20


class FactsFilter(BaseModel):
    """Restricted filter schema for ``query_facts``.

    Every field has explicit upper/lower bounds.  ``extra="forbid"`` prevents
    silent injection of unlisted parameters by the LLM or any other caller.

    ``chat_id`` is **mandatory** in the schema (so the caller must supply it),
    but ``query_facts`` always **ignores** the provided value and overwrites it
    with the ``current_chat_id`` keyword argument.  This two-step design keeps
    the schema self-documenting while enforcing server-side isolation.
    """

    model_config = ConfigDict(extra="forbid")

    # Isolation field — mandatory in schema; service layer ignores filt.chat_id
    # and uses its own current_chat_id instead.
    chat_id: uuid.UUID

    document_ids: list[uuid.UUID] | None = None
    """Limit results to these document IDs (must all belong to current_chat_id)."""

    kinds: list[FactKind] | None = None
    """Filter by fact kind(s)."""

    keys: Annotated[list[str] | None, Field(default=None)]
    """Exact-match key filter; at most 50 entries, each ≤ 100 chars."""

    page_range: tuple[int, int] | None = None
    """Inclusive [min_page, max_page] filter (1-indexed)."""

    numeric_min: float | None = None
    """Lower bound (inclusive) on ``value.numeric``."""

    numeric_max: float | None = None
    """Upper bound (inclusive) on ``value.numeric``."""

    unit_in: Annotated[list[str] | None, Field(default=None)]
    """Allow-list of unit strings; at most 20 entries."""

    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)

    @field_validator("keys")
    @classmethod
    def _validate_keys(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if len(v) > _MAX_KEYS:
            raise ValueError(f"keys may contain at most {_MAX_KEYS} entries, got {len(v)}")
        for k in v:
            if len(k) > _MAX_KEY_LEN:
                raise ValueError(
                    f"each key must be ≤ {_MAX_KEY_LEN} chars; got {len(k)!r} chars for {k!r}"
                )
        return v

    @field_validator("unit_in")
    @classmethod
    def _validate_unit_in(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if len(v) > _MAX_UNITS:
            raise ValueError(f"unit_in may contain at most {_MAX_UNITS} entries, got {len(v)}")
        return v

    @field_validator("page_range")
    @classmethod
    def _validate_page_range(cls, v: tuple[int, int] | None) -> tuple[int, int] | None:
        if v is None:
            return v
        lo, hi = v
        if lo < 1:
            raise ValueError(f"page_range min must be ≥ 1, got {lo}")
        if hi < lo:
            raise ValueError(f"page_range max ({hi}) must be ≥ min ({lo})")
        return v


# ---------------------------------------------------------------------------
# Deterministic ID helper
# ---------------------------------------------------------------------------


def _fact_id(
    document_id: uuid.UUID,
    kind: str,
    key: str,
    page: int | None,
    seq: int,
) -> uuid.UUID:
    """Compute a deterministic UUID-5 for a structured fact.

    The seed string uniquely identifies the fact within a document so that
    re-running ingestion produces the same ``id`` and the upsert is idempotent.
    """
    page_str = str(page) if page is not None else "None"
    name = f"{document_id}:fact:{kind}:{key}:{page_str}:{seq}"
    return uuid5(NAMESPACE_OID, name)


# ---------------------------------------------------------------------------
# query_facts — read path (isolation-enforced)
# ---------------------------------------------------------------------------


async def query_facts(
    session: AsyncSession,
    *,
    current_chat_id: uuid.UUID,
    filt: FactsFilter,
) -> list[StructuredFactRead]:
    """Return structured facts filtered by ``filt``, always scoped to ``current_chat_id``.

    CLAUDE.md §2 isolation: ``filt.chat_id`` is **ignored**; only
    ``current_chat_id`` (injected by the service layer) is used in the SQL
    WHERE clause.

    Parameters
    ----------
    session:
        Active async DB session.
    current_chat_id:
        The authoritative chat scope, injected by the service / tool layer.
    filt:
        Caller-supplied filter (``filt.chat_id`` silently overwritten).

    Returns
    -------
    list[StructuredFactRead]
        Facts belonging to ``current_chat_id`` that satisfy all filters.
    """
    # Enforce isolation: ignore whatever chat_id the caller put in filt.
    effective = filt.model_copy(update={"chat_id": current_chat_id})

    stmt = select(StructuredFact).where(StructuredFact.chat_id == effective.chat_id)

    if effective.document_ids is not None:
        stmt = stmt.where(StructuredFact.document_id.in_(effective.document_ids))

    if effective.kinds is not None:
        stmt = stmt.where(StructuredFact.kind.in_(effective.kinds))

    if effective.keys is not None:
        stmt = stmt.where(StructuredFact.key.in_(effective.keys))

    if effective.page_range is not None:
        lo, hi = effective.page_range
        stmt = stmt.where(StructuredFact.page >= lo).where(StructuredFact.page <= hi)

    # Numeric range filtering via JSONB path (SQLAlchemy ORM expression, no raw SQL)
    if effective.numeric_min is not None:
        # Cast the JSONB numeric field to a float for comparison.
        # SQLAlchemy expression: value['numeric'].as_float()
        stmt = stmt.where(
            StructuredFact.value["numeric"].as_float() >= effective.numeric_min  # type: ignore[index]
        )

    if effective.numeric_max is not None:
        stmt = stmt.where(
            StructuredFact.value["numeric"].as_float() <= effective.numeric_max  # type: ignore[index]
        )

    if effective.unit_in is not None:
        stmt = stmt.where(StructuredFact.unit.in_(effective.unit_in))

    # Fixed ordering — never dynamic
    stmt = stmt.order_by(StructuredFact.created_at, StructuredFact.id)
    stmt = stmt.offset(effective.offset).limit(effective.limit)

    result = await session.execute(stmt)
    rows = result.scalars().all()
    return [StructuredFactRead.model_validate(row) for row in rows]


# ---------------------------------------------------------------------------
# persist_facts — write path (idempotent upsert)
# ---------------------------------------------------------------------------


async def persist_facts(
    session: AsyncSession,
    facts: list[StructuredFactCreate],
    *,
    current_chat_id: uuid.UUID,
) -> list[uuid.UUID]:
    """Idempotently upsert structured facts into the database.

    - ``chat_id`` of every fact is overwritten with ``current_chat_id``.
    - ``StructuredFact.id`` is computed as ``uuid5`` from the fact's content
      so that repeated runs produce the same UUID and the upsert is a no-op
      for unchanged facts.
    - On conflict (same ``id``) the mutable fields (``value``, ``unit``,
      ``context_excerpt``, ``page``) are updated so that corrections are
      idempotent.

    Parameters
    ----------
    session:
        Active async DB session.
    facts:
        Facts to persist; ``chat_id`` is always overwritten.
    current_chat_id:
        Authoritative chat scope (service layer injection point).

    Returns
    -------
    list[uuid.UUID]
        IDs of all upserted rows in input order.
    """
    if not facts:
        return []

    ids: list[uuid.UUID] = []
    for seq, fact in enumerate(facts):
        # Enforce isolation: always use current_chat_id
        effective_chat_id = current_chat_id
        fact_id = _fact_id(fact.document_id, fact.kind, fact.key, fact.page, seq)
        ids.append(fact_id)

        # Check if row already exists
        existing = await session.get(StructuredFact, fact_id)
        if existing is not None:
            # Update mutable fields
            existing.value = fact.value.model_dump()
            existing.unit = fact.unit
            existing.context_excerpt = fact.context_excerpt
            existing.page = fact.page
        else:
            row = StructuredFact(
                id=fact_id,
                chat_id=effective_chat_id,
                document_id=fact.document_id,
                source_node_id=fact.source_node_id,
                kind=fact.kind,
                key=fact.key,
                value=fact.value.model_dump(),
                unit=fact.unit,
                context_excerpt=fact.context_excerpt,
                page=fact.page,
            )
            session.add(row)

    await session.flush()
    return ids


__all__ = ["FactsFilter", "query_facts", "persist_facts"]
