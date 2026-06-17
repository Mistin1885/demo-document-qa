"""FastAPI router for the /chats/{chat_id}/facts resource (Phase 5.3).

Design rules (CLAUDE.md §2, §8, §12)
---------------------------------------
- All endpoints are scoped to ``chat_id`` from the URL path; the body / LLM
  never controls the isolation boundary.
- Fact search uses ``POST /chats/{chat_id}/facts/search`` with a
  ``FactFilter`` JSON body — no raw SQL, no dynamic query strings.
- ``GET /chats/{chat_id}/facts/{fact_id}`` returns 404 for cross-chat access
  (``FactNotFound`` → 404).
- ``POST /chats/{chat_id}/facts/extract`` triggers heuristic extraction from
  the document's MinerU output; returns 503 if the parsed output is absent.

No domain logic lives here — it all delegates to
``app.services.facts_service`` and ``app.enrichment.facts``.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi import status as http_status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.models.domain import StructuredFactCreate, StructuredFactRead
from app.services.facts_service import (
    FactFilter,
    get_fact,
    list_facts,
    persist_facts,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class FactSearchResponse(BaseModel):
    """Response envelope for POST /facts/search."""

    count: int
    items: list[StructuredFactRead]


# ---------------------------------------------------------------------------
# POST /chats/{chat_id}/facts/search
# ---------------------------------------------------------------------------


@router.post(
    "/search",
    response_model=FactSearchResponse,
    summary="Search structured facts with a restricted filter",
)
async def search_facts(
    chat_id: uuid.UUID,
    filter: FactFilter,  # noqa: A002  (shadowing built-in 'filter' intentional here)
    session: AsyncSession = Depends(get_session),
) -> FactSearchResponse:
    """Search structured facts within a chat using a restricted filter schema.

    The ``chat_id`` is always injected from the URL path; the body can never
    override it.  Any unknown field in the filter body results in a 422 error
    (``extra="forbid"`` on ``FactFilter``).
    """
    facts = await list_facts(session, chat_id=chat_id, filter=filter)
    return FactSearchResponse(count=len(facts), items=facts)


# ---------------------------------------------------------------------------
# GET /chats/{chat_id}/facts/{fact_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{fact_id}",
    response_model=StructuredFactRead,
    summary="Fetch a single structured fact",
)
async def get_one_fact(
    chat_id: uuid.UUID,
    fact_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> StructuredFactRead:
    """Fetch a single structured fact by ID.

    Returns 404 when the fact does not exist **or** belongs to a different
    chat (CLAUDE.md §2 isolation — callers must not learn whether a fact
    exists in another chat).
    """
    return await get_fact(session, chat_id=chat_id, fact_id=fact_id)


# ---------------------------------------------------------------------------
# POST /chats/{chat_id}/facts/extract
# ---------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    """Request body for the extract endpoint."""

    document_id: uuid.UUID
    use_llm: bool = False


class ExtractResponse(BaseModel):
    """Response from the extract endpoint."""

    document_id: uuid.UUID
    facts_extracted: int
    fact_ids: list[uuid.UUID]


@router.post(
    "/extract",
    response_model=ExtractResponse,
    status_code=http_status.HTTP_201_CREATED,
    summary="Extract and persist structured facts from a parsed document",
)
async def extract_and_persist_facts(
    chat_id: uuid.UUID,
    body: ExtractRequest,
    session: AsyncSession = Depends(get_session),
) -> ExtractResponse:
    """Trigger heuristic extraction from the document's MinerU output.

    The document must have already been parsed (MinerU output must exist under
    ``data/parsed/<doc_stem>/hybrid_auto/``).

    Returns 503 when the parsed output directory / files are not found.
    Returns 422 when ``document_id`` does not belong to this chat.
    """
    from fastapi import HTTPException

    from app.enrichment.facts import extract_structured_facts
    from app.parsing.hierarchy import derive_hierarchy
    from app.parsing.mapping import load_middle_json, map_middle_to_parsed_blocks
    from app.services.document_service import get_document

    # Verify document belongs to this chat (raises DocumentNotFound → 404 handled globally)
    doc = await get_document(session, chat_id, body.document_id)

    # Locate the MinerU output
    settings = get_settings()
    # Canonical path: {app_data_root}/parsed/<stem>/hybrid_auto/<stem>_middle.json
    # where stem is the document's original filename without extension.
    from pathlib import PurePosixPath

    parsed_root = Path(settings.app_data_root) / "parsed"
    stem = PurePosixPath(doc.original_filename).stem
    middle_path = parsed_root / stem / "hybrid_auto" / f"{stem}_middle.json"

    if not middle_path.exists():
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Document not parsed yet: MinerU output not found at {middle_path}",
        )

    # Load and map
    middle = load_middle_json(middle_path)
    blocks = map_middle_to_parsed_blocks(middle, chat_id=chat_id, document_id=body.document_id)
    hierarchy = derive_hierarchy(blocks, chat_id=chat_id, document_id=body.document_id)

    # Extract facts (heuristic only; use_llm is reserved for future LLM-augmented mode)
    facts_raw: list[StructuredFactCreate] = extract_structured_facts(hierarchy, blocks)

    # Clear source_node_id: the hierarchy nodes are in-memory and not persisted to
    # document_nodes, so the FK constraint would fail if we kept the UUID.
    facts_create: list[StructuredFactCreate] = [
        f.model_copy(update={"source_node_id": None}) for f in facts_raw
    ]

    # Persist (idempotent upsert)
    fact_ids = await persist_facts(session, facts_create, current_chat_id=chat_id)

    return ExtractResponse(
        document_id=body.document_id,
        facts_extracted=len(fact_ids),
        fact_ids=fact_ids,
    )
