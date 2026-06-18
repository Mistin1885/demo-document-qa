"""Tool: expand_evidence — expand a specific EvidenceItem to its neighbors.

Contract (CLAUDE.md §8):
- Looks up the EvidenceItem by evidence_id in state.evidence_items.
- Then fetches sibling / parent / page-neighbor DocumentNodes via ORM.
- All queries scoped to state.chat_id (isolation enforced).
- ExpandEvidenceParams has NO chat_id field (extra="forbid").
- Returns new EvidenceItems (neighbors not already in state).

No FastAPI; no dict[str, Any]; fully async.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.agent.budget import ContextBudgetManager
from app.agent.state import (
    AgentError,
    AgentState,
    EvidenceItem,
    ToolCallRecord,
    make_evidence_id,
)
from app.agent.tools._invocation import ToolDeps, ToolInvocation
from app.agent.tools._models import ExpandEvidenceParams
from app.models.orm import DocumentNode


async def expand_evidence(
    state: AgentState,
    params: ExpandEvidenceParams,
    *,
    deps: ToolDeps,
) -> ToolInvocation:
    """Expand an EvidenceItem to adjacent DocumentNodes (section, page, paragraphs).

    Isolation: all ORM queries are scoped to state.chat_id.
    """
    call_id = str(uuid.uuid4())
    budget_mgr = ContextBudgetManager()
    errors: list[AgentError] = []
    evidence: list[EvidenceItem] = []

    try:
        # Find the seed evidence item
        seed = next(
            (ev for ev in state.evidence_items if ev.evidence_id == params.evidence_id),
            None,
        )
        if seed is None:
            raise ValueError(
                f"evidence_id {params.evidence_id!r} not found in state.evidence_items"
            )

        existing_ids = {ev.evidence_id for ev in state.evidence_items}

        async with deps.session_factory() as session:
            neighbor_nodes: list[DocumentNode] = []

            if params.neighborhood == "section":
                # Fetch siblings: same parent_id, same document, scoped to chat
                seed_node_stmt = select(DocumentNode).where(
                    DocumentNode.id == uuid.UUID(seed.source_node_id),
                    DocumentNode.chat_id == state.chat_id,
                )
                seed_node_result = await session.execute(seed_node_stmt)
                seed_node = seed_node_result.scalar_one_or_none()

                if seed_node is not None and seed_node.parent_id is not None:
                    sibling_stmt = (
                        select(DocumentNode)
                        .where(
                            DocumentNode.chat_id == state.chat_id,
                            DocumentNode.document_id == seed.document_id,
                            DocumentNode.parent_id == seed_node.parent_id,
                            DocumentNode.id != seed_node.id,
                        )
                        .order_by(DocumentNode.order_index)
                        .limit(params.max_extra)
                    )
                    sib_result = await session.execute(sibling_stmt)
                    neighbor_nodes = list(sib_result.scalars().all())

            elif params.neighborhood == "page":
                # Fetch nodes on the same page range
                page_stmt = (
                    select(DocumentNode)
                    .where(
                        DocumentNode.chat_id == state.chat_id,
                        DocumentNode.document_id == seed.document_id,
                        DocumentNode.page_start >= seed.page_start,
                        DocumentNode.page_end <= seed.page_end,
                        DocumentNode.id != uuid.UUID(seed.source_node_id),
                    )
                    .order_by(DocumentNode.order_index)
                    .limit(params.max_extra)
                )
                page_result = await session.execute(page_stmt)
                neighbor_nodes = list(page_result.scalars().all())

            else:  # paragraphs — adjacent by order_index
                seed_node_stmt = select(DocumentNode).where(
                    DocumentNode.id == uuid.UUID(seed.source_node_id),
                    DocumentNode.chat_id == state.chat_id,
                )
                seed_node_result = await session.execute(seed_node_stmt)
                seed_node = seed_node_result.scalar_one_or_none()

                if seed_node is not None:
                    para_stmt = (
                        select(DocumentNode)
                        .where(
                            DocumentNode.chat_id == state.chat_id,
                            DocumentNode.document_id == seed.document_id,
                            DocumentNode.order_index.between(
                                seed_node.order_index - params.max_extra,
                                seed_node.order_index + params.max_extra,
                            ),
                            DocumentNode.id != seed_node.id,
                        )
                        .order_by(DocumentNode.order_index)
                        .limit(params.max_extra * 2)
                    )
                    para_result = await session.execute(para_stmt)
                    neighbor_nodes = list(para_result.scalars().all())

        # Convert nodes to EvidenceItems, skip already present
        for node in neighbor_nodes:
            ev_id = make_evidence_id(
                "expand_evidence",
                str(node.id),
                str(node.document_id),
            )
            if ev_id in existing_ids:
                continue
            evidence.append(
                EvidenceItem(
                    evidence_id=ev_id,
                    source_type=node.node_type,
                    document_id=node.document_id,
                    source_node_id=str(node.id),
                    page_start=node.page_start,
                    page_end=node.page_end,
                    content=node.content or "",
                    score=None,
                    section_title=node.title,
                    heading_path=None,
                    origin_tool="expand_evidence",
                )
            )

        token_est = sum(budget_mgr.count_tokens(ev.content) for ev in evidence)
        status = (
            "empty" if not evidence else ("overflow" if token_est > params.max_tokens else "ok")
        )

        record = ToolCallRecord(
            call_id=call_id,
            tool_name="expand_evidence",
            params=params.model_dump(),
            status=status,  # type: ignore[arg-type]
            token_estimate=token_est,
            source_count=len(evidence),
        )

    except Exception as exc:
        errors.append(
            AgentError(
                code="expand_evidence_error",
                detail=str(exc),
                tool_name="expand_evidence",
            )
        )
        record = ToolCallRecord(
            call_id=call_id,
            tool_name="expand_evidence",
            params=params.model_dump(),
            status="error",
            token_estimate=0,
            source_count=0,
            error=str(exc),
        )

    return ToolInvocation(
        record=record,
        evidence=evidence,
        facts=[],
        errors=errors,
    )


__all__ = ["expand_evidence"]
