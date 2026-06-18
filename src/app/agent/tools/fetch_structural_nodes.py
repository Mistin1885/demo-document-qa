"""Tool: fetch_structural_nodes — deterministic PostgreSQL fetch (NOT top-k).

Contract (CLAUDE.md §8, §6.5):
- Uses SQLAlchemy ORM select() only; never Vespa.
- All queries scoped to state.chat_id.
- chat_id is NEVER in params (extra="forbid").
- Returns EvidenceItems from DocumentNode + Summary rows.

No FastAPI; no dict[str, Any]; fully async.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

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
from app.agent.tools._models import FetchStructuralNodesParams
from app.models.orm import DocumentNode, Summary

if TYPE_CHECKING:
    pass


async def fetch_structural_nodes(
    state: AgentState,
    params: FetchStructuralNodesParams,
    *,
    deps: ToolDeps,
) -> ToolInvocation:
    """Deterministically fetch DocumentNode + Summary rows from PostgreSQL.

    This is the "structural / deterministic fetch" path (CLAUDE.md §6.5):
    whole-doc / whole-chat / specific-section content — never replaced by top-k.

    Isolation: all queries use ``state.chat_id``.
    """
    call_id = str(uuid.uuid4())
    budget_mgr = ContextBudgetManager()
    errors: list[AgentError] = []
    evidence: list[EvidenceItem] = []

    try:
        async with deps.session_factory() as session:
            # ---- Fetch DocumentNodes scoped to chat ----
            node_stmt = select(DocumentNode).where(DocumentNode.chat_id == state.chat_id)

            if params.document_ids:
                node_stmt = node_stmt.where(DocumentNode.document_id.in_(params.document_ids))
            if params.node_types:
                node_stmt = node_stmt.where(DocumentNode.node_type.in_(params.node_types))

            node_stmt = node_stmt.order_by(
                DocumentNode.document_id,
                DocumentNode.order_index,
            ).limit(params.limit)

            node_result = await session.execute(node_stmt)
            nodes = list(node_result.scalars().all())

            # ---- Fetch Summaries scoped to chat (if source_types specified) ----
            summaries: list[Summary] = []
            if params.source_types:
                sum_stmt = select(Summary).where(
                    Summary.chat_id == state.chat_id,
                    Summary.kind.in_(params.source_types),
                )
                if params.document_ids:
                    sum_stmt = sum_stmt.where(Summary.document_id.in_(params.document_ids))
                sum_stmt = sum_stmt.order_by(
                    Summary.document_id,
                    Summary.created_at,
                ).limit(params.limit)

                sum_result = await session.execute(sum_stmt)
                summaries = list(sum_result.scalars().all())

        # Build evidence from nodes
        for node in nodes:
            ev_id = make_evidence_id(
                "fetch_structural_nodes",
                str(node.id),
                str(node.document_id),
            )
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
                    origin_tool="fetch_structural_nodes",
                )
            )

        # Build evidence from summaries
        for s in summaries:
            node_id = str(s.source_node_id) if s.source_node_id else str(s.id)
            ev_id = make_evidence_id(
                "fetch_structural_nodes",
                node_id,
                str(s.document_id),
            )
            evidence.append(
                EvidenceItem(
                    evidence_id=ev_id,
                    source_type=s.kind,
                    document_id=s.document_id,
                    source_node_id=node_id,
                    page_start=1,
                    page_end=1,
                    content=s.content or "",
                    score=None,
                    section_title=None,
                    heading_path=None,
                    origin_tool="fetch_structural_nodes",
                )
            )

        # Compute token estimate and status
        token_est = sum(budget_mgr.count_tokens(ev.content) for ev in evidence)
        status = (
            "empty" if not evidence else ("overflow" if token_est > params.max_tokens else "ok")
        )

        record = ToolCallRecord(
            call_id=call_id,
            tool_name="fetch_structural_nodes",
            params=params.model_dump(),
            status=status,  # type: ignore[arg-type]
            token_estimate=token_est,
            source_count=len(evidence),
        )

    except Exception as exc:
        errors.append(
            AgentError(
                code="fetch_structural_nodes_error",
                detail=str(exc),
                tool_name="fetch_structural_nodes",
            )
        )
        record = ToolCallRecord(
            call_id=call_id,
            tool_name="fetch_structural_nodes",
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


__all__ = ["fetch_structural_nodes"]
