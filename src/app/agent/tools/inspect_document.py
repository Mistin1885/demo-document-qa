"""Tool: inspect_document — read one document's metadata + section tree + abstract.

Contract (CLAUDE.md §8):
- chat_id is taken ONLY from state.chat_id; the LLM cannot supply it.
- Returns a DocumentManifest and 0..n structural EvidenceItems
  (document_overview / chapter_summary Summary rows).
- ``InspectDocumentParams`` has no chat_id field (extra="forbid").

No FastAPI; no dict[str, Any]; fully async.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.agent.budget import ContextBudgetManager
from app.agent.state import (
    AgentError,
    AgentState,
    DocumentManifest,
    EvidenceItem,
    ToolCallRecord,
    make_evidence_id,
)
from app.agent.tools._invocation import ToolDeps, ToolInvocation
from app.agent.tools._models import InspectDocumentParams
from app.models.orm import DocumentNode, Summary
from app.services.document_service import get_document


async def inspect_document(
    state: AgentState,
    params: InspectDocumentParams,
    *,
    deps: ToolDeps,
) -> ToolInvocation:
    """Return document metadata + structural evidence for one document.

    Isolation: all queries are scoped to state.chat_id.
    InspectDocumentParams.document_id must belong to state.chat_id.
    """
    call_id = str(uuid.uuid4())
    budget_mgr = ContextBudgetManager()
    errors: list[AgentError] = []
    evidence: list[EvidenceItem] = []
    doc_manifest: DocumentManifest | None = None

    try:
        async with deps.session_factory() as session:
            # Verify the document is in this chat's scope
            doc_read = await get_document(
                session,
                chat_id=state.chat_id,
                document_id=params.document_id,
            )

            # Fetch structural summaries (document_overview + chapter_summary)
            summary_stmt = (
                select(Summary)
                .where(
                    Summary.chat_id == state.chat_id,
                    Summary.document_id == params.document_id,
                    Summary.kind.in_(["document_overview", "chapter_summary"]),
                )
                .order_by(Summary.created_at)
            )
            result = await session.execute(summary_stmt)
            summaries = list(result.scalars().all())

            # Fetch section count
            section_stmt = select(DocumentNode).where(
                DocumentNode.chat_id == state.chat_id,
                DocumentNode.document_id == params.document_id,
                DocumentNode.node_type.in_(["section", "subsection"]),
            )
            section_result = await session.execute(section_stmt)
            sections = list(section_result.scalars().all())

            # Build DocumentManifest
            abstract_summary: str | None = None
            for s in summaries:
                if s.kind == "chapter_summary" and abstract_summary is None:
                    abstract_summary = s.content

            source_types: list[str] = list({s.kind for s in summaries})

            doc_manifest = DocumentManifest(
                document_id=params.document_id,
                title=doc_read.original_filename,
                abstract_summary=abstract_summary,
                section_count=len(sections),
                page_count=doc_read.page_count,
                source_types=source_types,
            )

            # Build structural evidence items from summaries
            for s in summaries:
                node_id = str(s.source_node_id) if s.source_node_id else str(s.id)
                ev_id = make_evidence_id(
                    "inspect_document",
                    node_id,
                    str(params.document_id),
                )
                evidence.append(
                    EvidenceItem(
                        evidence_id=ev_id,
                        source_type=s.kind,
                        document_id=params.document_id,
                        source_node_id=node_id,
                        page_start=1,
                        page_end=doc_read.page_count or 1,
                        content=s.content,
                        score=None,
                        section_title=None,
                        heading_path=None,
                        origin_tool="inspect_document",
                    )
                )

            # Optionally include section tree as evidence
            if params.include_section_tree:
                for node in sections:
                    node_id = str(node.id)
                    ev_id = make_evidence_id(
                        "inspect_document",
                        node_id,
                        str(params.document_id),
                    )
                    evidence.append(
                        EvidenceItem(
                            evidence_id=ev_id,
                            source_type="section_node",
                            document_id=params.document_id,
                            source_node_id=node_id,
                            page_start=node.page_start,
                            page_end=node.page_end,
                            content=node.content or "",
                            score=None,
                            section_title=node.title,
                            heading_path=None,
                            origin_tool="inspect_document",
                        )
                    )

        # Calculate token estimate
        token_est = sum(budget_mgr.count_tokens(ev.content) for ev in evidence)

        # Check overflow
        status = "ok"
        if token_est > params.max_tokens:
            status = "overflow"

        record = ToolCallRecord(
            call_id=call_id,
            tool_name="inspect_document",
            params=params.model_dump(),
            status=status,  # type: ignore[arg-type]
            token_estimate=token_est,
            source_count=len(evidence),
        )

    except Exception as exc:
        errors.append(
            AgentError(
                code="inspect_document_error",
                detail=str(exc),
                tool_name="inspect_document",
            )
        )
        record = ToolCallRecord(
            call_id=call_id,
            tool_name="inspect_document",
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
        document_manifest=doc_manifest,
    )


__all__ = ["inspect_document"]
