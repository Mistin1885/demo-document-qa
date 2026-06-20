"""Tool: search_hybrid — Vespa hybrid BM25 + ANN retrieval.

Contract (CLAUDE.md §7, §8):
- Calls RetrievalService.search() which enforces chat_id contains filter.
- chat_id is taken from state.chat_id and injected into RetrievalRequest.
- SearchHybridParams has NO chat_id field (extra="forbid").
- Returns EvidenceItems from SearchHit list; no StructuredFacts.

No FastAPI; no dict[str, Any]; fully async.
"""

from __future__ import annotations

import uuid

from app.agent.budget import ContextBudgetManager
from app.agent.state import (
    AgentError,
    AgentState,
    EvidenceItem,
    ToolCallRecord,
    make_evidence_id,
)
from app.agent.tools._invocation import ToolDeps, ToolInvocation
from app.agent.tools._models import SearchHybridParams
from app.retrieval.models import RetrievalRequest


async def search_hybrid(
    state: AgentState,
    params: SearchHybridParams,
    *,
    deps: ToolDeps,
) -> ToolInvocation:
    """Execute Vespa hybrid retrieval using RetrievalService.

    Isolation:
    - ``state.chat_id`` is injected into RetrievalRequest.chat_id.
    - The LLM cannot pass or override chat_id (not in SearchHybridParams).
    - RetrievalService._yql_where enforces the Vespa-level chat_id filter.
    """
    call_id = str(uuid.uuid4())
    budget_mgr = ContextBudgetManager()
    errors: list[AgentError] = []
    evidence: list[EvidenceItem] = []

    try:
        if params.preset == "broad":
            final_top_k = min(params.top_k * 2, 60)
        else:
            final_top_k = params.top_k

        req = RetrievalRequest(
            chat_id=state.chat_id,  # injected from state — LLM cannot change this
            query=params.query,
            document_ids=params.document_ids,
            source_types=params.source_types,
            final_top_k=final_top_k,
            rerank_mode=params.rerank_mode,
        )

        response = await deps.retrieval_service.search(req)

        for hit in response.hits:
            ev_id = make_evidence_id(
                "search_hybrid",
                hit.source_node_id,
                hit.document_id,
            )
            # Prefer rerank score, then fusion, then 0.0
            score: float | None = (
                hit.native_rerank_score
                or hit.cross_encoder_score
                or hit.fusion_score
                or hit.final_score
            )
            evidence.append(
                EvidenceItem(
                    evidence_id=ev_id,
                    source_type=hit.source_type,
                    document_id=uuid.UUID(hit.document_id),
                    source_node_id=hit.source_node_id,
                    page_start=hit.page_start,
                    page_end=hit.page_end,
                    content=hit.content,
                    score=score,
                    vector_score=hit.vector_score,
                    section_title=hit.title,
                    heading_path=hit.heading_path,
                    origin_tool="search_hybrid",
                )
            )

        token_est = sum(budget_mgr.count_tokens(ev.content) for ev in evidence)
        status = (
            "empty" if not evidence else ("overflow" if token_est > params.max_tokens else "ok")
        )

        record = ToolCallRecord(
            call_id=call_id,
            tool_name="search_hybrid",
            params=params.model_dump(),
            status=status,  # type: ignore[arg-type]
            token_estimate=token_est,
            source_count=len(evidence),
        )

    except Exception as exc:
        errors.append(
            AgentError(
                code="search_hybrid_error",
                detail=str(exc),
                tool_name="search_hybrid",
            )
        )
        record = ToolCallRecord(
            call_id=call_id,
            tool_name="search_hybrid",
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


__all__ = ["search_hybrid"]
