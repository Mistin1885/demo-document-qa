"""Tool: aggregate_sources — compact state.evidence_items into per-group summaries.

Contract (CLAUDE.md §8):
- Does NOT call an LLM (Phase 7.2; LLM aggregation is Phase 9).
- Groups evidence by (document_id, section_title) or (document_id,).
- Joins content with "\n---\n" then truncates to max_summary_tokens.
- Produces new EvidenceItems with source_type="aggregated_evidence".
- evidence_id is deterministic: uuid5(NS_OID, f"aggregate:{doc_id}:{section_title or ''}").
- AggregateSourcesParams has NO chat_id field (extra="forbid").

No FastAPI; no dict[str, Any]; fully async (no-op awaits but signature is async for uniformity).
"""

from __future__ import annotations

import uuid
from uuid import NAMESPACE_OID, uuid5

from app.agent.budget import ContextBudgetManager
from app.agent.state import (
    AgentError,
    AgentState,
    EvidenceItem,
    ToolCallRecord,
)
from app.agent.tools._invocation import ToolDeps, ToolInvocation
from app.agent.tools._models import AggregateSourcesParams

_SEPARATOR = "\n---\n"


def _truncate_to_tokens(
    text: str,
    max_tokens: int,
    budget_mgr: ContextBudgetManager,
) -> str:
    """Truncate *text* so that token count <= max_tokens (approximate)."""
    if budget_mgr.count_tokens(text) <= max_tokens:
        return text
    # Binary-search style: halve until within budget
    # Simple approach: split by separator and keep chunks that fit
    parts = text.split(_SEPARATOR)
    result_parts: list[str] = []
    used = 0
    for part in parts:
        part_tokens = budget_mgr.count_tokens(part)
        if used + part_tokens <= max_tokens:
            result_parts.append(part)
            used += part_tokens
        else:
            break
    if not result_parts:
        # Fallback: hard-truncate the first part by characters (4 chars ≈ 1 token)
        first = parts[0]
        char_limit = max_tokens * 4
        result_parts = [first[:char_limit]]
    return _SEPARATOR.join(result_parts)


async def aggregate_sources(
    state: AgentState,
    params: AggregateSourcesParams,
    *,
    deps: ToolDeps,  # noqa: ARG001 — not used in Phase 7.2 (no LLM call)
) -> ToolInvocation:
    """Compact state.evidence_items into per-group summaries.

    NOTE: This is a deterministic placeholder (Phase 7.2).
    Phase 9 can swap the body of this tool to call deps.chat_provider
    for LLM-based summarisation; the signature and isolation contract
    remain unchanged.

    Isolation: no chat_id in params; operates only on state.evidence_items
    which were already fetched with state.chat_id scope.
    """
    call_id = str(uuid.uuid4())
    budget_mgr = ContextBudgetManager()
    errors: list[AgentError] = []
    evidence: list[EvidenceItem] = []

    try:
        # Group evidence
        buckets: dict[tuple[str, str | None], list[EvidenceItem]] = {}
        for item in state.evidence_items:
            if params.strategy == "per_section":
                key = (str(item.document_id), item.section_title)
            else:  # per_document
                key = (str(item.document_id), None)
            buckets.setdefault(key, []).append(item)

        for (doc_id_str, section_title), items in buckets.items():
            # Deterministic evidence_id (uuid5)
            ev_id = str(uuid5(NAMESPACE_OID, f"aggregate:{doc_id_str}:{section_title or ''}"))

            # Join content with separator, then truncate
            joined = _SEPARATOR.join(item.content for item in items if item.content)
            truncated = _truncate_to_tokens(joined, params.max_summary_tokens, budget_mgr)

            # Use min page_start and max page_end across group
            page_start = min(item.page_start for item in items)
            page_end = max(item.page_end for item in items)

            # Use highest score in group
            scores = [item.score for item in items if item.score is not None]
            agg_score = max(scores) if scores else None

            evidence.append(
                EvidenceItem(
                    evidence_id=ev_id,
                    # source_type is "aggregated_evidence" per contract
                    source_type="aggregated_evidence",
                    document_id=uuid.UUID(doc_id_str),
                    source_node_id=f"aggregate:{doc_id_str}:{section_title or ''}",
                    page_start=page_start,
                    page_end=page_end,
                    content=truncated,
                    score=agg_score,
                    section_title=section_title,
                    heading_path=None,
                    origin_tool="aggregate_sources",
                )
            )

        token_est = sum(budget_mgr.count_tokens(ev.content) for ev in evidence)
        status = (
            "empty" if not evidence else ("overflow" if token_est > params.max_tokens else "ok")
        )

        record = ToolCallRecord(
            call_id=call_id,
            tool_name="aggregate_sources",
            params=params.model_dump(),
            status=status,  # type: ignore[arg-type]
            token_estimate=token_est,
            source_count=len(evidence),
        )

    except Exception as exc:
        errors.append(
            AgentError(
                code="aggregate_sources_error",
                detail=str(exc),
                tool_name="aggregate_sources",
            )
        )
        record = ToolCallRecord(
            call_id=call_id,
            tool_name="aggregate_sources",
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


__all__ = ["aggregate_sources"]
