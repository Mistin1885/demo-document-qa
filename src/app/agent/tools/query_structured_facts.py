"""Tool: query_structured_facts — read StructuredFact rows via facts service.

Contract (CLAUDE.md §8):
- Calls query_facts(session, current_chat_id=state.chat_id, filt=...).
  The facts service IGNORES filt.chat_id and always uses current_chat_id.
- QueryStructuredFactsParams has NO chat_id field (extra="forbid").
- The tool INTERNALLY builds FactsFilter with chat_id=state.chat_id.
  The service then overwrites it anyway — double-enforced isolation.
- Returns StructuredFactSnapshot list, no EvidenceItems.

No FastAPI; no dict[str, Any]; fully async.
"""

from __future__ import annotations

import uuid

from app.agent.budget import ContextBudgetManager
from app.agent.state import (
    AgentError,
    AgentState,
    StructuredFactSnapshot,
    ToolCallRecord,
)
from app.agent.tools._invocation import ToolDeps, ToolInvocation
from app.agent.tools._models import QueryStructuredFactsParams
from app.services.facts_service import FactsFilter, query_facts


async def query_structured_facts(
    state: AgentState,
    params: QueryStructuredFactsParams,
    *,
    deps: ToolDeps,
) -> ToolInvocation:
    """Query StructuredFact rows from PostgreSQL, always scoped to state.chat_id.

    Isolation:
    - FactsFilter.chat_id is set to state.chat_id (the only legitimate source).
    - query_facts() silently overwrites filt.chat_id with current_chat_id anyway.
    - The LLM cannot inject chat_id via params (extra="forbid").
    """
    call_id = str(uuid.uuid4())
    budget_mgr = ContextBudgetManager()
    errors: list[AgentError] = []
    facts: list[StructuredFactSnapshot] = []

    try:
        # Build FactsFilter using state.chat_id — not from params (no chat_id field)
        # The service will also enforce current_chat_id, so we have dual enforcement.
        filt = FactsFilter(
            chat_id=state.chat_id,  # injected from state
            document_ids=params.document_ids,
            kinds=params.kinds,  # type: ignore[arg-type]
            keys=params.keys,
            page_range=params.page_range,
            numeric_min=params.numeric_min,
            numeric_max=params.numeric_max,
            unit_in=params.unit_in,
            limit=params.limit,
        )

        async with deps.session_factory() as session:
            rows = await query_facts(
                session,
                current_chat_id=state.chat_id,  # service-layer isolation injection
                filt=filt,
            )

        for row in rows:
            val = row.value
            facts.append(
                StructuredFactSnapshot(
                    fact_id=row.id,
                    document_id=row.document_id,
                    kind=row.kind,
                    key=row.key,
                    value_numeric=val.numeric if val else None,
                    value_text=val.raw if val else None,
                    unit=row.unit,
                    page=row.page,
                    context_excerpt=row.context_excerpt,
                )
            )

        # Token estimate from context excerpts
        token_est = sum(
            budget_mgr.count_tokens(f.context_excerpt or f.value_text or "") for f in facts
        )
        status = "empty" if not facts else ("overflow" if token_est > params.max_tokens else "ok")

        record = ToolCallRecord(
            call_id=call_id,
            tool_name="query_structured_facts",
            params=params.model_dump(),
            status=status,  # type: ignore[arg-type]
            token_estimate=token_est,
            source_count=len(facts),
        )

    except Exception as exc:
        errors.append(
            AgentError(
                code="query_structured_facts_error",
                detail=str(exc),
                tool_name="query_structured_facts",
            )
        )
        record = ToolCallRecord(
            call_id=call_id,
            tool_name="query_structured_facts",
            params=params.model_dump(),
            status="error",
            token_estimate=0,
            source_count=0,
            error=str(exc),
        )

    return ToolInvocation(
        record=record,
        evidence=[],
        facts=facts,
        errors=errors,
    )


__all__ = ["query_structured_facts"]
