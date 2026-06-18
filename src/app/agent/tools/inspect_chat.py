"""Tool: inspect_chat — read chat manifest for routing decisions.

Contract (CLAUDE.md §8):
- chat_id is taken ONLY from state.chat_id; the LLM cannot supply it.
- Returns 0 EvidenceItems; 1 ToolCallRecord.
- Populates ToolInvocation.chat_manifest (a ChatManifestSnapshot).
- The caller (Phase 7.3 node) is responsible for setting state.chat_manifest.

No FastAPI; no dict[str, Any]; fully async.
"""

from __future__ import annotations

import uuid

from app.agent.budget import ContextBudgetManager
from app.agent.state import (
    AgentError,
    AgentState,
    ChatManifestSnapshot,
    ToolCallRecord,
)
from app.agent.tools._invocation import ToolDeps, ToolInvocation
from app.agent.tools._models import InspectChatParams
from app.services.manifest_service import get_chat_manifest


async def inspect_chat(
    state: AgentState,
    params: InspectChatParams,
    *,
    deps: ToolDeps,
) -> ToolInvocation:
    """Read the chat manifest and return a ChatManifestSnapshot.

    Isolation: ``state.chat_id`` is the only chat identifier used.
    ``InspectChatParams`` has no ``chat_id`` field (``extra="forbid"``).

    Returns a ToolInvocation with ``chat_manifest`` populated; no evidence items.
    """
    call_id = str(uuid.uuid4())
    budget_mgr = ContextBudgetManager()
    errors: list[AgentError] = []
    snapshot: ChatManifestSnapshot | None = None

    try:
        # session_factory is a zero-argument async context-manager factory
        async with deps.session_factory() as session:
            manifest = await get_chat_manifest(
                session,
                current_chat_id=state.chat_id,
            )

        # Build cross-document topics from document entries
        topics: list[str] = []
        seen_topics: set[str] = set()
        all_source_types: set[str] = set()

        for entry in manifest.documents:
            for t in entry.main_topics:
                if t not in seen_topics:
                    seen_topics.add(t)
                    topics.append(t)
            for st in entry.available_source_types:
                all_source_types.add(st)

        if not params.include_topics:
            topics = []

        snapshot = ChatManifestSnapshot(
            document_count=manifest.document_count,
            document_titles=[e.title or "" for e in manifest.documents if e.title],
            main_topics=topics,
            total_token_estimate=manifest.total_token_estimate,
            source_types=sorted(all_source_types),
        )
        token_est = budget_mgr.count_tokens(
            " ".join(snapshot.document_titles + snapshot.main_topics)
        )

        record = ToolCallRecord(
            call_id=call_id,
            tool_name="inspect_chat",
            params=params.model_dump(),
            status="ok",
            token_estimate=token_est,
            source_count=manifest.document_count,
        )
    except Exception as exc:
        errors.append(
            AgentError(code="inspect_chat_error", detail=str(exc), tool_name="inspect_chat")
        )
        record = ToolCallRecord(
            call_id=call_id,
            tool_name="inspect_chat",
            params=params.model_dump(),
            status="error",
            token_estimate=0,
            source_count=0,
            error=str(exc),
        )

    return ToolInvocation(
        record=record,
        evidence=[],
        facts=[],
        errors=errors,
        chat_manifest=snapshot,
    )


__all__ = ["inspect_chat"]
