"""Node: load_chat_and_session — validate chat/session exist and load basic context.

This is the first node in the graph. It:
1. Records node_enter trace event.
2. Verifies chat_id and session_id are set (enforced by AgentState field validation,
   but we record the intent here for the trace).
3. Attempts to load the chat manifest via inspect_chat tool so later nodes have
   document context for routing decisions.
4. Records node_exit trace event.

Isolation guarantee: chat_id and session_id are never reassigned here;
they are taken directly from state as set by the service layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agent.state import AgentState
from app.agent.tools._models import InspectChatParams

if TYPE_CHECKING:
    from app.agent.tools._invocation import ToolDeps


async def load_chat_and_session(state: AgentState, deps: ToolDeps) -> dict[str, Any]:
    """Load chat manifest and record trace events."""
    state.record_event("node_enter", "load_chat_and_session")

    # Import here to avoid circular imports at module level
    from app.agent.tools.inspect_chat import inspect_chat

    params = InspectChatParams(include_topics=True)
    invocation = await inspect_chat(state, params, deps=deps)

    # Merge results into state fields
    updates: dict[str, Any] = {}

    # Record tool call
    new_tool_calls = list(state.tool_calls) + [invocation.record]
    updates["tool_calls"] = new_tool_calls

    # Apply chat manifest if returned
    if invocation.chat_manifest is not None:
        updates["chat_manifest"] = invocation.chat_manifest

    # Propagate errors
    if invocation.errors:
        updates["errors"] = list(state.errors) + invocation.errors

    # Apply fingerprint
    from app.agent.state import AgentState as _AS  # noqa: PLC0415

    fp = _AS._fingerprint("inspect_chat", params.model_dump())  # type: ignore[attr-defined]
    updates["tool_invocations_fingerprints"] = state.tool_invocations_fingerprints | {fp}

    state.record_event("node_exit", "load_chat_and_session")
    updates["debug_trace"] = state.debug_trace

    return updates


__all__ = ["load_chat_and_session"]
