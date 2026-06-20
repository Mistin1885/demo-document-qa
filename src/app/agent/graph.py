"""graph.py — LangGraph StateGraph assembly for the Paper Notebook Agent.

Architecture decision — state schema approach:
---------------------------------------------------------------------------
LangGraph's default reducer for TypedDict merges returned dict keys into the
full state (field-by-field update, keeping unchanged keys).  We wrap the
serialised AgentState in a single ``state`` key inside a ``StateContainer``
TypedDict.  Each graph node:
  1. Reads ``container["state"]`` → deserialises into AgentState (type-checked).
  2. Runs its logic.
  3. Merges updates back into a full AgentState dict.
  4. Returns ``{"state": updated_dict}``.

This keeps AgentState as the **single source of truth** while working cleanly
with LangGraph's dict-merge reducer (no Annotated list reducers required).

Usage:
    from app.agent.graph import build_graph
    from app.agent.state import AgentState

    graph = build_graph(deps=deps, chat_provider=provider)
    container = await graph.ainvoke({"state": state.model_dump()})
    final_state = AgentState.model_validate(container["state"])
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.agent.budget import MAX_REPLAN_ROUNDS, ContextBudgetManager
from app.agent.nodes.aggregate_sources_node import aggregate_sources_node
from app.agent.nodes.check_context_budget import check_context_budget, is_overflow
from app.agent.nodes.check_coverage import check_coverage
from app.agent.nodes.enforce_scope_and_policies import enforce_scope_and_policies
from app.agent.nodes.execute_retrieval_tools import execute_retrieval_tools
from app.agent.nodes.generate_answer import generate_answer
from app.agent.nodes.inspect_scope import inspect_scope
from app.agent.nodes.llm_replan import llm_replan
from app.agent.nodes.load_chat_and_session import load_chat_and_session
from app.agent.nodes.merge_evidence_workspace import merge_evidence_workspace
from app.agent.nodes.persist_messages import InMemoryMessageStore, MessageStore, persist_messages
from app.agent.nodes.plan_gap_retrieval import plan_gap_retrieval
from app.agent.nodes.plan_information_needs import plan_information_needs
from app.agent.nodes.validate_citations import validate_citations
from app.agent.nodes.validate_scope_isolation import validate_scope_isolation
from app.agent.nodes.verify_critical_claims import verify_critical_claims
from app.agent.state import AgentState
from app.agent.tools._invocation import ToolDeps
from app.providers.base import ChatProvider

# ---------------------------------------------------------------------------
# LangGraph state container
# ---------------------------------------------------------------------------


class StateContainer(TypedDict):
    """Single-key TypedDict wrapper so LangGraph can merge updates cleanly.

    Each node receives the full container and returns ``{"state": <new_dict>}``.
    LangGraph merges this key back into the container, preserving all other keys.
    """

    state: dict[str, Any]


def _load(container: StateContainer) -> AgentState:
    """Deserialise the AgentState from the container (validates on each call)."""
    return AgentState.model_validate(container["state"])


def _wrap(state: AgentState, updates: dict[str, Any]) -> dict[str, Any]:
    """Apply *updates* to *state* and return the container update dict."""
    current = state.model_dump()
    current.update(updates)
    return {"state": current}


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------


def build_graph(
    deps: ToolDeps,
    *,
    chat_provider: ChatProvider,
    budget_manager: ContextBudgetManager | None = None,
    message_store: MessageStore | None = None,
) -> Any:  # returns CompiledStateGraph[StateContainer]
    """Assemble and compile the Paper Notebook Agent StateGraph.

    Parameters
    ----------
    deps:
        ToolDeps carrying retrieval_service, chat_provider (for tools),
        and session_factory.
    chat_provider:
        ChatProvider used by generate_answer node.
    budget_manager:
        Optional ContextBudgetManager; defaults to ContextBudgetManager().
    message_store:
        Optional MessageStore; defaults to InMemoryMessageStore().

    Returns
    -------
    CompiledStateGraph
        Invoke with ``await graph.ainvoke({"state": state.model_dump()})``.
        Returns ``{"state": <final_state_dict>}``.
    """
    bm = budget_manager or ContextBudgetManager()
    ms: MessageStore = message_store or InMemoryMessageStore()

    # -----------------------------------------------------------------------
    # Node wrappers — each deserialises state, calls the node fn, wraps result
    # -----------------------------------------------------------------------

    async def _node_load(container: StateContainer) -> dict[str, Any]:
        state = _load(container)
        updates = await load_chat_and_session(state, deps)
        return _wrap(state, updates)

    async def _node_inspect_scope(container: StateContainer) -> dict[str, Any]:
        state = _load(container)
        updates = await inspect_scope(state, deps)
        return _wrap(state, updates)

    async def _node_plan(container: StateContainer) -> dict[str, Any]:
        state = _load(container)
        updates = await plan_information_needs(state)
        return _wrap(state, updates)

    async def _node_enforce(container: StateContainer) -> dict[str, Any]:
        state = _load(container)
        updates = await enforce_scope_and_policies(state)
        return _wrap(state, updates)

    async def _node_execute(container: StateContainer) -> dict[str, Any]:
        state = _load(container)
        updates = await execute_retrieval_tools(state, deps)
        return _wrap(state, updates)

    async def _node_merge(container: StateContainer) -> dict[str, Any]:
        state = _load(container)
        updates = await merge_evidence_workspace(state)
        return _wrap(state, updates)

    async def _node_budget(container: StateContainer) -> dict[str, Any]:
        state = _load(container)
        updates = await check_context_budget(state, bm)
        return _wrap(state, updates)

    async def _node_aggregate(container: StateContainer) -> dict[str, Any]:
        state = _load(container)
        updates = await aggregate_sources_node(state, deps)
        return _wrap(state, updates)

    async def _node_coverage(container: StateContainer) -> dict[str, Any]:
        state = _load(container)
        updates = await check_coverage(state)
        return _wrap(state, updates)

    async def _node_gap(container: StateContainer) -> dict[str, Any]:
        state = _load(container)
        updates = await plan_gap_retrieval(state, deps)
        return _wrap(state, updates)

    async def _node_llm_replan(container: StateContainer) -> dict[str, Any]:
        state = _load(container)
        updates = await llm_replan(state, chat_provider)
        return _wrap(state, updates)

    async def _node_verify(container: StateContainer) -> dict[str, Any]:
        state = _load(container)
        updates = await verify_critical_claims(state)
        return _wrap(state, updates)

    async def _node_generate(container: StateContainer) -> dict[str, Any]:
        state = _load(container)
        updates = await generate_answer(state, chat_provider)
        return _wrap(state, updates)

    async def _node_validate_cit(container: StateContainer) -> dict[str, Any]:
        state = _load(container)
        updates = await validate_citations(state)
        return _wrap(state, updates)

    async def _node_validate_iso(container: StateContainer) -> dict[str, Any]:
        state = _load(container)
        updates = await validate_scope_isolation(state)
        return _wrap(state, updates)

    async def _node_persist(container: StateContainer) -> dict[str, Any]:
        state = _load(container)
        updates = await persist_messages(state, ms)
        return _wrap(state, updates)

    # -----------------------------------------------------------------------
    # Conditional edge functions
    # -----------------------------------------------------------------------

    def _route_budget(
        container: StateContainer,
    ) -> Literal["aggregate_sources_node", "check_coverage"]:
        state = _load(container)
        return "aggregate_sources_node" if is_overflow(state, bm) else "check_coverage"

    def _route_coverage(
        container: StateContainer,
    ) -> Literal["plan_gap_retrieval", "llm_replan", "verify_critical_claims"]:
        state = _load(container)
        has_unsatisfied = any(not r.satisfied for r in state.coverage_requirements)
        if state.coverage_state == "incomplete" and state.iteration_count < 2:
            return "plan_gap_retrieval"
        max_replan_rounds = (
            state.generation_config.max_replan_rounds
            if state.generation_config.max_replan_rounds is not None
            else MAX_REPLAN_ROUNDS
        )
        max_replan_rounds = min(max_replan_rounds, MAX_REPLAN_ROUNDS)
        if has_unsatisfied and state.replan_rounds < max_replan_rounds:
            return "llm_replan"
        return "verify_critical_claims"

    # -----------------------------------------------------------------------
    # Build graph
    # -----------------------------------------------------------------------

    # mypy cannot resolve LangGraph's StateGraph overloads with TypedDict state
    # schema + async node callables.  The runtime behaviour is correct; we suppress
    # the false-positive overload errors here.
    graph: Any = StateGraph(StateContainer)  # type: ignore[call-overload]

    graph.add_node("load_chat_and_session", _node_load)
    graph.add_node("inspect_scope", _node_inspect_scope)
    graph.add_node("plan_information_needs", _node_plan)
    graph.add_node("enforce_scope_and_policies", _node_enforce)
    graph.add_node("execute_retrieval_tools", _node_execute)
    graph.add_node("merge_evidence_workspace", _node_merge)
    graph.add_node("check_context_budget", _node_budget)
    graph.add_node("aggregate_sources_node", _node_aggregate)
    graph.add_node("check_coverage", _node_coverage)
    graph.add_node("plan_gap_retrieval", _node_gap)
    graph.add_node("llm_replan", _node_llm_replan)
    graph.add_node("verify_critical_claims", _node_verify)
    graph.add_node("generate_answer", _node_generate)
    graph.add_node("validate_citations", _node_validate_cit)
    graph.add_node("validate_scope_isolation", _node_validate_iso)
    graph.add_node("persist_messages", _node_persist)

    # -----------------------------------------------------------------------
    # Edges (DAG — matches CLAUDE.md §8 workflow)
    # -----------------------------------------------------------------------

    graph.set_entry_point("load_chat_and_session")
    graph.add_edge("load_chat_and_session", "inspect_scope")
    graph.add_edge("inspect_scope", "plan_information_needs")
    graph.add_edge("plan_information_needs", "enforce_scope_and_policies")
    graph.add_edge("enforce_scope_and_policies", "execute_retrieval_tools")
    graph.add_edge("execute_retrieval_tools", "merge_evidence_workspace")
    graph.add_edge("merge_evidence_workspace", "check_context_budget")

    graph.add_conditional_edges(
        "check_context_budget",
        _route_budget,
        {
            "aggregate_sources_node": "aggregate_sources_node",
            "check_coverage": "check_coverage",
        },
    )
    graph.add_edge("aggregate_sources_node", "check_coverage")

    graph.add_conditional_edges(
        "check_coverage",
        _route_coverage,
        {
            "plan_gap_retrieval": "plan_gap_retrieval",
            "llm_replan": "llm_replan",
            "verify_critical_claims": "verify_critical_claims",
        },
    )
    graph.add_edge("plan_gap_retrieval", "execute_retrieval_tools")
    graph.add_edge("llm_replan", "execute_retrieval_tools")

    graph.add_edge("verify_critical_claims", "generate_answer")
    graph.add_edge("generate_answer", "validate_citations")
    graph.add_edge("validate_citations", "validate_scope_isolation")
    graph.add_edge("validate_scope_isolation", "persist_messages")
    graph.add_edge("persist_messages", END)

    return graph.compile()


__all__ = ["build_graph", "StateContainer"]
