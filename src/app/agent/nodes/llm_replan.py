"""Node: llm_replan — bounded LLM-driven retrieval replanning.

The LLM can only emit a JSON object validated by ``ReplanDecision``.  It never
executes tools directly; validated nominations are converted into
``AgentState.replan_tool_calls`` for ``execute_retrieval_tools`` to dispatch
through the normal policy and isolation path.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agent.budget import MAX_REPLAN_ROUNDS
from app.agent.policies import PolicyEngine, PolicyViolation
from app.agent.state import AgentState, ReplanToolRequest
from app.providers.base import ChatMessage, ChatProvider

_ALLOWED_SOURCE_TYPES = Literal[
    "chunk",
    "raw_block",
    "document_overview",
    "chapter_summary",
    "compact_chapter_summary",
    "section_summary",
    "compact_section_summary",
    "performance_fact",
    "table_record",
    "figure_caption",
]
_ALLOWED_FACT_KIND = Literal[
    "metric",
    "benchmark",
    "dataset",
    "hyperparameter",
    "ablation",
    "other",
]


class ReplanFactFilter(BaseModel):
    """Restricted fact filter accepted from the replan LLM."""

    model_config = ConfigDict(extra="forbid")

    kinds: list[_ALLOWED_FACT_KIND] = Field(default_factory=list, max_length=5)
    keys: list[str] = Field(default_factory=list, max_length=10)


class ReplanToolCall(BaseModel):
    """One schema-validated retrieval tool nomination from the LLM."""

    model_config = ConfigDict(extra="forbid")

    tool: Literal[
        "search_hybrid",
        "fetch_structural_nodes",
        "query_structured_facts",
        "inspect_document",
    ]
    query: str | None = Field(default=None, min_length=1, max_length=200)
    source_types: list[_ALLOWED_SOURCE_TYPES] = Field(default_factory=list, max_length=8)
    fact_filter: ReplanFactFilter | None = None
    document_id: uuid.UUID | None = None


class ReplanDecision(BaseModel):
    """Top-level JSON decision emitted by ``llm_replan``."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["search_more", "answer_now", "no_info"]
    tool_calls: list[ReplanToolCall] = Field(default_factory=list, max_length=4)
    reasoning: str = Field(default="", max_length=400)


_SYSTEM_PROMPT = """\
You are a bounded retrieval planner for a document QA StateGraph.
Return JSON only, with keys: action, tool_calls, reasoning.
Allowed action: search_more, answer_now, no_info.
Allowed tools: search_hybrid, fetch_structural_nodes, query_structured_facts, inspect_document.
Do not include chat_id, session_id, SQL, URLs, or tools outside the allowed set.
Prefer 1-3 targeted search_hybrid queries when evidence is missing.
"""


def _evidence_summary(state: AgentState, limit: int = 1000) -> str:
    parts: list[str] = []
    for idx, ev in enumerate(state.evidence_items[:8], start=1):
        snippet = re.sub(r"\s+", " ", ev.content).strip()[:220]
        parts.append(f"[{idx}] {ev.source_type} p{ev.page_start}-{ev.page_end}: {snippet}")
    text = "\n".join(parts)
    return text[:limit]


def _build_prompt(state: AgentState) -> str:
    titles = [dm.title or str(dm.document_id) for dm in state.document_manifests[:8]]
    unsatisfied = [r.description for r in state.coverage_requirements if not r.satisfied]
    return (
        f"Question: {state.question}\n"
        f"Documents: {', '.join(titles) if titles else '(unknown)'}\n"
        f"Replan round: {state.replan_rounds + 1}/{MAX_REPLAN_ROUNDS}\n"
        f"Remaining budget estimate: {state.remaining_budget}\n"
        f"Unsatisfied requirements: {unsatisfied[:6]}\n"
        f"Existing evidence summary:\n{_evidence_summary(state)}\n"
        "Return the next retrieval step as JSON only."
    )


def _extract_json(text: str) -> str:
    """Extract a JSON object from raw provider content."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        return stripped[start : end + 1]
    return stripped


def _fallback(state: AgentState, detail: str) -> dict[str, Any]:
    state.record_error(code="LLM_REPLAN_FALLBACK", detail=detail, tool_name="llm_replan")
    state.record_event("node_exit", "llm_replan", action="answer_now", fallback=True)
    return {
        "replan_rounds": min(state.replan_rounds + 1, MAX_REPLAN_ROUNDS),
        "replan_tool_calls": [],
        "errors": list(state.errors),
        "debug_trace": state.debug_trace,
    }


def _to_requests(decision: ReplanDecision) -> list[ReplanToolRequest]:
    requests: list[ReplanToolRequest] = []
    if decision.action != "search_more":
        return requests
    for call in decision.tool_calls:
        fact_kinds: list[str] = []
        fact_keys: list[str] = []
        if call.fact_filter is not None:
            fact_kinds = list(call.fact_filter.kinds)
            fact_keys = list(call.fact_filter.keys)
        requests.append(
            ReplanToolRequest(
                tool=call.tool,
                query=call.query,
                source_types=list(call.source_types),
                fact_kinds=fact_kinds,
                fact_keys=fact_keys,
                document_id=call.document_id,
            )
        )
    return requests


async def llm_replan(state: AgentState, chat_provider: ChatProvider) -> dict[str, Any]:
    """Ask the chat provider for a bounded retrieval replan decision."""
    state.record_event("node_enter", "llm_replan", round=state.replan_rounds + 1)

    if state.replan_rounds >= MAX_REPLAN_ROUNDS:
        state.record_event("node_exit", "llm_replan", skipped="round_cap")
        return {"replan_tool_calls": [], "debug_trace": state.debug_trace}

    messages = [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user", content=_build_prompt(state)),
    ]

    try:
        completion = await chat_provider.complete(messages, temperature=0.0, max_tokens=300)
        raw = json.loads(_extract_json(completion.content))
        decision = ReplanDecision.model_validate(raw)
        PolicyEngine().enforce_replan_decision(state, decision)
    except (json.JSONDecodeError, ValidationError, PolicyViolation, Exception) as exc:
        return _fallback(state, f"invalid llm_replan decision: {exc}")

    requests = _to_requests(decision)
    state.record_event(
        "node_exit",
        "llm_replan",
        action=decision.action,
        tool_calls=len(requests),
        reasoning=decision.reasoning,
    )
    return {
        "replan_rounds": state.replan_rounds + 1,
        "replan_tool_calls": requests,
        "debug_trace": state.debug_trace,
    }


__all__ = ["ReplanDecision", "ReplanToolCall", "llm_replan"]
