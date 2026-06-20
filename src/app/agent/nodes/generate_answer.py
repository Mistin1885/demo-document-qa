"""Node: generate_answer — call ChatProvider to produce a cited markdown answer.

Phase 7.4 additions:
  - PolicyEngine.enforce_pre_answer(state) runs policies 6, 7, 8 before
    calling the LLM.  PolicyViolation from policy 7 (incomplete coverage at
    cap) short-circuits to the no-information fallback answer.
  - PolicyEngine.enforce_provider_result(state, exc) wraps provider
    exceptions as policy 14 violations (explicit error, no silent fallback).

System prompt + chat manifest summary + evidence excerpts + question → answer.

The LLM receives:
  - A system prompt (no chat_id).
  - A formatted context block with evidence excerpts (no chat_id).
  - The user question.

Inline citation markers use the form [c<idx>] where idx is 1-based index
into evidence_items order.

If evidence_items is empty, returns the "no information" fallback answer
(CLAUDE.md §0 rule 5 / policy 11).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agent.policies import PolicyEngine, PolicyViolation
from app.agent.state import AgentState
from app.providers.base import ChatMessage

if TYPE_CHECKING:
    from app.providers.base import ChatProvider

_engine = PolicyEngine()

_NO_INFO_ANSWER = (
    "There is not enough information in the current chat's documents to answer this question."
)

_SYSTEM_PROMPT = """\
You are a document question-answering assistant.
Answer the question using ONLY the provided evidence excerpts.
Cite each piece of evidence with an inline marker like [c1], [c2], etc.
If the evidence does not contain sufficient information, reply with:
"There is not enough information in the current chat's documents to answer this question."
Do NOT use your own knowledge; rely exclusively on the evidence provided.
"""


def _build_context_block(state: AgentState) -> str:
    """Build a formatted context block from evidence items."""
    if not state.evidence_items:
        return ""

    manifest_summary = ""
    if state.chat_manifest:
        titles = ", ".join(state.chat_manifest.document_titles[:5])
        manifest_summary = f"Documents in this chat: {titles}\n\n"

    lines: list[str] = [manifest_summary + "Evidence excerpts:"]
    for idx, ev in enumerate(state.evidence_items, start=1):
        doc_title = ev.section_title or f"doc:{ev.document_id}"
        lines.append(f"[c{idx}] (page {ev.page_start}–{ev.page_end}, {doc_title})\n{ev.content}")
    return "\n\n".join(lines)


def _build_session_memory_block(state: AgentState) -> str:
    """Build recent same-session memory for deep QA follow-up mode."""
    if not state.generation_config.deep_qa_mode or not state.conversation_history:
        return ""
    lines = ["Recent same-session conversation (for resolving follow-up references):"]
    for turn in state.conversation_history[-8:]:
        lines.append(f"{turn.role}: {turn.content}")
    return "\n".join(lines)


async def generate_answer(state: AgentState, chat_provider: ChatProvider) -> dict[str, Any]:
    """Call ChatProvider to produce the answer; apply pre-answer policies."""
    state.record_event("node_enter", "generate_answer")

    # --- Policy 6 + 7: pre-answer checks ---
    try:
        _engine.enforce_pre_answer(state)
    except PolicyViolation as pv:
        # Policy 7 fires when coverage is incomplete at iteration cap.
        # Return the no-information fallback answer (never the model's knowledge).
        state.record_event(
            "node_exit",
            "generate_answer",
            answer_source="policy_violation_fallback",
            policy_id=pv.policy_id,
            code=pv.code,
        )
        return {
            "answer": _NO_INFO_ANSWER,
            "errors": list(state.errors),
            "debug_trace": state.debug_trace,
        }

    # --- No evidence: policy 11 short-circuit ---
    if not state.evidence_items:
        state.record_event("node_exit", "generate_answer", answer_source="no_evidence_fallback")
        return {
            "answer": _NO_INFO_ANSWER,
            "debug_trace": state.debug_trace,
        }

    memory_block = _build_session_memory_block(state)
    context_block = _build_context_block(state)
    user_content = f"{context_block}\n\nQuestion: {state.question}"
    if memory_block:
        user_content = f"{memory_block}\n\n{user_content}"
    messages = [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    ]

    # Per-request overrides (frontend may bump max_tokens for long summaries).
    gen_cfg = state.generation_config
    complete_kwargs: dict[str, Any] = {
        "temperature": 0.0 if gen_cfg.temperature is None else gen_cfg.temperature,
    }
    if gen_cfg.max_answer_tokens is not None:
        complete_kwargs["max_tokens"] = gen_cfg.max_answer_tokens

    # --- Policy 14: provider failure → explicit error ---
    try:
        completion = await chat_provider.complete(messages, **complete_kwargs)
    except Exception as exc:
        try:
            _engine.enforce_provider_result(state, exc)
        except PolicyViolation:
            # Policy 14 recorded the error; return a clear error answer instead
            # of letting the exception propagate (which would crash the graph).
            return {
                "answer": _NO_INFO_ANSWER,
                "errors": list(state.errors),
                "debug_trace": state.debug_trace,
            }

    answer = completion.content

    state.record_event(
        "node_exit",
        "generate_answer",
        model=completion.model,
        prompt_tokens=completion.usage.prompt_tokens,
        completion_tokens=completion.usage.completion_tokens,
    )

    return {
        "answer": answer,
        "debug_trace": state.debug_trace,
    }


__all__ = ["generate_answer"]
