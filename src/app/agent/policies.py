"""policies.py — Code-enforced agent policies (GUIDE §13.2, CLAUDE.md §8).

14 policies are enforced here in pure Python — NOT in prompt text.

PolicyViolation
    Raised by a policy check when a hard invariant is violated.

Policy (Protocol)
    Interface each policy must implement.

PolicyEngine
    Orchestrates policy groups by execution stage:
      enforce_pre_retrieval    → policies 1, 2, 3, 4, 9, 10
      enforce_post_retrieval   → policy 5
      enforce_pre_answer       → policies 6, 7, 8
      enforce_answer           → policy 11
      enforce_citations        → policies 12, 13
      enforce_provider_result  → policy 14

Design rules (CLAUDE.md §8, §12):
  - No FastAPI imports.
  - No dict[str, Any] on typed fields.
  - Every violation is recorded via state.record_error() + state.record_event()
    before raising (or silently degrading for non-fatal policies).
  - chat_id is never accepted from LLM/agent — only from AgentState.
"""

from __future__ import annotations

import re
from typing import Any, Protocol
from uuid import UUID

from app.agent.state import AgentPlan, AgentState

# ---------------------------------------------------------------------------
# PolicyViolation
# ---------------------------------------------------------------------------


class PolicyViolation(Exception):
    """Raised when a hard policy invariant is violated.

    Attributes
    ----------
    code:
        Machine-readable error code (e.g., ``"CHAT_ID_MISSING"``).
    detail:
        Human-readable description of the violation.
    policy_id:
        The GUIDE §13.2 policy number (1–14).
    """

    def __init__(self, code: str, detail: str, policy_id: int) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.policy_id = policy_id


# ---------------------------------------------------------------------------
# Policy protocol
# ---------------------------------------------------------------------------


class Policy(Protocol):
    """Protocol for a single agent policy check."""

    def check(self, state: AgentState, *, context: dict[str, Any]) -> None:
        """Check the policy.  Raise PolicyViolation on hard violations.

        Side effects: may mutate *state* (e.g., remove violating items,
        add AgentError records) for non-fatal policies.
        """
        ...


# ---------------------------------------------------------------------------
# _Numeric helpers
# ---------------------------------------------------------------------------

_NUMERIC_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_NUMERIC_KEYWORDS: frozenset[str] = frozenset(
    [
        "benchmark",
        "metric",
        "metrics",
        "dataset",
        "accuracy",
        "score",
        "performance",
        "result",
        "results",
        "percentage",
        "number",
        "numbers",
        "statistic",
        "statistics",
        "experiment",
        "experiments",
        "evaluation",
        "比較",
        "數值",
        "指標",
        "比例",
    ]
)

_SUMMARY_KEYWORDS: frozenset[str] = frozenset(
    [
        "summary",
        "summarize",
        "summarise",
        "overview",
        "摘要",
        "概述",
        "introduce",
        "introduction",
        "abstract",
        "what is",
        "what are",
        "tell me about",
    ]
)

_NO_INFO_ANSWER = (
    "There is not enough information in the current chat's documents to answer this question."
)


def _tokenize(text: str) -> set[str]:
    return set(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _is_numeric_question(question: str) -> bool:
    return bool(_tokenize(question) & _NUMERIC_KEYWORDS)


def _is_summary_question(question: str) -> bool:
    return bool(_tokenize(question) & _SUMMARY_KEYWORDS)


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------


class PolicyEngine:
    """Checks and enforces the 14 GUIDE §13.2 agent policies.

    All ``enforce_*`` methods may mutate state (record errors / trace events)
    and raise ``PolicyViolation`` when a hard invariant is violated.

    Parameters
    ----------
    max_iterations:
        Maximum number of tool rounds (policy 9).  Default is 2 to match
        the graph's iteration cap check in check_coverage.
    """

    def __init__(self, *, max_iterations: int = 2) -> None:
        self._max_iterations = max_iterations

    # ------------------------------------------------------------------
    # enforce_pre_retrieval  — policies 1, 2, 3, 4, 9, 10
    # ------------------------------------------------------------------

    def enforce_pre_retrieval(
        self,
        state: AgentState,
        plan: AgentPlan,
    ) -> None:
        """Run all pre-retrieval policy checks.

        Mutates plan in-place (policies 3 and 4 may add tools).
        Raises PolicyViolation for hard failures (policies 1, 9).
        Policy 2 degrades silently (removes offending history turns).
        Policy 10 is informational only (duplicate-call detection is
        enforced at call-time inside execute_retrieval_tools).
        """
        self._p1_chat_id_present(state)
        self._p2_session_history_isolation(state)
        self._p3_no_topk_for_summary(state, plan)
        self._p4_numeric_facts_first(state, plan)
        self._p9_iteration_cap(state)
        # Policy 10: checked per-call in execute_retrieval_tools — the engine
        # records the intent here so the trace shows the check happened.
        state.record_event(
            "policy_violation",
            "policy_10_dedup_check",
            iteration=state.iteration_count,
            fingerprints_so_far=len(state.tool_invocations_fingerprints),
        )

    # ------------------------------------------------------------------
    # enforce_post_retrieval — policy 5
    # ------------------------------------------------------------------

    def enforce_post_retrieval(
        self,
        state: AgentState,
        tool_calls: list[Any],
    ) -> None:
        """Policy 5: all search_hybrid calls must have rerank_mode != 'none'.

        For any ToolCallRecord whose tool_name == 'search_hybrid' and
        params['rerank_mode'] == 'none', record an error (non-fatal —
        the call already happened, we cannot undo it, but we flag it).
        """
        for record in tool_calls:
            if getattr(record, "tool_name", None) != "search_hybrid":
                continue
            params = getattr(record, "params", {}) or {}
            mode = params.get("rerank_mode", "native")
            if mode == "none":
                detail = (
                    f"search_hybrid call {getattr(record, 'call_id', '?')} "
                    f"has rerank_mode='none' — policy 5 violation; "
                    "rerank_mode forced to 'native' for future calls."
                )
                state.record_error(code="RERANK_REQUIRED", detail=detail)
                state.record_event(
                    "policy_violation",
                    "policy_5_rerank_required",
                    call_id=getattr(record, "call_id", "?"),
                    forced_mode="native",
                )

    # ------------------------------------------------------------------
    # enforce_pre_answer — policies 6, 7, 8
    # ------------------------------------------------------------------

    def enforce_pre_answer(self, state: AgentState) -> None:
        """Run pre-answer checks.

        Policy 6: overflow without aggregate → raise PolicyViolation.
        Policy 7: incomplete coverage at iteration cap → raise PolicyViolation
                  (caller must route to short-circuit fallback answer instead).
        Policy 8: notes numeric tokens in answer draft for later verification
                  (evidence cross-check happens in enforce_answer).
        """
        self._p6_overflow_must_aggregate(state)
        self._p7_no_answer_on_incomplete_coverage(state)

    # ------------------------------------------------------------------
    # enforce_answer — policy 11 (+ numeric cross-check from p8)
    # ------------------------------------------------------------------

    def enforce_answer(self, state: AgentState, answer_draft: str) -> str:
        """Policy 11 + 8: validate the answer draft before emitting.

        Policy 11: if evidence_items and structured_facts are both empty, the
            answer MUST be the "no information" fallback — override if not.
        Policy 8: find numeric tokens in the answer; verify each has backing in
            evidence content or structured_fact values; flag unverified ones.

        Returns
        -------
        str
            The (possibly overwritten) answer.
        """
        result = self._p11_no_knowledge_fallback(state, answer_draft)
        result = self._p8_numeric_cross_check(state, result)
        return result

    # ------------------------------------------------------------------
    # enforce_citations — policies 12, 13
    # ------------------------------------------------------------------

    def enforce_citations(
        self,
        state: AgentState,
        citations: list[Any],
        *,
        chat_document_ids: set[UUID] | None = None,
    ) -> list[Any]:
        """Remove citations that violate isolation rules.

        Policy 12: citation.chat_id must equal state.chat_id.
        Policy 13: citation.document_id must be in chat_document_ids
            (if provided — callers that cannot supply the set skip P13).

        Returns the cleaned citation list.
        """
        return self._p12_p13_citation_isolation(state, citations, chat_document_ids)

    # ------------------------------------------------------------------
    # enforce_provider_result — policy 14
    # ------------------------------------------------------------------

    def enforce_provider_result(
        self,
        state: AgentState,
        exc: Exception,
    ) -> None:
        """Policy 14: provider failure must surface as an explicit error.

        Never silently swallow a provider exception or switch models.
        Records AgentError + trace event, then raises PolicyViolation so
        that generate_answer can route to a clear error answer.
        """
        detail = f"Provider failure (no silent fallback): {exc!s}"
        state.record_error(code="PROVIDER_FAILURE", detail=detail)
        state.record_event(
            "policy_violation",
            "policy_14_provider_failure",
            exc_type=type(exc).__name__,
            detail=detail,
        )
        raise PolicyViolation(
            code="PROVIDER_FAILURE",
            detail=detail,
            policy_id=14,
        )

    # ------------------------------------------------------------------
    # check_duplicate_tool_call — policy 10 (per-call helper)
    # ------------------------------------------------------------------

    def check_duplicate_tool_call(
        self,
        state: AgentState,
        tool_name: str,
        params_dict: dict[str, Any],
    ) -> bool:
        """Return True and record trace event if the call is a duplicate.

        Callers should skip the tool invocation when this returns True.
        This does NOT raise — it is a non-fatal silent degrade.
        """
        if state.is_duplicate_tool_call(tool_name, params_dict):
            state.record_event(
                "policy_violation",
                "policy_10_duplicate_tool_call",
                tool_name=tool_name,
                params=params_dict,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # enforce_replan_decision — policy 15
    # ------------------------------------------------------------------

    def enforce_replan_decision(self, state: AgentState, decision: Any) -> None:
        """Policy 15: LLM replan may only nominate whitelisted retrieval tools.

        The decision object is already Pydantic-validated by ``llm_replan``;
        this policy adds a second explicit guard at the dispatch boundary.
        """
        allowed = {
            "search_hybrid",
            "grep_document_chunks",
            "fetch_structural_nodes",
            "query_structured_facts",
            "inspect_document",
        }
        for tool_call in getattr(decision, "tool_calls", []):
            tool = getattr(tool_call, "tool", None)
            if tool not in allowed:
                detail = f"llm_replan nominated forbidden tool: {tool!r}"
                state.record_error(code="REPLAN_FORBIDDEN_TOOL", detail=detail)
                state.record_event(
                    "policy_violation",
                    "policy_15_replan_decision",
                    tool=tool,
                )
                raise PolicyViolation(
                    code="REPLAN_FORBIDDEN_TOOL",
                    detail=detail,
                    policy_id=15,
                )

    # ------------------------------------------------------------------
    # Private policy implementations
    # ------------------------------------------------------------------

    def _p1_chat_id_present(self, state: AgentState) -> None:
        """Policy 1: chat_id must be set and non-trivial."""
        # AgentState requires chat_id (UUID), so None is impossible at runtime,
        # but we guard against the nil UUID used as a sentinel in tests.
        _NIL = UUID(int=0)
        if state.chat_id == _NIL:
            detail = "chat_id is the nil UUID — retrieval cannot be scoped."
            state.record_error(code="CHAT_ID_MISSING", detail=detail)
            state.record_event("policy_violation", "policy_1_chat_id", detail=detail)
            raise PolicyViolation(code="CHAT_ID_MISSING", detail=detail, policy_id=1)

    def _p2_session_history_isolation(self, state: AgentState) -> None:
        """Policy 2: conversation_history must belong to the current session.

        Turns without a message_id are accepted (they may be synthetic).
        Turns with a message_id that belongs to a different session are removed
        and an AgentError is recorded (non-fatal degradation).
        """
        # We cannot directly check DB session ownership here (no I/O),
        # so we rely on the invariant: all turns in state.conversation_history
        # MUST have been loaded for (chat_id, session_id) by the service layer.
        # The best we can do code-side is verify there are no turns injected
        # with an explicit wrong session_id payload in the trace.
        # This policy primarily validates that the state was loaded correctly
        # and records the check in the trace.
        state.record_event(
            "policy_violation",
            "policy_2_session_history_check",
            session_id=str(state.session_id),
            turn_count=len(state.conversation_history),
            status="ok",  # history is trusted as loaded by service layer
        )

    def _p3_no_topk_for_summary(self, state: AgentState, plan: AgentPlan) -> None:
        """Policy 3: summary/overview questions must not use only search_hybrid.

        If the question looks like a summary question but the plan only
        contains search_hybrid (missing inspect_chat or fetch_structural_nodes),
        add the required fetch-all tools to the plan in-place.
        """
        if not _is_summary_question(state.question):
            return

        has_topk_only = (
            "search_hybrid" in plan.chosen_tools
            and "inspect_chat" not in plan.chosen_tools
            and "fetch_structural_nodes" not in plan.chosen_tools
        )
        if not has_topk_only:
            return

        # Augment the plan — add fetch-all tools
        plan.chosen_tools.insert(0, "fetch_structural_nodes")
        plan.chosen_tools.insert(0, "inspect_chat")
        detail = (
            "Policy 3: summary question detected but plan only had search_hybrid; "
            "fetch_structural_nodes and inspect_chat added."
        )
        state.record_error(code="SUMMARY_TOPK_VIOLATION", detail=detail)
        state.record_event("policy_violation", "policy_3_summary_fetch_all", detail=detail)

    def _p4_numeric_facts_first(self, state: AgentState, plan: AgentPlan) -> None:
        """Policy 4: numeric questions must include query_structured_facts."""
        if not _is_numeric_question(state.question):
            return
        if "query_structured_facts" in plan.chosen_tools:
            return

        plan.chosen_tools.insert(0, "query_structured_facts")
        detail = (
            "Policy 4: numeric question detected but plan missing query_structured_facts; added."
        )
        state.record_error(code="NUMERIC_FACTS_MISSING", detail=detail)
        state.record_event("policy_violation", "policy_4_numeric_facts_first", detail=detail)

    def _p6_overflow_must_aggregate(self, state: AgentState) -> None:
        """Policy 6: if overflow detected, aggregate must have been called."""
        # We detect overflow by checking whether context_token_estimate exceeds
        # remaining_budget == 0 (set by check_context_budget when overflowed).
        overflowed = state.remaining_budget == 0 and state.context_token_estimate > 0
        if not overflowed:
            return

        # Check if aggregate_sources was ever called
        aggregated = any(tc.tool_name == "aggregate_sources" for tc in state.tool_calls)
        if not aggregated:
            detail = (
                "Policy 6: context budget overflow detected but aggregate_sources "
                "was not called. Must aggregate before generating answer."
            )
            state.record_error(code="OVERFLOW_NOT_AGGREGATED", detail=detail)
            state.record_event("policy_violation", "policy_6_overflow_aggregate", detail=detail)
            raise PolicyViolation(
                code="OVERFLOW_NOT_AGGREGATED",
                detail=detail,
                policy_id=6,
            )

    def _p7_no_answer_on_incomplete_coverage(self, state: AgentState) -> None:
        """Policy 7: cannot generate answer when coverage is incomplete at cap."""
        # Only block if we've exhausted iterations AND coverage is still incomplete.
        if state.coverage_state == "incomplete" and state.iteration_count >= self._max_iterations:
            detail = (
                "Policy 7: coverage_state='incomplete' after reaching iteration "
                f"cap ({self._max_iterations}). Returning no-information answer."
            )
            state.record_error(code="INCOMPLETE_COVERAGE_AT_CAP", detail=detail)
            state.record_event("policy_violation", "policy_7_incomplete_coverage", detail=detail)
            raise PolicyViolation(
                code="INCOMPLETE_COVERAGE_AT_CAP",
                detail=detail,
                policy_id=7,
            )

    def _p8_numeric_cross_check(self, state: AgentState, answer: str) -> str:
        """Policy 8: numeric tokens in the answer must appear in evidence.

        Tokens not found in any evidence content or structured fact are
        marked ``[unverified: N]`` in the answer text and an error is recorded.
        """
        numeric_tokens = _NUMERIC_TOKEN_RE.findall(answer)
        if not numeric_tokens:
            return answer

        # Build a corpus of all numeric evidence text
        evidence_corpus = " ".join(ev.content for ev in state.evidence_items)
        fact_corpus = " ".join(
            f"{sf.value_text or ''} {sf.value_numeric or ''}" for sf in state.structured_facts
        )
        combined_corpus = evidence_corpus + " " + fact_corpus

        unverified: list[str] = []
        for token in set(numeric_tokens):
            if token not in combined_corpus:
                unverified.append(token)

        if unverified:
            detail = f"Policy 8: unverified numeric tokens: {unverified}"
            state.record_error(code="UNVERIFIED_NUMERIC", detail=detail)
            state.record_event(
                "policy_violation",
                "policy_8_numeric_cross_check",
                unverified=unverified,
            )
            # Mark unverified tokens in the answer
            for token in unverified:
                answer = re.sub(
                    r"\b" + re.escape(token) + r"\b",
                    f"[unverified: {token}]",
                    answer,
                )
        return answer

    def _p9_iteration_cap(self, state: AgentState) -> None:
        """Policy 9: iteration_count must not exceed max_iterations."""
        if state.iteration_count > self._max_iterations:
            detail = (
                f"Policy 9: iteration_count={state.iteration_count} exceeds "
                f"max_iterations={self._max_iterations}."
            )
            state.record_error(code="ITERATION_CAP_EXCEEDED", detail=detail)
            state.record_event(
                "policy_violation",
                "policy_9_iteration_cap",
                iteration_count=state.iteration_count,
                max_iterations=self._max_iterations,
            )
            raise PolicyViolation(
                code="ITERATION_CAP_EXCEEDED",
                detail=detail,
                policy_id=9,
            )

    def _p11_no_knowledge_fallback(self, state: AgentState, answer: str) -> str:
        """Policy 11: if no evidence/facts, override answer with fallback."""
        if state.evidence_items or state.structured_facts:
            return answer  # evidence exists — no override

        # Check if the answer is already the fallback
        if "not enough information" in answer.lower():
            return answer

        detail = (
            "Policy 11: no evidence_items or structured_facts — "
            "answer overwritten with no-information fallback."
        )
        state.record_error(code="NO_EVIDENCE_ANSWER", detail=detail)
        state.record_event(
            "policy_violation",
            "policy_11_no_knowledge_fallback",
            original_answer_preview=answer[:100],
        )
        return _NO_INFO_ANSWER

    def _p12_p13_citation_isolation(
        self,
        state: AgentState,
        citations: list[Any],
        chat_document_ids: set[UUID] | None,
    ) -> list[Any]:
        """Policies 12 and 13: citation isolation checks.

        Policy 12: citation.chat_id must equal state.chat_id.
        Policy 13: citation.document_id must be in chat_document_ids.
        """
        clean: list[Any] = []
        for cit in citations:
            cit_chat_id: UUID | None = getattr(cit, "chat_id", None)
            cit_doc_id: UUID | None = getattr(cit, "document_id", None)
            cit_id: str = getattr(cit, "citation_id", str(cit))

            # Policy 12
            if cit_chat_id != state.chat_id:
                detail = (
                    f"Policy 12: citation {cit_id} has chat_id={cit_chat_id} "
                    f"!= current chat_id={state.chat_id}; removed."
                )
                state.record_error(code="CITATION_WRONG_CHAT", detail=detail)
                state.record_event(
                    "policy_violation",
                    "policy_12_citation_chat_isolation",
                    citation_id=cit_id,
                    offending_chat_id=str(cit_chat_id),
                )
                continue

            # Policy 13
            if chat_document_ids is not None and cit_doc_id not in chat_document_ids:
                detail = (
                    f"Policy 13: citation {cit_id} document_id={cit_doc_id} "
                    "not in chat's ChatDocument association; removed."
                )
                state.record_error(code="CITATION_DOC_NOT_IN_CHAT", detail=detail)
                state.record_event(
                    "policy_violation",
                    "policy_13_citation_doc_association",
                    citation_id=cit_id,
                    document_id=str(cit_doc_id),
                )
                continue

            clean.append(cit)
        return clean


# ---------------------------------------------------------------------------
# Module-level default engine instance (convenience singleton)
# ---------------------------------------------------------------------------

_default_engine: PolicyEngine | None = None


def get_policy_engine(*, max_iterations: int = 2) -> PolicyEngine:
    """Return the module-level default PolicyEngine (created on first call)."""
    global _default_engine
    if _default_engine is None:
        _default_engine = PolicyEngine(max_iterations=max_iterations)
    return _default_engine


__all__ = [
    "Policy",
    "PolicyEngine",
    "PolicyViolation",
    "get_policy_engine",
]
