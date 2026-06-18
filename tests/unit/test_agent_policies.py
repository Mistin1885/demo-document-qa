"""Unit tests for Phase 7.4 agent policies (GUIDE §13.2).

Test density cap: ≤ 10 items per file (CLAUDE.md §12.1).

Grouping strategy:
  - test_isolation_layered        (policies 1, 12, 13)
  - test_no_topk_for_summary      (policies 3, 6)
  - test_numeric_facts_first      (policies 4, 8)
  - test_retrieval_must_hybrid_rerank  (policy 5)
  - test_no_answer_on_incomplete_or_empty  (policies 7, 11)
  - test_tool_round_and_dup_caps  (policies 9, 10)
  - test_provider_failure_explicit (policy 14)
  - test_session_history_isolation (policy 2)
  - parametrize over violation codes × 2 representative groups
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from app.agent.policies import PolicyEngine, PolicyViolation
from app.agent.state import (
    AgentPlan,
    AgentState,
    CitationDraft,
    EvidenceItem,
    make_evidence_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHAT_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_SESSION_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_DOC_ID = uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
_OTHER_CHAT = uuid.UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
_OTHER_DOC = uuid.UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")


def _state(**kwargs: Any) -> AgentState:
    base: dict[str, Any] = dict(
        chat_id=_CHAT_ID,
        session_id=_SESSION_ID,
        question="What is the main contribution?",
    )
    base.update(kwargs)
    return AgentState(**base)


def _plan(*tools: str) -> AgentPlan:
    return AgentPlan(goal="test", chosen_tools=list(tools))


def _ev(content: str = "accuracy 95.3%") -> EvidenceItem:
    return EvidenceItem(
        evidence_id=make_evidence_id("search_hybrid", "n1", _DOC_ID),
        source_type="chunk",
        document_id=_DOC_ID,
        source_node_id="n1",
        page_start=1,
        page_end=2,
        content=content,
        origin_tool="search_hybrid",
    )


def _cit(chat_id: uuid.UUID = _CHAT_ID, doc_id: uuid.UUID = _DOC_ID) -> CitationDraft:
    return CitationDraft(
        citation_id="cit-1",
        chat_id=chat_id,
        document_id=doc_id,
        document_name="doc",
        page_start=1,
        page_end=1,
        source_node_id="n1",
        excerpt="x",
    )


engine = PolicyEngine(max_iterations=2)


# ---------------------------------------------------------------------------
# 1. isolation_layered — policies 1, 12, 13
# ---------------------------------------------------------------------------


def test_isolation_layered() -> None:
    """Policy 1: nil chat_id raises; Policies 12+13: wrong-chat/doc citations removed."""
    # Policy 1
    nil_state = _state(chat_id=uuid.UUID(int=0))
    with pytest.raises(PolicyViolation) as exc_info:
        engine.enforce_pre_retrieval(nil_state, _plan("search_hybrid"))
    assert exc_info.value.policy_id == 1

    # Policies 12 + 13: wrong chat_id citation removed
    s = _state()
    bad_cit = _cit(chat_id=_OTHER_CHAT)
    good_cit = _cit()
    result = engine.enforce_citations(s, [good_cit, bad_cit])
    assert len(result) == 1
    assert result[0].chat_id == _CHAT_ID

    # Policy 13: wrong document_id removed; valid doc_id retained
    s2 = _state()
    wrong_doc_cit = _cit(doc_id=_OTHER_DOC)
    result2 = engine.enforce_citations(s2, [wrong_doc_cit], chat_document_ids={_DOC_ID})
    assert len(result2) == 0
    good_doc_cit = _cit(doc_id=_DOC_ID)
    result3 = engine.enforce_citations(s2, [good_doc_cit], chat_document_ids={_DOC_ID})
    assert len(result3) == 1


# ---------------------------------------------------------------------------
# 2. test_session_history_isolation — policy 2
# ---------------------------------------------------------------------------


def test_session_history_isolation() -> None:
    """Policy 2: session history check records trace event with status='ok'."""
    s = _state()
    engine.enforce_pre_retrieval(s, _plan("search_hybrid"))
    trace_names = [e.name for e in s.debug_trace.events]
    assert "policy_2_session_history_check" in trace_names


# ---------------------------------------------------------------------------
# 3. no_topk_for_summary — policies 3, 6
# ---------------------------------------------------------------------------


def test_no_topk_for_summary() -> None:
    """Policy 3: summary question with only search_hybrid → fetch-all tools added.
    Policy 6: overflow without aggregate → raises PolicyViolation."""
    # Policy 3
    s = _state(question="Give me a summary overview of all documents")
    plan = _plan("search_hybrid")  # missing fetch_structural_nodes
    engine.enforce_pre_retrieval(s, plan)
    assert "fetch_structural_nodes" in plan.chosen_tools
    assert "inspect_chat" in plan.chosen_tools

    # Policy 6: overflowed but no aggregate_sources call
    s6 = _state(remaining_budget=0, context_token_estimate=999)
    with pytest.raises(PolicyViolation) as exc_info:
        engine.enforce_pre_answer(s6)
    assert exc_info.value.policy_id == 6


# ---------------------------------------------------------------------------
# 4. numeric_facts_first — policies 4, 8
# ---------------------------------------------------------------------------


def test_numeric_facts_first() -> None:
    """Policy 4: numeric question missing query_structured_facts → added.
    Policy 8: numeric token not in evidence → marked [unverified: ...]."""
    # Policy 4
    s = _state(question="What accuracy score did they achieve on the benchmark?")
    plan = _plan("search_hybrid")
    engine.enforce_pre_retrieval(s, plan)
    assert "query_structured_facts" in plan.chosen_tools

    # Policy 8: token "99.9%" not in evidence
    s8 = _state(evidence_items=[_ev("no numeric data here")])
    result = engine.enforce_answer(s8, "They achieved 99.9% accuracy.")
    assert "unverified" in result


# ---------------------------------------------------------------------------
# 5. retrieval_must_hybrid_rerank — policy 5
# ---------------------------------------------------------------------------


def test_retrieval_must_hybrid_rerank() -> None:
    """Policy 5: search_hybrid call with rerank_mode='none' records an error."""
    from app.agent.state import ToolCallRecord  # noqa: PLC0415

    s = _state()
    bad_call = ToolCallRecord(
        call_id="x",
        tool_name="search_hybrid",
        params={"query": "test", "rerank_mode": "none"},
        status="ok",
    )
    engine.enforce_post_retrieval(s, [bad_call])
    codes = [e.code for e in s.errors]
    assert "RERANK_REQUIRED" in codes


# ---------------------------------------------------------------------------
# 6. no_answer_on_incomplete_or_empty — policies 7, 11
# ---------------------------------------------------------------------------


def test_no_answer_on_incomplete_or_empty() -> None:
    """Policy 7: incomplete coverage at iteration cap → raise PolicyViolation.
    Policy 11: no evidence → answer overridden with fallback."""
    # Policy 7: iteration_count == max_iterations and incomplete
    s7 = _state(coverage_state="incomplete", iteration_count=2)
    with pytest.raises(PolicyViolation) as exc_info:
        engine.enforce_pre_answer(s7)
    assert exc_info.value.policy_id == 7

    # Policy 11: no evidence → enforce_answer overwrites
    s11 = _state()
    answer = engine.enforce_answer(s11, "The model is excellent.")
    assert "not enough information" in answer.lower()

    # Policy 11: answer already correct fallback → passes through
    s11b = _state()
    passthru = engine.enforce_answer(
        s11b,
        "There is not enough information in the current chat's documents to answer this question.",
    )
    assert "not enough information" in passthru.lower()


# ---------------------------------------------------------------------------
# 7. tool_round_and_dup_caps — policies 9, 10
# ---------------------------------------------------------------------------


def test_tool_round_and_dup_caps() -> None:
    """Policy 9: iteration_count > max → raises.  Policy 10: dup call → skipped."""
    # Policy 9
    s9 = _state(iteration_count=3)
    with pytest.raises(PolicyViolation) as exc_info:
        engine.enforce_pre_retrieval(s9, _plan("search_hybrid"))
    assert exc_info.value.policy_id == 9

    # Policy 10: duplicate detection returns True for second identical call
    s10 = _state()
    params = {
        "query": "hello",
        "top_k": 8,
        "rerank_mode": "native",
        "max_tokens": 5000,
        "document_ids": None,
        "source_types": None,
    }
    s10.record_tool_call(
        call_id="c1",
        tool_name="search_hybrid",
        params=params,
    )
    assert engine.check_duplicate_tool_call(s10, "search_hybrid", params) is True


# ---------------------------------------------------------------------------
# 8. provider_failure_explicit — policy 14
# ---------------------------------------------------------------------------


def test_provider_failure_explicit() -> None:
    """Policy 14: any provider exception → PolicyViolation, no silent fallback."""
    s = _state()
    exc = RuntimeError("connection refused")
    with pytest.raises(PolicyViolation) as exc_info:
        engine.enforce_provider_result(s, exc)
    assert exc_info.value.policy_id == 14
    assert exc_info.value.code == "PROVIDER_FAILURE"
    codes = [e.code for e in s.errors]
    assert "PROVIDER_FAILURE" in codes


# ---------------------------------------------------------------------------
# 9. parametrize — policy violation codes surface in state.errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "plan_tools", "expected_error_code"),
    [
        (
            "Give me a summary of the paper",
            ["search_hybrid"],
            "SUMMARY_TOPK_VIOLATION",
        ),
        (
            "What accuracy score did they achieve on the benchmark?",
            ["search_hybrid"],
            "NUMERIC_FACTS_MISSING",
        ),
    ],
)
def test_plan_augmentation_error_codes(
    question: str, plan_tools: list[str], expected_error_code: str
) -> None:
    """Policies 3 and 4 augment plan AND record AgentErrors."""
    s = _state(question=question)
    plan = _plan(*plan_tools)
    engine.enforce_pre_retrieval(s, plan)
    codes = [e.code for e in s.errors]
    assert expected_error_code in codes
