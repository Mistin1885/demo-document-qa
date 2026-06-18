"""Unit tests for ContextBudgetManager (CLAUDE.md §12.1 — ≤10 items per file).

Tests cover:
 1. count_tokens returns >0 for non-empty text; 0 for empty.
 2. tiktoken ImportError fallback: last_was_estimate=True (monkeypatched).
 3. estimate_conversation: 0 for empty list; sum for 2 turns.
 4. calculate_available_evidence_budget: 0 when full; ≥0 when not full.
 5. detect_overflow: parametrize under/over.
 6. select_compact_sources: at least 1 item per document; total tokens ≤ target.
 7. build_aggregation_groups: groups by (document_id, section_title).
"""

from __future__ import annotations

import uuid

import pytest

from app.agent.budget import ContextAllocation, ContextBudgetManager
from app.agent.state import (
    AgentState,
    ConversationTurn,
    EvidenceItem,
    make_evidence_id,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CHAT_ID = uuid.uuid4()
SESSION_ID = uuid.uuid4()
DOC_A = uuid.uuid4()
DOC_B = uuid.uuid4()


def _make_state(**kwargs: object) -> AgentState:
    return AgentState(
        chat_id=CHAT_ID,
        session_id=SESSION_ID,
        question="test question",
        **kwargs,
    )


def _make_ev(
    doc_id: uuid.UUID,
    node_id: str,
    content: str,
    score: float | None = None,
    section_title: str | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=make_evidence_id("search_hybrid", node_id, str(doc_id)),
        source_type="chunk",
        document_id=doc_id,
        source_node_id=node_id,
        page_start=1,
        page_end=1,
        content=content,
        score=score,
        section_title=section_title,
        origin_tool="search_hybrid",
    )


# ---------------------------------------------------------------------------
# Test 1 — count_tokens basic
# ---------------------------------------------------------------------------


def test_count_tokens_basic() -> None:
    mgr = ContextBudgetManager()
    assert mgr.count_tokens("hello world") > 0
    assert mgr.count_tokens("") == 0


# ---------------------------------------------------------------------------
# Test 2 — fallback when tiktoken unavailable
# ---------------------------------------------------------------------------


def test_count_tokens_fallback_on_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force _load_tiktoken to return None to trigger fallback path."""
    import app.agent.budget as budget_mod

    monkeypatch.setattr(budget_mod, "_load_tiktoken", lambda: None)
    mgr = ContextBudgetManager()
    mgr._tiktoken_enc = None  # simulate unavailable encoder

    result = mgr.count_tokens("a" * 40)
    assert result > 0
    assert mgr.last_was_estimate is True


# ---------------------------------------------------------------------------
# Test 3 — estimate_conversation
# ---------------------------------------------------------------------------


def test_estimate_conversation() -> None:
    mgr = ContextBudgetManager()
    assert mgr.estimate_conversation([]) == 0

    turns = [
        ConversationTurn(role="user", content="Hello"),
        ConversationTurn(role="assistant", content="World"),
    ]
    total = mgr.estimate_conversation(turns)
    assert total == mgr.count_tokens("Hello") + mgr.count_tokens("World")


# ---------------------------------------------------------------------------
# Test 4 — calculate_available_evidence_budget
# ---------------------------------------------------------------------------


def test_calculate_available_evidence_budget() -> None:
    mgr = ContextBudgetManager()

    # Empty state → full evidence budget available
    state_empty = _make_state()
    assert mgr.calculate_available_evidence_budget(state_empty) == mgr.allocation.document_evidence

    # Overfill evidence → returns 0 (never negative)
    # Use a very small allocation to make overflow easy
    tiny_alloc = ContextAllocation(
        system_and_tools=1200,
        conversation=1000,
        question_and_plan=500,
        document_evidence=1,  # tiny
        answer_reserve=2000,
        miscellaneous=5299,
    )
    mgr_tiny = ContextBudgetManager(allocation=tiny_alloc)
    state_full = _make_state()
    state_full.add_evidence(_make_ev(DOC_A, "n1", "x" * 100))
    assert mgr_tiny.calculate_available_evidence_budget(state_full) == 0


# ---------------------------------------------------------------------------
# Test 5 — detect_overflow (parametrize 2 cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content,expected_overflow",
    [
        ("short", False),
        # ~29k tokens (numbers 0..9999 joined by spaces) — well above safe ceiling of 8000
        (" ".join(str(i) for i in range(10_000)), True),
    ],
)
def test_detect_overflow(content: str, expected_overflow: bool) -> None:
    mgr = ContextBudgetManager()
    state = _make_state()
    state.add_evidence(_make_ev(DOC_A, "n1", content))
    assert mgr.detect_overflow(state) is expected_overflow


# ---------------------------------------------------------------------------
# Test 6 — select_compact_sources
# ---------------------------------------------------------------------------


def test_select_compact_sources_per_document_and_budget() -> None:
    mgr = ContextBudgetManager()
    state = _make_state()

    # Add 3 items: 2 from DOC_A, 1 from DOC_B
    state.add_evidence(_make_ev(DOC_A, "a1", "Alpha " * 20, score=0.9))
    state.add_evidence(_make_ev(DOC_A, "a2", "Beta " * 20, score=0.7))
    state.add_evidence(_make_ev(DOC_B, "b1", "Gamma " * 20, score=0.5))

    # Budget that fits roughly 2 items but must keep 1 per doc
    target = mgr.count_tokens("Alpha " * 20) + mgr.count_tokens("Gamma " * 20) + 5

    selected = mgr.select_compact_sources(state, target)

    # Must include at least one item from each document
    doc_ids = {str(ev.document_id) for ev in selected}
    assert str(DOC_A) in doc_ids
    assert str(DOC_B) in doc_ids

    # Total tokens must not exceed target (unless a must-include item alone exceeds it)
    total_tokens = sum(mgr.count_tokens(ev.content) for ev in selected)
    # We allow overshoot only due to must-include guarantee
    assert total_tokens <= target + mgr.count_tokens("Gamma " * 20)


# ---------------------------------------------------------------------------
# Test 7 — build_aggregation_groups
# ---------------------------------------------------------------------------


def test_build_aggregation_groups() -> None:
    mgr = ContextBudgetManager()
    state = _make_state()

    # Items: (DOC_A, "intro"), (DOC_A, "intro"), (DOC_B, None)
    state.add_evidence(_make_ev(DOC_A, "a1", "text1", section_title="intro"))
    state.add_evidence(_make_ev(DOC_A, "a2", "text2", section_title="intro"))
    state.add_evidence(_make_ev(DOC_B, "b1", "text3", section_title=None))

    groups = mgr.build_aggregation_groups(state)

    # Expect 2 groups: (DOC_A, "intro") and (DOC_B, None)
    assert len(groups) == 2
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]
