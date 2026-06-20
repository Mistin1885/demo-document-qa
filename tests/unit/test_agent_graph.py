"""Unit tests for Phase 7.3 agent graph nodes.

Test density cap: ≤ 10 items per file (CLAUDE.md §12.1).
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from app.agent.budget import ContextBudgetManager
from app.agent.nodes.check_context_budget import is_overflow
from app.agent.nodes.check_coverage import check_coverage
from app.agent.nodes.merge_evidence_workspace import merge_evidence_workspace
from app.agent.nodes.plan_information_needs import (
    _is_numeric_question,
    _is_summary_question,
    plan_information_needs,
)
from app.agent.nodes.validate_citations import validate_citations
from app.agent.nodes.validate_scope_isolation import validate_scope_isolation
from app.agent.state import (
    AgentState,
    CitationDraft,
    CoverageRequirement,
    EvidenceItem,
    make_evidence_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**kwargs: Any) -> AgentState:
    base = dict(
        chat_id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        session_id=uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        question="What is the main contribution?",
    )
    base.update(kwargs)
    return AgentState(**base)


def _make_evidence(
    doc_id: uuid.UUID | None = None, content: str = "test content", score: float = 0.5
) -> EvidenceItem:
    d = doc_id or uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    return EvidenceItem(
        evidence_id=make_evidence_id("search_hybrid", "node-1", d),
        source_type="chunk",
        document_id=d,
        source_node_id="node-1",
        page_start=1,
        page_end=2,
        content=content,
        score=score,
        origin_tool="search_hybrid",
    )


# ---------------------------------------------------------------------------
# 1. Planner routes summary questions correctly
# ---------------------------------------------------------------------------


def test_planner_summary_routing() -> None:
    assert _is_summary_question("Give me a summary of the paper")
    assert not _is_summary_question("What benchmark did they use?")


# ---------------------------------------------------------------------------
# 2. Planner routes numeric/benchmark questions correctly
# ---------------------------------------------------------------------------


def test_planner_numeric_routing() -> None:
    assert _is_numeric_question("What benchmark accuracy did they achieve?")
    assert not _is_numeric_question("Who are the authors?")


# ---------------------------------------------------------------------------
# 3. plan_information_needs — summary → structural tools plus broad evidence search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_summary_chosen_tools() -> None:
    state = _make_state(question="Give me a summary overview of this paper")
    result = await plan_information_needs(state)
    tools = result["plan"].chosen_tools
    assert "inspect_chat" in tools
    assert "fetch_structural_nodes" in tools
    assert "search_hybrid" in tools


# ---------------------------------------------------------------------------
# 4. plan_information_needs — benchmark → query_structured_facts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_numeric_chosen_tools() -> None:
    state = _make_state(question="What benchmark metric did the model achieve?")
    result = await plan_information_needs(state)
    tools = result["plan"].chosen_tools
    assert "query_structured_facts" in tools


# ---------------------------------------------------------------------------
# 5. merge_evidence_workspace — de-duplicates identical evidence_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_deduplication() -> None:
    ev = _make_evidence()
    # Add same evidence twice to simulate duplicate
    state = _make_state(evidence_items=[ev, ev])
    result = await merge_evidence_workspace(state)
    assert len(result["evidence_items"]) == 1


# ---------------------------------------------------------------------------
# 6. check_context_budget overflow detection routes correctly
# ---------------------------------------------------------------------------


def test_budget_overflow_detection() -> None:
    # Create evidence that exceeds the budget threshold
    big_content = "x " * 10_000  # ~20000 chars -> overflow
    ev = EvidenceItem(
        evidence_id=make_evidence_id("search_hybrid", "node-big", uuid.uuid4()),
        source_type="chunk",
        document_id=uuid.uuid4(),
        source_node_id="node-big",
        page_start=1,
        page_end=2,
        content=big_content,
        score=1.0,
        origin_tool="search_hybrid",
    )
    state = _make_state(evidence_items=[ev])
    bm = ContextBudgetManager()
    assert is_overflow(state, bm) is True


# ---------------------------------------------------------------------------
# 7. check_coverage — unsatisfied req with iteration_count < 2 → incomplete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coverage_incomplete_routes_to_gap() -> None:
    req = CoverageRequirement(requirement_id="req-0", description="method details")
    state = _make_state(
        coverage_requirements=[req],
        iteration_count=0,
    )
    result = await check_coverage(state)
    assert result["coverage_state"] == "incomplete"


# ---------------------------------------------------------------------------
# 8. validate_citations — extracts [c1] markers and maps to evidence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_citations_extracts_markers() -> None:
    ev = _make_evidence(content="The model achieves 95% accuracy.")
    state = _make_state(
        evidence_items=[ev],
        answer="The paper shows [c1] great results.",
    )
    result = await validate_citations(state)
    citations = result["citations"]
    assert len(citations) == 1
    assert citations[0].citation_id == "cit-1"
    assert citations[0].chat_id == state.chat_id


# ---------------------------------------------------------------------------
# 9. validate_scope_isolation — removes wrong-chat citations and records error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scope_isolation_removes_wrong_chat_citation() -> None:
    wrong_chat_id = uuid.UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
    ev = _make_evidence()
    cit_good = CitationDraft(
        citation_id="cit-1",
        chat_id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        document_id=ev.document_id,
        document_name="doc",
        page_start=1,
        page_end=2,
        source_node_id="node-1",
        excerpt="ok",
    )
    cit_bad = CitationDraft(
        citation_id="cit-2",
        chat_id=wrong_chat_id,
        document_id=ev.document_id,
        document_name="doc",
        page_start=1,
        page_end=2,
        source_node_id="node-1",
        excerpt="bad",
    )
    state = _make_state(citations=[cit_good, cit_bad])
    result = await validate_scope_isolation(state)
    assert len(result["citations"]) == 1
    assert result["citations"][0].citation_id == "cit-1"
    # Phase 7.4: PolicyEngine uses CITATION_WRONG_CHAT (was ISOLATION_VIOLATION)
    isolation_errors = [
        e for e in result["errors"] if e.code in ("ISOLATION_VIOLATION", "CITATION_WRONG_CHAT")
    ]
    assert len(isolation_errors) == 1


# ---------------------------------------------------------------------------
# 10. Trace events: node_enter / node_exit recorded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_events_recorded() -> None:
    state = _make_state(question="What is the paper about?")
    result = await plan_information_needs(state)
    # plan_information_needs records enter/exit on state.debug_trace
    trace = result["debug_trace"]
    kinds = [e.kind for e in trace.events]
    names = [e.name for e in trace.events]
    assert "node_enter" in kinds
    assert "node_exit" in kinds
    assert "plan_information_needs" in names
