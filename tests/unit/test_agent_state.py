"""Unit tests for AgentState (CLAUDE.md §12.1 — ≤10 items per file).

Tests cover the contract, not the surface:
 1. Happy path: construct a valid state, verify defaults.
 2. add_evidence de-duplication.
 3. record_tool_call registers fingerprint; same params → same fingerprint.
 4. extra="forbid" raises ValidationError on unknown fields.
 5. clone() is a deep copy (mutation isolation).
 6. record_event appends a TraceEvent with a datetime ts.
 7. coverage_state Literal validation.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

from app.agent.state import (
    AgentState,
    EvidenceItem,
    make_evidence_id,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHAT_ID = uuid.uuid4()
SESSION_ID = uuid.uuid4()
DOC_ID = uuid.uuid4()


def _base_state(**kwargs: object) -> AgentState:
    return AgentState(
        chat_id=CHAT_ID,
        session_id=SESSION_ID,
        question="What is the main contribution?",
        **kwargs,
    )


def _make_evidence(idx: int = 0) -> EvidenceItem:
    node_id = f"node-{idx}"
    return EvidenceItem(
        evidence_id=make_evidence_id("search_hybrid", node_id, str(DOC_ID)),
        source_type="chunk",
        document_id=DOC_ID,
        source_node_id=node_id,
        page_start=1,
        page_end=2,
        content=f"Evidence content {idx}",
        origin_tool="search_hybrid",
    )


# ---------------------------------------------------------------------------
# Test 1 — happy path: defaults are correct
# ---------------------------------------------------------------------------


def test_agent_state_defaults() -> None:
    state = _base_state()
    assert state.chat_id == CHAT_ID
    assert state.session_id == SESSION_ID
    assert state.question == "What is the main contribution?"
    assert state.iteration_count == 0
    assert state.coverage_state == "pending"
    assert state.answer is None
    assert state.token_count_is_estimate is False
    assert len(state.tool_invocations_fingerprints) == 0
    assert len(state.evidence_items) == 0


# ---------------------------------------------------------------------------
# Test 2 — add_evidence de-duplicates by evidence_id
# ---------------------------------------------------------------------------


def test_add_evidence_dedup() -> None:
    state = _base_state()
    ev = _make_evidence(0)
    state.add_evidence(ev)
    state.add_evidence(ev)  # same object → same evidence_id
    assert len(state.evidence_items) == 1


# ---------------------------------------------------------------------------
# Test 3 — record_tool_call registers fingerprint; duplicate detected
# ---------------------------------------------------------------------------


def test_record_tool_call_fingerprint() -> None:
    state = _base_state()
    params = {"query": "attention mechanism", "top_k": 8}
    state.record_tool_call(call_id="c1", tool_name="search_hybrid", params=params)
    assert len(state.tool_invocations_fingerprints) == 1
    assert state.is_duplicate_tool_call("search_hybrid", params) is True
    # Different params → not a duplicate
    assert state.is_duplicate_tool_call("search_hybrid", {"query": "other"}) is False


# ---------------------------------------------------------------------------
# Test 4 — extra="forbid" raises ValidationError for unknown fields
# ---------------------------------------------------------------------------


def test_extra_forbid_raises() -> None:
    with pytest.raises(ValidationError):
        AgentState(
            chat_id=CHAT_ID,
            session_id=SESSION_ID,
            question="test?",
            unknown_field="oops",  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# Test 5 — clone() produces a deep copy
# ---------------------------------------------------------------------------


def test_clone_deep_copy() -> None:
    state = _base_state()
    ev = _make_evidence(0)
    state.add_evidence(ev)

    copy_state = state.clone()
    # Mutate the copy
    copy_state.iteration_count = 99
    new_ev = _make_evidence(1)
    copy_state.add_evidence(new_ev)

    # Original is unaffected
    assert state.iteration_count == 0
    assert len(state.evidence_items) == 1


# ---------------------------------------------------------------------------
# Test 6 — record_event appends a TraceEvent with datetime ts
# ---------------------------------------------------------------------------


def test_record_event_appends() -> None:
    state = _base_state()
    state.record_event("node_enter", "inspect_scope", chat_id=str(CHAT_ID))
    assert len(state.debug_trace.events) == 1
    event = state.debug_trace.events[0]
    assert event.kind == "node_enter"
    assert event.name == "inspect_scope"
    assert isinstance(event.ts, datetime)
    assert event.payload["chat_id"] == str(CHAT_ID)


# ---------------------------------------------------------------------------
# Test 7 — coverage_state Literal validation
# ---------------------------------------------------------------------------


def test_coverage_state_literal_invalid() -> None:
    with pytest.raises(ValidationError):
        _base_state(coverage_state="done")  # type: ignore[arg-type]
