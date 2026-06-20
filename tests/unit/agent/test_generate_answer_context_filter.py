"""Tests for answer-context evidence filtering."""

from __future__ import annotations

import uuid

from app.agent.nodes.generate_answer import _build_context_block
from app.agent.state import AgentState, EvidenceItem, make_evidence_id

_CHAT_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_SESSION_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_DOC_ID = uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")


def _ev(content: str, idx: int) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=make_evidence_id("test", f"node-{idx}", _DOC_ID),
        source_type="figure_caption",
        document_id=_DOC_ID,
        source_node_id=f"node-{idx}",
        page_start=idx,
        page_end=idx,
        content=content,
        score=1.0,
        vector_score=None,
        section_title=None,
        heading_path=None,
        origin_tool="test",
    )


def test_context_filter_removes_wrong_same_modality_label_when_exact_exists() -> None:
    state = AgentState(
        chat_id=_CHAT_ID,
        session_id=_SESSION_ID,
        question="What exact token and API-call contrast is shown in Figure 2?",
        evidence_items=[
            _ev("Figure 2 compares GraphRAG 610 × 1,000 tokens with LightRAG <100 tokens.", 2),
            _ev("Figure 1 shows the LightRAG architecture.", 1),
            _ev("Table 2 reports unrelated ablation results.", 3),
        ],
    )

    context = _build_context_block(state)

    assert "Figure 2" in context
    assert "610" in context
    assert "Figure 1" not in context
    assert "Table 2" not in context
