"""Phase D semantic coverage tests (≤10 tests)."""

from __future__ import annotations

import uuid

import pytest

from app.agent.nodes.check_coverage import check_coverage
from app.agent.state import AgentState, CoverageRequirement, EvidenceItem, make_evidence_id

_CHAT_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_SESSION_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_DOC_ID = uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")


def _evidence(content: str, *, vector_score: float | None = None) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=make_evidence_id("search_hybrid", str(uuid.uuid4()), _DOC_ID),
        source_type="chunk",
        document_id=_DOC_ID,
        source_node_id="node-1",
        page_start=1,
        page_end=1,
        content=content,
        score=vector_score,
        vector_score=vector_score,
        origin_tool="search_hybrid",
    )


@pytest.mark.asyncio
async def test_vector_score_satisfies_requirement_without_token_overlap() -> None:
    state = AgentState(
        chat_id=_CHAT_ID,
        session_id=_SESSION_ID,
        question="What are the ablation results?",
        coverage_requirements=[CoverageRequirement(requirement_id="r1", description="ablation")],
        evidence_items=[_evidence("Removing the retriever lowers F1.", vector_score=0.72)],
    )
    result = await check_coverage(state)
    assert result["coverage_state"] == "complete"
    assert result["coverage_requirements"][0].satisfied is True


@pytest.mark.asyncio
async def test_semantic_synonym_fallback_covers_ablation_without_component() -> None:
    state = AgentState(
        chat_id=_CHAT_ID,
        session_id=_SESSION_ID,
        question="What are the ablation results?",
        coverage_requirements=[CoverageRequirement(requirement_id="r1", description="ablation")],
        evidence_items=[_evidence("The study reports performance without component X.")],
    )
    result = await check_coverage(state)
    assert result["coverage_requirements"][0].satisfied is True
