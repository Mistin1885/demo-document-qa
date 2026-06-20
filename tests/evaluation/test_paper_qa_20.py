"""20-question LightRAG paper QA benchmark.

This exercises the repo's LangGraph QA path with deep_qa_mode disabled and
enabled.  Each case is grounded in 2410.05779v3.pdf and asserts answer
coverage, not just citation shape.

Test density cap: 2 tests in this file (≤10).
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.agent.graph import build_graph
from app.agent.nodes.persist_messages import InMemoryMessageStore
from app.agent.state import AgentState, ConversationTurn, GenerationConfig
from app.agent.tools._invocation import ToolDeps
from app.providers.extractive import ExtractiveEvidenceChatProvider
from app.retrieval.models import RetrievalRequest, RetrievalResponse, SearchHit

_FIXTURE = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "paper_qa_2410_05779v3.json"
_CHAT_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_SESSION_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_DOC_ID = uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")


class HistoryTurnSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class PaperQACase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: str
    question: str
    must: list[str] = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    requires_table: bool = False
    history: list[HistoryTurnSpec] = Field(default_factory=list)


class _ScoredRetrieval:
    """Small deterministic retrieval fake over the case's PDF-grounded evidence."""

    def __init__(self, evidence: list[str], source_type: str) -> None:
        self.evidence = evidence
        self.source_type = source_type
        self.requests: list[RetrievalRequest] = []

    async def search(self, request: RetrievalRequest) -> RetrievalResponse:
        self.requests.append(request)
        query_terms = _terms(request.query)
        scored: list[tuple[int, int, str]] = []
        for idx, content in enumerate(self.evidence):
            score = len(query_terms & _terms(content))
            # Give table / figure / formula queries enough lexical signal to
            # surface the relevant modality-specific chunk.
            lower = content.lower()
            if "table" in lower and "table" in query_terms:
                score += 3
            if "figure" in lower and {"figure", "cost", "architecture"} & query_terms:
                score += 3
            if any(ch.isdigit() for ch in content) and {"performance", "cost", "numbers", "rates"} & query_terms:
                score += 2
            scored.append((score, -idx, content))

        ranked = sorted(scored, reverse=True)
        # Return a bounded but non-empty set. If all scores are zero, the QA
        # mechanism effectively failed to target the case evidence; returning
        # the top document chunk still lets coverage/answer tests reveal gaps.
        limit = min(max(request.final_top_k, 4), len(ranked))
        hits: list[SearchHit] = []
        for rank, (_score, neg_idx, content) in enumerate(ranked[:limit], start=1):
            idx = -neg_idx
            hits.append(
                SearchHit(
                    vespa_document_id=f"id::document_chunk::paperqa-{idx}",
                    chat_id=str(request.chat_id),
                    document_id=str(_DOC_ID),
                    source_node_id=f"paperqa-node-{idx}",
                    source_type=self.source_type,
                    content=content,
                    page_start=1 + idx,
                    page_end=1 + idx,
                    order_index=idx,
                    vector_score=0.82 if _score > 0 else 0.2,
                    fusion_score=float(_score),
                    final_score=float(_score),
                    final_rank=rank,
                )
            )
        return RetrievalResponse(hits=hits)


@asynccontextmanager
async def _null_session() -> AsyncIterator[Any]:
    yield MagicMock()


def _terms(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", text)}


def _source_type(case_type: str) -> str:
    if case_type == "table":
        return "table_record"
    if case_type == "figure":
        return "figure_caption"
    if case_type == "formula":
        return "raw_block"
    if case_type == "performance":
        return "performance_fact"
    return "chunk"


def _load_cases() -> list[PaperQACase]:
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return [PaperQACase.model_validate(item) for item in data["cases"]]


async def _run_case(case: PaperQACase, *, deep_qa_mode: bool) -> tuple[str, AgentState, _ScoredRetrieval]:
    retrieval = _ScoredRetrieval(case.evidence, _source_type(case.type))
    provider = ExtractiveEvidenceChatProvider()
    deps = ToolDeps(
        retrieval_service=retrieval,  # type: ignore[arg-type]
        chat_provider=provider,
        session_factory=_null_session,
    )
    state = AgentState(
        chat_id=_CHAT_ID,
        session_id=_SESSION_ID,
        question=case.question,
        conversation_history=[
            ConversationTurn(role=turn.role, content=turn.content) for turn in case.history
        ],
        generation_config=GenerationConfig(
            deep_qa_mode=deep_qa_mode,
            context_window=200_000 if deep_qa_mode else 20_000,
            max_answer_tokens=4096,
        ),
    )
    graph = build_graph(
        deps=deps,
        chat_provider=provider,
        message_store=InMemoryMessageStore(expected_chat_id=_CHAT_ID),
    )
    container = await graph.ainvoke({"state": state.model_dump()})
    final = AgentState.model_validate(container["state"])
    return final.answer or "", final, retrieval


def _assert_case_answer(case: PaperQACase, answer: str, final: AgentState, retrieval: _ScoredRetrieval) -> None:
    lower = answer.lower()
    missing = [term for term in case.must if term.lower() not in lower]
    assert not missing, f"{case.id} missing {missing}; answer={answer!r}"
    assert "not enough information" not in lower, f"{case.id} returned no-info: {answer!r}"
    assert retrieval.requests, f"{case.id} never called search_hybrid"
    assert final.evidence_items, f"{case.id} produced no evidence"
    if case.requires_table:
        assert "|" in answer or "table" in lower, f"{case.id} did not include a comparison table: {answer!r}"


@pytest.mark.asyncio
async def test_twenty_paper_questions_answer_with_deep_qa_disabled() -> None:
    for case in _load_cases():
        answer, final, retrieval = await _run_case(case, deep_qa_mode=False)
        _assert_case_answer(case, answer, final, retrieval)


@pytest.mark.asyncio
async def test_twenty_paper_questions_answer_with_deep_qa_enabled() -> None:
    for case in _load_cases():
        answer, final, retrieval = await _run_case(case, deep_qa_mode=True)
        _assert_case_answer(case, answer, final, retrieval)
