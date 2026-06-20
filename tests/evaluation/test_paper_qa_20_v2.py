"""Immutable second 20-question LightRAG paper QA benchmark.

The fixture questions/expected terms are intentionally fixed. If this test
fails, improve retrieval, table evidence, planning, or answer synthesis; do not
edit the fixture to make the test easier.

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

_FIXTURE = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "paper_qa_2410_05779v3_v2.json"
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
    requires_table: bool = False
    history: list[HistoryTurnSpec] = Field(default_factory=list)


class CorpusChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: str
    content: str
    embedding_content: str | None = None


class Fixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper: str
    description: str
    chunks: list[CorpusChunk]
    cases: list[PaperQACase]


class _CorpusRetrieval:
    """Deterministic retrieval fake over one shared paper corpus."""

    def __init__(self, chunks: list[CorpusChunk]) -> None:
        self.chunks = chunks
        self.requests: list[RetrievalRequest] = []

    async def search(self, request: RetrievalRequest) -> RetrievalResponse:
        self.requests.append(request)
        query_terms = _terms(request.query)
        scored: list[tuple[int, int, CorpusChunk]] = []
        for idx, chunk in enumerate(self.chunks):
            haystack = f"{chunk.content}\n{chunk.embedding_content or ''}"
            score = len(query_terms & _terms(haystack))
            lower = haystack.lower()
            if {"table", "comparison", "rates", "rows"} & query_terms and "<table" in lower:
                score += 5
            if {"figure", "architecture", "cost"} & query_terms and chunk.source_type == "figure_caption":
                score += 4
            if {"ablation", "ablated", "-high", "-low", "-origin"} & query_terms and "table 2" in lower:
                score += 6
            if {"graphrag", "lightrag", "compare", "contrast"} & query_terms and "graphrag" in lower and "lightrag" in lower:
                score += 4
            scored.append((score, -idx, chunk))

        ranked = sorted(scored, reverse=True)
        limit = min(max(request.final_top_k, 8), len(ranked))
        hits: list[SearchHit] = []
        for rank, (score, neg_idx, chunk) in enumerate(ranked[:limit], start=1):
            idx = -neg_idx
            hits.append(
                SearchHit(
                    vespa_document_id=f"id::document_chunk::paperqa-v2-{idx}",
                    chat_id=str(request.chat_id),
                    document_id=str(_DOC_ID),
                    source_node_id=f"paperqa-v2-node-{idx}",
                    source_type=chunk.source_type,
                    content=chunk.content,
                    page_start=1 + idx,
                    page_end=1 + idx,
                    order_index=idx,
                    vector_score=0.85 if score > 0 else 0.1,
                    fusion_score=float(score),
                    final_score=float(score),
                    final_rank=rank,
                )
            )
        return RetrievalResponse(hits=hits)


@asynccontextmanager
async def _null_session() -> AsyncIterator[Any]:
    yield MagicMock()


def _terms(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}|[0-9]+(?:\.[0-9]+)?%?", text)}


def _load_fixture() -> Fixture:
    return Fixture.model_validate(json.loads(_FIXTURE.read_text(encoding="utf-8")))


async def _run_case(case: PaperQACase, corpus: list[CorpusChunk], *, deep_qa_mode: bool) -> tuple[str, AgentState, _CorpusRetrieval]:
    retrieval = _CorpusRetrieval(corpus)
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
        conversation_history=[ConversationTurn(role=t.role, content=t.content) for t in case.history],
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


def _assert_case_answer(case: PaperQACase, answer: str, final: AgentState, retrieval: _CorpusRetrieval) -> None:
    lower = answer.lower()
    missing = [term for term in case.must if term.lower() not in lower]
    assert not missing, f"{case.id} missing {missing}; answer={answer!r}"
    assert "not enough information" not in lower, f"{case.id} returned no-info: {answer!r}"
    assert retrieval.requests, f"{case.id} never called search_hybrid"
    assert final.evidence_items, f"{case.id} produced no evidence"
    if case.requires_table:
        assert "<table" in answer.lower(), f"{case.id} did not use HTML table evidence: {answer!r}"


@pytest.mark.asyncio
async def test_immutable_twenty_paper_questions_answer_with_deep_qa_disabled() -> None:
    fixture = _load_fixture()
    for case in fixture.cases:
        answer, final, retrieval = await _run_case(case, fixture.chunks, deep_qa_mode=False)
        _assert_case_answer(case, answer, final, retrieval)


@pytest.mark.asyncio
async def test_immutable_twenty_paper_questions_answer_with_deep_qa_enabled() -> None:
    fixture = _load_fixture()
    for case in fixture.cases:
        answer, final, retrieval = await _run_case(case, fixture.chunks, deep_qa_mode=True)
        _assert_case_answer(case, answer, final, retrieval)
