"""Tests for Phase 6.5 retrieval evaluation harness.

Coverage
--------
Metric unit tests (12+ known-answer cases):
  - recall_at_k: empty relevant, empty retrieved, exact match, partial match
  - mean_reciprocal_rank: first hit, second hit, no hit, empty relevant
  - ndcg_at_k: perfect ranking, reversed ranking, partial match, empty sets
  - cross_chat_leakage: all correct, one leak, two leaks, empty hits

EvalCase and model validation:
  - EvalCase round-trips through Pydantic
  - EvalReport round-trips through JSON → model_validate

Integration (FakeVespa e2e):
  - All 5 modes run without error
  - hybrid recall >= bm25_only recall (recall@10)
  - hybrid recall >= vector_only recall (recall@10)
  - cross-chat leakage == 0
  - evaluate_gates() returns passed=True on a well-formed report

Report writing:
  - write_report() creates json + md files
  - JSON can be parsed back via EvalReport.model_validate
  - MD file contains expected table headers

All tests are fully deterministic; no real Vespa/OpenAI calls.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest

from app.evaluation.retrieval_eval import (
    EvalCase,
    EvalReport,
    ModeMetrics,
    cross_chat_leakage,
    evaluate,
    evaluate_gates,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    write_report,
)
from app.providers.base import ProviderTestResult, RerankerProvider
from app.providers.mock import MockEmbeddingProvider
from app.retrieval.models import SearchHit
from app.retrieval.service import RetrievalService

# ---------------------------------------------------------------------------
# Keyword-overlap reranker for tests (semantically consistent with FakeVespa)
# ---------------------------------------------------------------------------


class _KeywordRerankerProvider(RerankerProvider):
    """Reranker that scores docs by keyword overlap with the query.

    This produces scores consistent with the FakeVespa BM25 scoring so that
    cross-encoder rerank does not regress significantly vs. hybrid in tests.
    """

    async def score(self, query: str, docs: list[str]) -> list[float]:
        query_tokens = set(query.lower().split())
        scores: list[float] = []
        for doc in docs:
            doc_tokens = set(doc.lower().split())
            overlap = len(query_tokens & doc_tokens)
            scores.append(overlap / max(len(query_tokens), 1))
        return scores

    async def test_connection(self) -> ProviderTestResult:
        return ProviderTestResult(ok=True, model="keyword-reranker", latency_ms=0)

# ---------------------------------------------------------------------------
# Helpers: build fake SearchHits and services
# ---------------------------------------------------------------------------

CHAT_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
CHAT_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

_CORPUS_PATH = (
    Path(__file__).parent.parent.parent / "data" / "fixtures" / "retrieval" / "corpus.json"
)
_CASES_PATH = (
    Path(__file__).parent.parent.parent / "data" / "fixtures" / "retrieval" / "eval_cases.json"
)


def _make_hit(
    vespa_document_id: str,
    chat_id: UUID = CHAT_A,
    content: str = "test content",
) -> SearchHit:
    return SearchHit(
        vespa_document_id=vespa_document_id,
        chat_id=str(chat_id),
        document_id="dddddddd-dddd-dddd-dddd-dddddddddd01",
        source_node_id="node-test",
        source_type="chunk",
        content=content,
        page_start=1,
        page_end=1,
        order_index=0,
    )


def _load_corpus() -> list[dict[str, Any]]:
    data: dict[str, Any] = json.loads(_CORPUS_PATH.read_text(encoding="utf-8"))
    return data["chunks"]


def _load_cases() -> list[EvalCase]:
    data: dict[str, Any] = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    result = []
    for raw in data["cases"]:
        raw_copy = {k: v for k, v in raw.items() if not k.startswith("_")}
        result.append(EvalCase(**raw_copy))
    return result


# ---------------------------------------------------------------------------
# FakeVespa transport (same as scripts/run_retrieval_eval.py)
# ---------------------------------------------------------------------------


class _FakeVespaTransport(httpx.AsyncBaseTransport):
    """httpx AsyncBaseTransport that simulates Vespa search (no real HTTP).

    Enforces chat_id isolation: chunks from other chats are excluded from
    results based on the YQL ``chat_id contains`` filter.

    Stores the last non-empty query text so native rerank (which doesn't
    pass a query string) can still score docs meaningfully.
    """

    def __init__(self, corpus: list[dict[str, Any]]) -> None:
        self._corpus = corpus
        self._last_query_text: str = ""

    def _extract_chat_id_filter(self, yql: str) -> str | None:
        m = re.search(r'chat_id contains "([^"]+)"', yql)
        return m.group(1) if m else None

    def _score_bm25(self, query: str, chunk: dict[str, Any]) -> float:
        query_tokens = set(query.lower().split())
        content = str(chunk.get("content", "")).lower()
        keywords = [k.lower() for k in chunk.get("keywords", [])]
        all_text = content + " " + " ".join(keywords)
        text_tokens = set(all_text.split())
        overlap = len(query_tokens & text_tokens)
        return overlap / max(len(query_tokens), 1)

    def _score_vector(self, query: str, chunk: dict[str, Any]) -> float:
        query_tokens = set(query.lower().split())
        tech_keywords = [k.lower() for k in chunk.get("technical_keywords", [])]
        title = str(chunk.get("title", "")).lower()
        alt_text = title + " " + " ".join(tech_keywords)
        alt_tokens = set(alt_text.split())
        overlap = len(query_tokens & alt_tokens)
        chunk_id = str(chunk.get("vespa_document_id", ""))
        hash_val = int(hashlib.sha256(f"{query}|{chunk_id}".encode()).hexdigest()[:4], 16)
        jitter = (hash_val % 100) / 10000.0
        return overlap / max(len(query_tokens), 1) + jitter

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body_bytes = await request.aread()
        body: dict[str, Any] = json.loads(body_bytes.decode())

        yql: str = str(body.get("yql", ""))
        query_text: str = str(body.get("query", ""))
        ranking_profile: str = str(body.get("ranking.profile", "bm25_only"))
        hits_limit: int = int(body.get("hits", 10))

        # Cache non-empty query text for use by native rerank (which sends no query)
        if query_text:
            self._last_query_text = query_text
        elif self._last_query_text:
            query_text = self._last_query_text

        chat_id_filter = self._extract_chat_id_filter(yql)

        # Extract doc_id whitelist for native rerank queries
        doc_id_whitelist: set[str] | None = None
        doc_in_match = re.search(r"vespa_document_id in \(([^)]+)\)", yql)
        if doc_in_match:
            raw_ids = doc_in_match.group(1)
            doc_id_whitelist = {
                m.group(1) for m in re.finditer(r'"([^"]+)"', raw_ids)
            }

        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk in self._corpus:
            if chat_id_filter and str(chunk.get("chat_id", "")) != chat_id_filter:
                continue
            vid = str(chunk.get("vespa_document_id", ""))
            if doc_id_whitelist is not None and vid not in doc_id_whitelist:
                continue
            if ranking_profile == "semantic_only":
                score = self._score_vector(query_text, chunk)
            elif ranking_profile == "hybrid_with_native_rerank":
                score = self._score_vector(query_text, chunk)
            else:
                score = self._score_bm25(query_text, chunk)
            scored.append((score, chunk))

        seen_ids: set[str] = set()
        unique_scored: list[tuple[float, dict[str, Any]]] = []
        for score, chunk in sorted(scored, key=lambda t: t[0], reverse=True):
            vid = str(chunk.get("vespa_document_id", ""))
            if vid not in seen_ids:
                seen_ids.add(vid)
                unique_scored.append((score, chunk))

        top = unique_scored[:hits_limit]
        children: list[dict[str, Any]] = []
        for score, chunk in top:
            fields = dict(chunk)
            children.append(
                {
                    "id": f"id:default:document_chunk::{fields.get('vespa_document_id', '')}",
                    "relevance": score,
                    "fields": fields,
                }
            )

        response_body = json.dumps({"root": {"children": children}}).encode()
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=response_body,
            request=request,
        )


def _build_fake_service(
    corpus: list[dict[str, Any]] | None = None,
    embedding_dim: int = 32,
    with_reranker: bool = True,
) -> RetrievalService:
    if corpus is None:
        corpus = _load_corpus()

    fake_transport = _FakeVespaTransport(corpus)
    # Use keyword-overlap reranker so cross-encoder scores are consistent
    # with the FakeVespa BM25 scoring (avoids arbitrary hash-based reranking).
    reranker: RerankerProvider | None = _KeywordRerankerProvider() if with_reranker else None
    return RetrievalService(
        endpoint="http://fake-vespa:8080",
        embedding_provider=MockEmbeddingProvider(dimension=embedding_dim),
        reranker_provider=reranker,
        embedding_dim=embedding_dim,
        http_transport=fake_transport,
    )


# ---------------------------------------------------------------------------
# Section 1: recall_at_k unit tests
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)

    def test_recall_at_1_first_hit(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a"}, k=1) == pytest.approx(1.0)

    def test_recall_at_1_miss(self) -> None:
        assert recall_at_k(["b", "c"], {"a"}, k=1) == pytest.approx(0.0)

    def test_recall_partial(self) -> None:
        # 1 of 2 relevant found in top-3
        result = recall_at_k(["a", "x", "y"], {"a", "z"}, k=3)
        assert result == pytest.approx(0.5)

    def test_empty_relevant_returns_1(self) -> None:
        assert recall_at_k(["a", "b"], set(), k=5) == pytest.approx(1.0)

    def test_empty_retrieved_nonzero_relevant_returns_0(self) -> None:
        assert recall_at_k([], {"a", "b"}, k=5) == pytest.approx(0.0)

    def test_k_larger_than_retrieved(self) -> None:
        # k > len(retrieved) — uses all retrieved
        assert recall_at_k(["a"], {"a", "b"}, k=100) == pytest.approx(0.5)

    def test_no_overlap(self) -> None:
        assert recall_at_k(["x", "y"], {"a", "b"}, k=5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Section 2: mean_reciprocal_rank unit tests
# ---------------------------------------------------------------------------


class TestMRR:
    def test_first_hit(self) -> None:
        assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_second_hit(self) -> None:
        assert mean_reciprocal_rank(["x", "a", "c"], {"a"}) == pytest.approx(0.5)

    def test_third_hit(self) -> None:
        assert mean_reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_hit(self) -> None:
        assert mean_reciprocal_rank(["x", "y"], {"a"}) == pytest.approx(0.0)

    def test_empty_relevant(self) -> None:
        assert mean_reciprocal_rank(["a", "b"], set()) == pytest.approx(1.0)

    def test_empty_retrieved(self) -> None:
        assert mean_reciprocal_rank([], {"a"}) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Section 3: ndcg_at_k unit tests
# ---------------------------------------------------------------------------


class TestNDCG:
    def test_perfect_ranking(self) -> None:
        # Only relevant doc at rank 1
        result = ndcg_at_k(["a"], {"a"}, k=1)
        assert result == pytest.approx(1.0)

    def test_all_irrelevant(self) -> None:
        result = ndcg_at_k(["x", "y", "z"], {"a", "b"}, k=3)
        assert result == pytest.approx(0.0)

    def test_empty_relevant(self) -> None:
        assert ndcg_at_k(["a", "b"], set(), k=5) == pytest.approx(1.0)

    def test_empty_retrieved(self) -> None:
        assert ndcg_at_k([], {"a"}, k=5) == pytest.approx(0.0)

    def test_partial_ndcg(self) -> None:
        # 1 relevant at rank 2, 1 at rank 4
        result = ndcg_at_k(["x", "a", "y", "b"], {"a", "b"}, k=5)
        assert 0.0 < result < 1.0

    def test_k_zero_relevant_returned(self) -> None:
        # k=10 but only 2 relevant exist, first 2 are relevant → perfect
        result = ndcg_at_k(["a", "b", "x", "y"], {"a", "b"}, k=10)
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Section 4: cross_chat_leakage unit tests
# ---------------------------------------------------------------------------


class TestCrossChat:
    def test_no_leakage(self) -> None:
        hits = [_make_hit("doc-1", CHAT_A), _make_hit("doc-2", CHAT_A)]
        assert cross_chat_leakage(hits, CHAT_A) == 0

    def test_one_leakage(self) -> None:
        hits = [_make_hit("doc-1", CHAT_A), _make_hit("doc-2", CHAT_B)]
        assert cross_chat_leakage(hits, CHAT_A) == 1

    def test_two_leakages(self) -> None:
        hits = [
            _make_hit("doc-1", CHAT_B),
            _make_hit("doc-2", CHAT_B),
            _make_hit("doc-3", CHAT_A),
        ]
        assert cross_chat_leakage(hits, CHAT_A) == 2

    def test_empty_hits(self) -> None:
        assert cross_chat_leakage([], CHAT_A) == 0

    def test_all_leaked(self) -> None:
        hits = [_make_hit("doc-1", CHAT_B), _make_hit("doc-2", CHAT_B)]
        assert cross_chat_leakage(hits, CHAT_A) == 2


# ---------------------------------------------------------------------------
# Section 5: EvalCase model validation
# ---------------------------------------------------------------------------


class TestEvalCase:
    def test_round_trip(self) -> None:
        case = EvalCase(
            query="test query",
            chat_id=CHAT_A,
            relevant_vespa_document_ids=["doc-1", "doc-2"],
            notes="test note",
        )
        dumped = case.model_dump()
        restored = EvalCase(**dumped)
        assert restored.query == case.query
        assert restored.relevant_vespa_document_ids == case.relevant_vespa_document_ids

    def test_no_document_ids_optional(self) -> None:
        case = EvalCase(
            query="q",
            chat_id=CHAT_A,
            relevant_vespa_document_ids=["a"],
        )
        assert case.document_ids is None

    def test_extra_fields_forbidden(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            EvalCase(
                query="q",
                chat_id=CHAT_A,
                relevant_vespa_document_ids=[],
                unknown_field="oops",  # type: ignore[call-arg]
            )


# ---------------------------------------------------------------------------
# Section 6: Integration — full evaluate() run with FakeVespa
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEvaluateIntegration:
    async def test_evaluate_runs_all_modes(self) -> None:
        service = _build_fake_service()
        cases = _load_cases()
        report = await evaluate(service, cases)

        # All 5 modes should be present
        assert set(report.per_mode.keys()) == {
            "bm25_only",
            "vector_only",
            "hybrid",
            "hybrid_native_rerank",
            "hybrid_cross_encoder",
        }

    async def test_hybrid_recall_ge_bm25(self) -> None:
        service = _build_fake_service()
        cases = _load_cases()
        report = await evaluate(service, cases)

        hybrid_r10 = report.per_mode["hybrid"].recall_at_10
        bm25_r10 = report.per_mode["bm25_only"].recall_at_10
        assert hybrid_r10 >= bm25_r10, (
            f"hybrid recall@10={hybrid_r10:.4f} < bm25 recall@10={bm25_r10:.4f}"
        )

    async def test_hybrid_recall_ge_vector(self) -> None:
        service = _build_fake_service()
        cases = _load_cases()
        report = await evaluate(service, cases)

        hybrid_r10 = report.per_mode["hybrid"].recall_at_10
        vec_r10 = report.per_mode["vector_only"].recall_at_10
        assert hybrid_r10 >= vec_r10, (
            f"hybrid recall@10={hybrid_r10:.4f} < vector recall@10={vec_r10:.4f}"
        )

    async def test_leakage_is_zero(self) -> None:
        service = _build_fake_service()
        cases = _load_cases()
        report = await evaluate(service, cases)

        assert report.leakage_total == 0, (
            f"Expected 0 cross-chat leakage, got {report.leakage_total}"
        )

    async def test_per_mode_leakage_zero(self) -> None:
        service = _build_fake_service()
        cases = _load_cases()
        report = await evaluate(service, cases)

        for mode, metrics in report.per_mode.items():
            assert metrics.leakage_count == 0, (
                f"Mode {mode} has leakage={metrics.leakage_count}"
            )

    async def test_evaluate_gates_pass(self) -> None:
        service = _build_fake_service()
        cases = _load_cases()
        report = await evaluate(service, cases)
        gates = evaluate_gates(report)

        assert gates.passed, (
            f"Gates failed: {gates.failure_reasons}"
        )

    async def test_mrr_in_valid_range(self) -> None:
        service = _build_fake_service()
        cases = _load_cases()
        report = await evaluate(service, cases)

        for mode, metrics in report.per_mode.items():
            assert 0.0 <= metrics.mrr <= 1.0, f"{mode} mrr={metrics.mrr} out of [0,1]"

    async def test_recall_in_valid_range(self) -> None:
        service = _build_fake_service()
        cases = _load_cases()
        report = await evaluate(service, cases)

        for _mode, metrics in report.per_mode.items():
            assert 0.0 <= metrics.recall_at_10 <= 1.0

    async def test_embedding_dim_recorded(self) -> None:
        service = _build_fake_service(embedding_dim=32)
        cases = _load_cases()
        report = await evaluate(service, cases)
        assert report.embedding_dim == 32

    async def test_per_case_count_matches(self) -> None:
        service = _build_fake_service()
        cases = _load_cases()
        report = await evaluate(service, cases)
        assert len(report.per_case) == len(cases)


# ---------------------------------------------------------------------------
# Section 7: evaluate_gates() unit tests on synthetic reports
# ---------------------------------------------------------------------------


def _make_mode_metrics(
    mode: str,
    recall_at_10: float = 0.8,
    mrr: float = 0.8,
    leakage: int = 0,
) -> ModeMetrics:
    return ModeMetrics(
        mode=mode,
        recall_at_1=recall_at_10 * 0.5,
        recall_at_5=recall_at_10 * 0.8,
        recall_at_10=recall_at_10,
        mrr=mrr,
        ndcg_at_10=recall_at_10 * 0.9,
        leakage_count=leakage,
    )


def _make_report_with_modes(**overrides: ModeMetrics) -> EvalReport:
    defaults: dict[str, ModeMetrics] = {
        "bm25_only": _make_mode_metrics("bm25_only", 0.7),
        "vector_only": _make_mode_metrics("vector_only", 0.75),
        "hybrid": _make_mode_metrics("hybrid", 0.80),
        "hybrid_native_rerank": _make_mode_metrics("hybrid_native_rerank", 0.80),
        "hybrid_cross_encoder": _make_mode_metrics("hybrid_cross_encoder", 0.80),
    }
    defaults.update(overrides)
    return EvalReport(per_mode=defaults)


class TestEvaluateGates:
    def test_all_pass(self) -> None:
        report = _make_report_with_modes()
        gates = evaluate_gates(report)
        assert gates.passed

    def test_leakage_fails(self) -> None:
        report = _make_report_with_modes(
            hybrid=_make_mode_metrics("hybrid", 0.80, leakage=1),
        )
        report.leakage_total = 1
        gates = evaluate_gates(report)
        assert not gates.passed
        assert not gates.gate_leakage_zero

    def test_hybrid_less_than_bm25_fails(self) -> None:
        report = _make_report_with_modes(
            hybrid=_make_mode_metrics("hybrid", 0.60),
            bm25_only=_make_mode_metrics("bm25_only", 0.80),
        )
        gates = evaluate_gates(report)
        assert not gates.gate_hybrid_ge_bm25
        assert not gates.passed

    def test_hybrid_less_than_vector_fails(self) -> None:
        report = _make_report_with_modes(
            hybrid=_make_mode_metrics("hybrid", 0.60),
            vector_only=_make_mode_metrics("vector_only", 0.80),
        )
        gates = evaluate_gates(report)
        assert not gates.gate_hybrid_ge_vector
        assert not gates.passed

    def test_native_rerank_regression_fails(self) -> None:
        report = _make_report_with_modes(
            hybrid=_make_mode_metrics("hybrid", 0.80),
            hybrid_native_rerank=_make_mode_metrics("hybrid_native_rerank", 0.70),
        )
        gates = evaluate_gates(report)
        assert not gates.gate_native_rerank_no_regression
        assert not gates.passed

    def test_native_rerank_within_tolerance_passes(self) -> None:
        # Exactly at tolerance: 0.80 - 0.02 = 0.78 — should pass
        report = _make_report_with_modes(
            hybrid=_make_mode_metrics("hybrid", 0.80, 0.80),
            hybrid_native_rerank=_make_mode_metrics("hybrid_native_rerank", 0.78, 0.78),
        )
        gates = evaluate_gates(report)
        assert gates.gate_native_rerank_no_regression
        assert gates.passed


# ---------------------------------------------------------------------------
# Section 8: write_report() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWriteReport:
    async def test_json_file_created(self, tmp_path: Path) -> None:
        service = _build_fake_service()
        cases = _load_cases()[:2]  # use subset for speed
        report = await evaluate(service, cases)

        json_path = tmp_path / "retrieval-report.json"
        md_path = tmp_path / "retrieval-report.md"
        write_report(report, json_path, md_path)

        assert json_path.exists()
        assert md_path.exists()

    async def test_json_roundtrip(self, tmp_path: Path) -> None:
        service = _build_fake_service()
        cases = _load_cases()[:2]
        report = await evaluate(service, cases)

        json_path = tmp_path / "retrieval-report.json"
        md_path = tmp_path / "retrieval-report.md"
        write_report(report, json_path, md_path)

        raw_json = json.loads(json_path.read_text(encoding="utf-8"))
        restored = EvalReport.model_validate(raw_json)
        assert restored.leakage_total == report.leakage_total
        assert set(restored.per_mode.keys()) == set(report.per_mode.keys())

    async def test_md_contains_table_headers(self, tmp_path: Path) -> None:
        service = _build_fake_service()
        cases = _load_cases()[:2]
        report = await evaluate(service, cases)

        json_path = tmp_path / "retrieval-report.json"
        md_path = tmp_path / "retrieval-report.md"
        write_report(report, json_path, md_path)

        md_content = md_path.read_text(encoding="utf-8")
        assert "| Mode |" in md_content
        assert "Recall@1" in md_content or "recall@1" in md_content.lower()
        assert "leakage" in md_content.lower()
        assert "Mandatory Gate Results" in md_content

    async def test_md_shows_gate_pass(self, tmp_path: Path) -> None:
        service = _build_fake_service()
        cases = _load_cases()
        report = await evaluate(service, cases)

        json_path = tmp_path / "retrieval-report.json"
        md_path = tmp_path / "retrieval-report.md"
        write_report(report, json_path, md_path)

        md_content = md_path.read_text(encoding="utf-8")
        # Should indicate PASS overall
        assert "PASS" in md_content

    async def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        service = _build_fake_service()
        cases = _load_cases()[:1]
        report = await evaluate(service, cases)

        nested = tmp_path / "a" / "b" / "c"
        write_report(report, nested / "report.json", nested / "report.md")
        assert (nested / "report.json").exists()
