"""Tests for Phase 6.5 retrieval evaluation harness — trimmed to ≤10 items.

Mandatory gates retained
------------------------
- cross_chat_leakage == 0  (leakage=0 assertion never relaxed)
- hybrid recall@10 >= bm25_only recall@10
- hybrid recall@10 >= vector_only recall@10
- native rerank no-regression gate (evaluate_gates passes)
- write_report produces valid JSON (round-trips via EvalReport.model_validate)
  + MD with mandatory headers

Metric unit tests (recall_at_k, MRR, NDCG, leakage) are merged into a
single parametrized test to reduce count while keeping all edge-case coverage.

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
# Keyword-overlap reranker for tests
# ---------------------------------------------------------------------------


class _KeywordRerankerProvider(RerankerProvider):
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
# Helpers
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
# FakeVespa transport
# ---------------------------------------------------------------------------


class _FakeVespaTransport(httpx.AsyncBaseTransport):
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

        if query_text:
            self._last_query_text = query_text
        elif self._last_query_text:
            query_text = self._last_query_text

        chat_id_filter = self._extract_chat_id_filter(yql)

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
            if ranking_profile in ("semantic_only", "hybrid_with_native_rerank"):
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
    reranker: RerankerProvider | None = _KeywordRerankerProvider() if with_reranker else None
    return RetrievalService(
        endpoint="http://fake-vespa:8080",
        embedding_provider=MockEmbeddingProvider(dimension=embedding_dim),
        reranker_provider=reranker,
        embedding_dim=embedding_dim,
        http_transport=fake_transport,
    )


# ===========================================================================
# 1. Parametrized metric unit tests (recall_at_k, MRR, NDCG, leakage)
# ===========================================================================


def test_metric_helpers_combined() -> None:
    """Covers recall_at_k, MRR, NDCG, and cross_chat_leakage edge cases in one test."""
    # recall_at_k: perfect, zero, empty relevant, empty retrieved
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)
    assert recall_at_k(["b", "c"], {"a"}, k=1) == pytest.approx(0.0)
    assert recall_at_k(["a", "b"], set(), k=5) == pytest.approx(1.0)
    assert recall_at_k([], {"a", "b"}, k=5) == pytest.approx(0.0)

    # MRR: first hit, second hit, no hit, empty retrieved
    assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == pytest.approx(1.0)
    assert mean_reciprocal_rank(["x", "a", "c"], {"a"}) == pytest.approx(0.5)
    assert mean_reciprocal_rank(["x", "y"], {"a"}) == pytest.approx(0.0)
    assert mean_reciprocal_rank([], {"a"}) == pytest.approx(0.0)

    # NDCG: perfect, all irrelevant, empty relevant
    assert ndcg_at_k(["a"], {"a"}, k=1) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y", "z"], {"a", "b"}, k=3) == pytest.approx(0.0)
    assert ndcg_at_k(["a", "b"], set(), k=5) == pytest.approx(1.0)

    # cross_chat_leakage: no leak, one leak, empty
    no_leak = [_make_hit("doc-1", CHAT_A), _make_hit("doc-2", CHAT_A)]
    assert cross_chat_leakage(no_leak, CHAT_A) == 0
    one_leak = [_make_hit("doc-1", CHAT_A), _make_hit("doc-2", CHAT_B)]
    assert cross_chat_leakage(one_leak, CHAT_A) == 1
    assert cross_chat_leakage([], CHAT_A) == 0


# ===========================================================================
# 2. MANDATORY GATE: leakage == 0 across all modes (FakeVespa e2e)
# ===========================================================================


@pytest.mark.asyncio
async def test_mandatory_gate_leakage_zero() -> None:
    """MANDATORY: no cross-chat leakage in any mode."""
    service = _build_fake_service()
    cases = _load_cases()
    report = await evaluate(service, cases)

    assert report.leakage_total == 0, (
        f"Expected 0 cross-chat leakage, got {report.leakage_total}"
    )
    for mode, metrics in report.per_mode.items():
        assert metrics.leakage_count == 0, (
            f"Mode {mode} has leakage={metrics.leakage_count}"
        )


# ===========================================================================
# 3. MANDATORY GATE: hybrid recall@10 >= bm25_only recall@10
# ===========================================================================


@pytest.mark.asyncio
async def test_mandatory_gate_hybrid_ge_bm25() -> None:
    """MANDATORY: hybrid recall@10 must be >= bm25_only recall@10."""
    service = _build_fake_service()
    cases = _load_cases()
    report = await evaluate(service, cases)

    hybrid_r10 = report.per_mode["hybrid"].recall_at_10
    bm25_r10 = report.per_mode["bm25_only"].recall_at_10
    assert hybrid_r10 >= bm25_r10, (
        f"hybrid recall@10={hybrid_r10:.4f} < bm25 recall@10={bm25_r10:.4f}"
    )


# ===========================================================================
# 4. MANDATORY GATE: hybrid recall@10 >= vector_only recall@10
# ===========================================================================


@pytest.mark.asyncio
async def test_mandatory_gate_hybrid_ge_vector() -> None:
    """MANDATORY: hybrid recall@10 must be >= vector_only recall@10."""
    service = _build_fake_service()
    cases = _load_cases()
    report = await evaluate(service, cases)

    hybrid_r10 = report.per_mode["hybrid"].recall_at_10
    vec_r10 = report.per_mode["vector_only"].recall_at_10
    assert hybrid_r10 >= vec_r10, (
        f"hybrid recall@10={hybrid_r10:.4f} < vector recall@10={vec_r10:.4f}"
    )


# ===========================================================================
# 5. MANDATORY GATE: evaluate_gates passes + rerank no-regression
# ===========================================================================


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


def test_evaluate_gates_pass_and_failures() -> None:
    """MANDATORY: evaluate_gates all-pass; fails on leakage, hybrid<bm25, rerank regression."""
    # All-pass
    report_ok = _make_report_with_modes()
    assert evaluate_gates(report_ok).passed

    # Leakage > 0 fails gate_leakage_zero
    report_leak = _make_report_with_modes(hybrid=_make_mode_metrics("hybrid", 0.80, leakage=1))
    report_leak.leakage_total = 1
    gates_leak = evaluate_gates(report_leak)
    assert not gates_leak.passed
    assert not gates_leak.gate_leakage_zero

    # hybrid recall < bm25 fails gate_hybrid_ge_bm25
    report_bm25 = _make_report_with_modes(
        hybrid=_make_mode_metrics("hybrid", 0.60),
        bm25_only=_make_mode_metrics("bm25_only", 0.80),
    )
    gates_bm25 = evaluate_gates(report_bm25)
    assert not gates_bm25.gate_hybrid_ge_bm25
    assert not gates_bm25.passed

    # native rerank regression fails gate_native_rerank_no_regression
    report_rr = _make_report_with_modes(
        hybrid=_make_mode_metrics("hybrid", 0.80),
        hybrid_native_rerank=_make_mode_metrics("hybrid_native_rerank", 0.70),
    )
    gates_rr = evaluate_gates(report_rr)
    assert not gates_rr.gate_native_rerank_no_regression
    assert not gates_rr.passed


# ===========================================================================
# 6. write_report — creates valid JSON + MD with mandatory headers
# ===========================================================================


@pytest.mark.asyncio
async def test_write_report_valid_json_and_md_headers(tmp_path: Path) -> None:
    """write_report writes JSON that round-trips via EvalReport + MD with gate headers."""
    service = _build_fake_service()
    cases = _load_cases()[:2]
    report = await evaluate(service, cases)

    json_path = tmp_path / "retrieval-report.json"
    md_path = tmp_path / "retrieval-report.md"
    write_report(report, json_path, md_path)

    assert json_path.exists()
    assert md_path.exists()

    # JSON round-trip
    restored = EvalReport.model_validate(
        json.loads(json_path.read_text(encoding="utf-8"))
    )
    assert restored.leakage_total == report.leakage_total
    assert set(restored.per_mode.keys()) == set(report.per_mode.keys())

    # MD mandatory headers
    md_content = md_path.read_text(encoding="utf-8")
    assert "| Mode |" in md_content
    assert "leakage" in md_content.lower()
    assert "Mandatory Gate Results" in md_content
    assert "PASS" in md_content
