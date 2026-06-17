#!/usr/bin/env python
"""CLI script for Phase 6.5 retrieval evaluation.

Usage (mock mode — default, no real Vespa needed):
    uv run python scripts/run_retrieval_eval.py --output-dir artifacts/evaluation/

Usage (real Vespa — requires running Vespa at localhost:8080):
    uv run python scripts/run_retrieval_eval.py --real-vespa --output-dir artifacts/evaluation/

Environment variables:
    RETRIEVAL_EVAL_REAL_VESPA=1   equivalent to --real-vespa flag

Exit code:
    0  all mandatory gates passed
    1  one or more mandatory gates failed
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Ensure repo root is in sys.path so we can import app.*
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from app.evaluation.retrieval_eval import (  # noqa: E402
    EvalCase,
    evaluate,
    evaluate_gates,
    write_report,
)
from app.providers.base import ProviderTestResult, RerankerProvider  # noqa: E402
from app.providers.mock import MockEmbeddingProvider, MockRerankerProvider  # noqa: E402
from app.retrieval.service import RetrievalService  # noqa: E402

# ---------------------------------------------------------------------------
# Keyword-overlap reranker for mock mode (consistent with FakeVespa BM25)
# ---------------------------------------------------------------------------


class _KeywordRerankerProvider(RerankerProvider):
    """Reranker that scores documents by keyword overlap with the query.

    Used in mock mode so that cross-encoder scores are semantically consistent
    with FakeVespa BM25 scoring.  Avoids arbitrary hash-based rankings.
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
# Default fixture paths
# ---------------------------------------------------------------------------

_FIXTURES_DIR = _REPO_ROOT / "data" / "fixtures" / "retrieval"
_DEFAULT_CASES = _FIXTURES_DIR / "eval_cases.json"
_DEFAULT_CORPUS = _FIXTURES_DIR / "corpus.json"

# ---------------------------------------------------------------------------
# FakeVespa: in-memory search backend via httpx transport
# ---------------------------------------------------------------------------


class _FakeVespaTransport(httpx.AsyncBaseTransport):
    """httpx AsyncBaseTransport that simulates Vespa search (no real HTTP).

    All Vespa ``POST /search/`` requests are intercepted.  The fake backend:

    1. Parses the ``chat_id contains "<id>"`` filter from the ``yql`` field
       to enforce isolation — hits from other chats are excluded.
    2. Scores hits using keyword overlap between the query text and chunk
       ``content`` + ``keywords`` fields (BM25-like proxy).  For ANN queries
       (``ranking.profile == "semantic_only"``), scoring is based on a
       deterministic hash similarity so results differ from BM25 (simulating
       complementary retrieval).
    3. Returns a Vespa-shaped JSON response.

    This transport is passed to ``RetrievalService`` via ``http_transport=``
    so the full ``_post_search`` code path is exercised.

    Stores the last non-empty query text so that native rerank (which does not
    pass a ``"query"`` field in its POST body) can still produce meaningful
    per-document scores.
    """

    def __init__(self, corpus: list[dict[str, Any]]) -> None:
        self._corpus = corpus
        self._last_query_text: str = ""

    def _extract_chat_id_filter(self, yql: str) -> str | None:
        """Extract chat_id value from YQL WHERE clause."""
        m = re.search(r'chat_id contains "([^"]+)"', yql)
        return m.group(1) if m else None

    def _score_bm25(self, query: str, chunk: dict[str, Any]) -> float:
        """Simple keyword overlap score (BM25 proxy)."""
        query_tokens = set(query.lower().split())
        content = str(chunk.get("content", "")).lower()
        keywords = [k.lower() for k in chunk.get("keywords", [])]
        all_text = content + " " + " ".join(keywords)
        text_tokens = set(all_text.split())
        overlap = len(query_tokens & text_tokens)
        return overlap / max(len(query_tokens), 1)

    def _score_vector(self, query: str, chunk: dict[str, Any]) -> float:
        """Hash-based similarity score (deterministic ANN proxy).

        Uses a different scoring approach from BM25 to simulate complementary
        retrieval — focuses on technical_keywords and title rather than content.
        """
        query_tokens = set(query.lower().split())
        tech_keywords = [k.lower() for k in chunk.get("technical_keywords", [])]
        title = str(chunk.get("title", "")).lower()
        alt_text = title + " " + " ".join(tech_keywords)
        alt_tokens = set(alt_text.split())
        overlap = len(query_tokens & alt_tokens)
        chunk_id = str(chunk.get("vespa_document_id", ""))
        hash_val = int(hashlib.sha256(f"{query}|{chunk_id}".encode()).hexdigest()[:4], 16)
        jitter = (hash_val % 100) / 10000.0  # tiny jitter [0, 0.01)
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

        # Extract doc_id whitelist for native rerank (vespa_document_id in (...))
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
            # For native rerank: use combined bm25+vector score using the
            # query from the YQL context (native rerank doesn't send "query" text)
            # Use vector score (hash-based) as a stable approximation.
            if ranking_profile == "semantic_only":
                score = self._score_vector(query_text, chunk)
            elif ranking_profile == "hybrid_with_native_rerank":
                # Simulate second-phase rerank: use vector similarity as proxy
                # (chunk content tokens combined with hash jitter)
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
                    "id": (
                        f"id:default:document_chunk::{fields.get('vespa_document_id', '')}"
                    ),
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


def _build_service_with_fake_vespa(
    corpus: list[dict[str, Any]],
    embedding_dim: int = 32,
) -> RetrievalService:
    """Build a RetrievalService backed by the FakeVespa transport.

    Uses ``_KeywordRerankerProvider`` so cross-encoder rerank scores are
    semantically consistent with FakeVespa BM25 scoring.
    """
    fake_transport = _FakeVespaTransport(corpus)
    return RetrievalService(
        endpoint="http://fake-vespa:8080",
        embedding_provider=MockEmbeddingProvider(dimension=embedding_dim),
        reranker_provider=_KeywordRerankerProvider(),
        embedding_dim=embedding_dim,
        http_transport=fake_transport,
    )


def _load_cases(path: Path) -> list[EvalCase]:
    """Load EvalCase list from a JSON file."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = data.get("cases", [])
    result: list[EvalCase] = []
    for raw in raw_cases:
        raw_copy = {k: v for k, v in raw.items() if not k.startswith("_")}
        result.append(EvalCase(**raw_copy))
    return result


def _load_corpus(path: Path) -> list[dict[str, Any]]:
    """Load corpus chunks from a JSON file."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data.get("chunks", [])


async def _run(
    cases_path: Path,
    corpus_path: Path,
    output_dir: Path,
    real_vespa: bool,
    embedding_dim: int = 32,
) -> int:
    """Main async entrypoint. Returns exit code (0=pass, 1=fail)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_cases(cases_path)
    print(f"Loaded {len(cases)} eval cases from {cases_path}")

    if real_vespa:
        # Real Vespa mode: use production settings
        from app.config import get_settings

        settings = get_settings()
        service = RetrievalService(
            endpoint=settings.vespa_endpoint,
            embedding_provider=MockEmbeddingProvider(dimension=settings.embedding_dim),
            reranker_provider=MockRerankerProvider(),
            embedding_dim=settings.embedding_dim,
        )
        print(f"Using real Vespa at {settings.vespa_endpoint}")
    else:
        corpus = _load_corpus(corpus_path)
        print(f"Loaded {len(corpus)} corpus chunks from {corpus_path}")
        service = _build_service_with_fake_vespa(corpus, embedding_dim=embedding_dim)
        print("Using FakeVespa transport (mock mode)")

    print("Running evaluation...")
    report = await evaluate(service, cases)

    json_path = output_dir / "retrieval-report.json"
    md_path = output_dir / "retrieval-report.md"
    write_report(report, json_path, md_path)
    print(f"Report written to {json_path} and {md_path}")

    # Print summary
    print("\n=== Mode Summary ===")
    for mode, metrics in report.per_mode.items():
        print(
            f"  {mode:<30} recall@10={metrics.recall_at_10:.4f}  "
            f"mrr={metrics.mrr:.4f}  leakage={metrics.leakage_count}"
        )

    print(f"\nCross-chat leakage total: {report.leakage_total}")

    gates = evaluate_gates(report)
    print(f"\n=== Mandatory Gate Results: {'PASS' if gates.passed else 'FAIL'} ===")
    print(f"  leakage == 0:              {'PASS' if gates.gate_leakage_zero else 'FAIL'}")
    print(f"  hybrid >= bm25_only:       {'PASS' if gates.gate_hybrid_ge_bm25 else 'FAIL'}")
    print(f"  hybrid >= vector_only:     {'PASS' if gates.gate_hybrid_ge_vector else 'FAIL'}")
    print(
        f"  native_rerank no regress:  "
        f"{'PASS' if gates.gate_native_rerank_no_regression else 'FAIL'}"
    )
    print(
        f"  cross_encoder no regress:  "
        f"{'PASS' if gates.gate_cross_encoder_no_regression else 'FAIL'}"
    )

    if gates.failure_reasons:
        print("\nFailure reasons:")
        for reason in gates.failure_reasons:
            print(f"  - {reason}")

    return 0 if gates.passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 6.5 retrieval evaluation harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=_DEFAULT_CASES,
        help="Path to eval_cases.json fixture file.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=_DEFAULT_CORPUS,
        help="Path to corpus.json fixture file (mock mode only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation"),
        help="Directory to write retrieval-report.json and .md.",
    )
    parser.add_argument(
        "--real-vespa",
        action="store_true",
        default=os.environ.get("RETRIEVAL_EVAL_REAL_VESPA", "0") == "1",
        help="Connect to real Vespa (localhost:8080) instead of FakeVespa.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
        help="Embedding dimension for mock mode (ignored in real-vespa mode).",
    )
    args = parser.parse_args()

    return asyncio.run(
        _run(
            cases_path=args.cases,
            corpus_path=args.corpus,
            output_dir=args.output_dir,
            real_vespa=args.real_vespa,
            embedding_dim=args.embedding_dim,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
