"""Unit tests for Phase 5.2 (LLM JSON path) — DocumentOverview enrichment.

Reduced to ≤10 tests covering: happy-path structure, determinism, retry on
garbage, failure on double-garbage, source_section_ids fallback,
to_document_summary_rows structure, and ids from hierarchy.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest

from app.enrichment._orm_bridge import to_document_summary_rows
from app.enrichment.document_overview import enrich_document_overview
from app.enrichment.models import SectionEnrichment
from app.enrichment.section import EnrichmentParseError, enrich_document_sections
from app.parsing.hierarchy import derive_hierarchy
from app.parsing.mapping import map_middle_to_parsed_blocks
from app.parsing.models import HierarchyResult, ParsedBlock
from app.providers.base import (
    ChatChunk,
    ChatCompletion,
    ChatMessage,
    ChatProvider,
    ProviderTestResult,
    Usage,
)
from app.providers.mock import FixtureChatProvider

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHAT_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_DOC_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "mineru_sample_paper"


# ---------------------------------------------------------------------------
# Shared fixture — parse + hierarchy + section enrichments
# ---------------------------------------------------------------------------


def _load_fixture_middle() -> dict:  # type: ignore[type-arg]
    middle_path = _FIXTURE_DIR / "middle.json"
    return json.loads(middle_path.read_text())


async def _build_full_pipeline(
    chat_provider: ChatProvider | None = None,
) -> tuple[list[ParsedBlock], HierarchyResult, list[SectionEnrichment]]:
    middle_data = _load_fixture_middle()
    blocks = map_middle_to_parsed_blocks(middle_data, chat_id=_CHAT_ID, document_id=_DOC_ID)
    hierarchy = derive_hierarchy(blocks, chat_id=_CHAT_ID, document_id=_DOC_ID)
    provider = chat_provider or FixtureChatProvider(model="mock-chat")
    section_enrichments = await enrich_document_sections(hierarchy, blocks, chat_provider=provider)
    return blocks, hierarchy, section_enrichments


# ---------------------------------------------------------------------------
# Helper providers for retry / failure tests
# ---------------------------------------------------------------------------


class _SequentialChatProvider(ChatProvider):
    def __init__(self, responses: list[str], model: str = "sequential-mock") -> None:
        self._responses = responses
        self._index = 0
        self._model_name = model

    @property
    def name(self) -> str:
        return "sequential"

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def context_window(self) -> int:
        return 8192

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> ChatCompletion:
        content = self._responses[self._index % len(self._responses)]
        self._index += 1
        return ChatCompletion(
            content=content,
            usage=Usage(prompt_tokens=10, completion_tokens=len(content.split())),
            model=self._model_name,
        )

    async def stream(  # type: ignore[override]
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        raise NotImplementedError

    async def test_connection(self) -> ProviderTestResult:
        return ProviderTestResult(ok=True, model=self._model_name, latency_ms=0)


def _make_valid_overview_json(section_enrichments: list[SectionEnrichment]) -> str:
    return json.dumps(
        {
            "overview": "This paper presents a novel approach to scientific document question answering. "
            "The authors propose an adaptive retrieval framework that combines sparse and dense signals. "
            "Experiments demonstrate significant improvements over existing baselines. "
            "The method achieves state-of-the-art results on five benchmarks. "
            "The key insight is that query complexity should drive retrieval strategy selection.",
            "contributions": [
                {
                    "title": "Adaptive Query Router",
                    "summary": "Routes queries based on complexity for optimal retrieval.",
                    "source_section_ids": [str(section_enrichments[0].source_node_id)]
                    if section_enrichments
                    else [],
                }
            ],
            "methods": [
                {
                    "name": "Hybrid Retrieval",
                    "description": "Combines BM25 and dense vector search.",
                    "source_section_ids": [],
                }
            ],
            "findings": [
                {
                    "statement": "AdaRAG outperforms all baselines by 6.3% F1 on SciQ.",
                    "evidence": "Table 2",
                    "source_section_ids": [],
                }
            ],
            "limitations": [
                {
                    "text": "Requires labelled complexity annotations for the query router.",
                    "source_section_ids": [],
                }
            ],
            "datasets": [
                {
                    "name": "SciQ",
                    "role": "benchmark",
                    "size_hint": "13.7K questions",
                    "source_section_ids": [],
                }
            ],
            "metrics": [
                {
                    "name": "F1",
                    "best_value": 0.874,
                    "baseline_value": 0.811,
                    "improvement": "+6.3%",
                    "source_section_ids": [],
                }
            ],
            "conclusions": [
                {
                    "statement": "Future work should explore RL-based routing.",
                    "category": "future_work",
                    "source_section_ids": [],
                }
            ],
        }
    )


def _make_valid_overview_json_no_ids(section_enrichments: list[SectionEnrichment]) -> str:
    return json.dumps(
        {
            "overview": "This paper presents a novel approach to scientific document question answering. "
            "The authors propose an adaptive retrieval framework that combines sparse and dense signals. "
            "Experiments demonstrate significant improvements over existing baselines. "
            "The method achieves state-of-the-art results on five benchmarks. "
            "The key insight is that query complexity should drive retrieval strategy selection.",
            "contributions": [
                {
                    "title": "Adaptive Query Router",
                    "summary": "Routes queries based on complexity for optimal retrieval.",
                    "source_section_ids": [],
                }
            ],
            "methods": [],
            "findings": [],
            "limitations": [],
            "datasets": [
                {
                    "name": "SciQ",
                    "role": "benchmark",
                    "size_hint": None,
                    "source_section_ids": [],
                }
            ],
            "metrics": [],
            "conclusions": [],
        }
    )


# ---------------------------------------------------------------------------
# Test 1 — happy-path: overview length, contributions, datasets, token_count, model_used
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_document_overview_happy_path() -> None:
    """Full pipeline: overview >= 80 words, contributions >= 1, datasets >= 1, token_count > 0."""
    blocks, hierarchy, section_enrichments = await _build_full_pipeline()
    provider = FixtureChatProvider(model="mock-chat")

    overview = await enrich_document_overview(
        hierarchy, section_enrichments, chat_provider=provider, blocks=blocks
    )

    assert overview.overview
    assert len(overview.overview.split()) >= 80
    assert len(overview.contributions) >= 1
    all_node_ids = {se.source_node_id for se in section_enrichments}
    for contrib in overview.contributions:
        for sid in contrib.source_section_ids:
            assert sid in all_node_ids
    assert len(overview.datasets) >= 1
    assert overview.token_count_estimate > 0
    assert "mock" in overview.model_used.lower()
    assert overview.chat_id == hierarchy.chat_id
    assert overview.document_id == hierarchy.document_id


# ---------------------------------------------------------------------------
# Test 2 — determinism: two calls produce byte-identical model_dump_json
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_document_overview_deterministic() -> None:
    """Two identical calls must produce identical JSON."""
    blocks, hierarchy, section_enrichments = await _build_full_pipeline()
    provider = FixtureChatProvider(model="mock-chat")

    run1 = await enrich_document_overview(
        hierarchy, section_enrichments, chat_provider=provider, blocks=blocks
    )
    run2 = await enrich_document_overview(
        hierarchy, section_enrichments, chat_provider=provider, blocks=blocks
    )
    assert run1.model_dump_json() == run2.model_dump_json()


# ---------------------------------------------------------------------------
# Test 3 — retry on garbage: first call garbage, second valid JSON → success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_document_overview_retry_on_garbage() -> None:
    """First response is garbage; second is valid JSON → success after retry."""
    blocks, hierarchy, section_enrichments = await _build_full_pipeline()
    valid_json = _make_valid_overview_json(section_enrichments)
    provider = _SequentialChatProvider(
        responses=["this is not valid json at all !!!!", valid_json],
        model="sequential-mock",
    )
    overview = await enrich_document_overview(
        hierarchy, section_enrichments, chat_provider=provider, blocks=blocks
    )
    assert overview.overview
    assert overview.model_used == "sequential-mock"


# ---------------------------------------------------------------------------
# Test 4 — failure: two consecutive garbage responses → EnrichmentParseError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_document_overview_fails_on_double_garbage() -> None:
    """Two consecutive garbage responses must raise EnrichmentParseError."""
    blocks, hierarchy, section_enrichments = await _build_full_pipeline()
    provider = _SequentialChatProvider(
        responses=["garbage one", "garbage two"],
        model="garbage-mock",
    )
    with pytest.raises(EnrichmentParseError):
        await enrich_document_overview(
            hierarchy, section_enrichments, chat_provider=provider, blocks=blocks
        )


# ---------------------------------------------------------------------------
# Test 5 — source_section_ids fallback when LLM returns empty ids
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_document_overview_source_ids_fallback() -> None:
    """When LLM omits source_section_ids, fallback = abstract+section+subsection IDs."""
    blocks, hierarchy, section_enrichments = await _build_full_pipeline()
    valid_json_no_ids = _make_valid_overview_json_no_ids(section_enrichments)
    provider = _SequentialChatProvider(responses=[valid_json_no_ids], model="no-ids-mock")

    overview = await enrich_document_overview(
        hierarchy, section_enrichments, chat_provider=provider, blocks=blocks
    )

    fallback_ids = {
        se.source_node_id
        for se in section_enrichments
        if se.node_type in ("abstract", "section", "subsection")
    }
    assert len(overview.source_section_ids) > 0
    for sid in overview.source_section_ids:
        assert sid in fallback_ids


# ---------------------------------------------------------------------------
# Test 6 — to_document_summary_rows: >= 8 rows, all kinds unique
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_to_document_summary_rows_structure() -> None:
    """to_document_summary_rows must produce >= 8 rows with unique kinds."""
    blocks, hierarchy, section_enrichments = await _build_full_pipeline()
    provider = FixtureChatProvider(model="mock-chat")

    overview = await enrich_document_overview(
        hierarchy, section_enrichments, chat_provider=provider, blocks=blocks
    )
    rows = to_document_summary_rows(overview)

    assert len(rows) >= 8
    kinds = [row.kind for row in rows]
    assert len(kinds) == len(set(kinds))
    expected_kinds = {
        "document_overview",
        "document_contributions",
        "document_methods",
        "document_findings",
        "document_limitations",
        "document_datasets",
        "document_metrics",
        "document_conclusions",
    }
    missing = expected_kinds - set(kinds)
    assert not missing, f"Missing expected kinds: {missing}"
    for row in rows:
        assert row.chat_id == _CHAT_ID
        assert row.document_id == _DOC_ID
        assert row.source_node_id is None
