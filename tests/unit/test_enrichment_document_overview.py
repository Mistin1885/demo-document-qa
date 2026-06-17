"""Unit tests for Phase 5.2 (LLM JSON path) — DocumentOverview enrichment.

All tests are fully deterministic — no real LLM is called.
``FixtureChatProvider`` (backed by ``tests/fixtures/enrichment_mock_responses.json``)
serves the deterministic ``"document_default"`` fixture for document-level prompts
(identified by the ``[DOCUMENT_OVERVIEW]`` marker in the system prompt).

Coverage
--------
1. happy-path: parse → hierarchy → section-enrichments → ``enrich_document_overview``
   - ``overview`` non-empty and >= 80 chars word count.
   - ``contributions`` >= 1, each ``source_section_ids`` is a subset of all
     section-enrichment node IDs.
   - ``datasets`` >= 1.
   - ``token_count_estimate > 0``.
   - ``model_used`` contains "mock".
2. retry on garbage: first call returns garbage, second returns valid JSON → success.
3. failure: two consecutive garbage responses → ``EnrichmentParseError``.
4. source_section_ids fallback: LLM returns no source_section_ids → fallback to
   all abstract + section + subsection node IDs.
5. ``to_document_summary_rows(overview)`` → >= 8 rows, all kinds unique.
6. deterministic: two identical calls produce byte-identical ``model_dump_json()``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest

from app.enrichment._orm_bridge import to_document_summary_rows
from app.enrichment.document_overview import enrich_document_overview
from app.enrichment.models import DocumentOverview, SectionEnrichment
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
    """Parse the sample paper and enrich sections."""
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
    """Returns responses from a fixed sequence (cycling when exhausted)."""

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
    """Return a minimal valid DocumentOverview JSON string."""
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
    """Return a valid DocumentOverview JSON with empty source_section_ids everywhere."""
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
# Test 1 — happy-path with FixtureChatProvider (uses document_default fixture)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_document_overview_happy_path() -> None:
    """Full pipeline: parse → hierarchy → section enrichments → DocumentOverview."""
    blocks, hierarchy, section_enrichments = await _build_full_pipeline()
    provider = FixtureChatProvider(model="mock-chat")

    overview = await enrich_document_overview(
        hierarchy,
        section_enrichments,
        chat_provider=provider,
        blocks=blocks,
    )

    # overview non-empty and >= 80 words
    assert overview.overview, "overview must not be empty"
    word_count = len(overview.overview.split())
    assert word_count >= 80, f"overview too short: {word_count} words"

    # contributions >= 1, source_section_ids is a subset of all section node IDs
    assert len(overview.contributions) >= 1, "expected at least 1 contribution"
    all_node_ids = {se.source_node_id for se in section_enrichments}
    for contrib in overview.contributions:
        for sid in contrib.source_section_ids:
            assert sid in all_node_ids, (
                f"source_section_id {sid} not in any section enrichment node IDs"
            )

    # datasets >= 1
    assert len(overview.datasets) >= 1, "expected at least 1 dataset"

    # token_count_estimate > 0
    assert overview.token_count_estimate > 0, "token_count_estimate must be positive"

    # model_used contains "mock"
    assert "mock" in overview.model_used.lower(), (
        f"model_used '{overview.model_used}' does not contain 'mock'"
    )


# ---------------------------------------------------------------------------
# Test 2 — determinism: two calls produce byte-identical model_dump_json
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_document_overview_deterministic() -> None:
    """Two identical calls with FixtureChatProvider must produce identical JSON."""
    blocks, hierarchy, section_enrichments = await _build_full_pipeline()
    provider = FixtureChatProvider(model="mock-chat")

    run1 = await enrich_document_overview(
        hierarchy, section_enrichments, chat_provider=provider, blocks=blocks
    )
    run2 = await enrich_document_overview(
        hierarchy, section_enrichments, chat_provider=provider, blocks=blocks
    )

    assert run1.model_dump_json() == run2.model_dump_json(), (
        "enrich_document_overview is not deterministic"
    )


# ---------------------------------------------------------------------------
# Test 3 — retry on garbage: first call returns garbage, second returns JSON
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

    assert overview.overview, "overview must not be empty after retry"
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
    # Section enrichments already built — use a provider that returns JSON with no ids
    provider = _SequentialChatProvider(
        responses=[valid_json_no_ids],
        model="no-ids-mock",
    )

    overview = await enrich_document_overview(
        hierarchy, section_enrichments, chat_provider=provider, blocks=blocks
    )

    # source_section_ids fallback = all abstract + section + subsection node IDs
    fallback_ids = {
        se.source_node_id
        for se in section_enrichments
        if se.node_type in ("abstract", "section", "subsection")
    }

    assert len(overview.source_section_ids) > 0, (
        "source_section_ids must not be empty after fallback"
    )
    for sid in overview.source_section_ids:
        assert sid in fallback_ids, (
            f"fallback source_section_id {sid} not in abstract/section/subsection enrichments"
        )


# ---------------------------------------------------------------------------
# Test 6 — to_document_summary_rows: >= 8 rows, all kinds unique
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_to_document_summary_rows_structure() -> None:
    """to_document_summary_rows must produce >= 8 rows with all unique kinds."""
    blocks, hierarchy, section_enrichments = await _build_full_pipeline()
    provider = FixtureChatProvider(model="mock-chat")

    overview = await enrich_document_overview(
        hierarchy, section_enrichments, chat_provider=provider, blocks=blocks
    )

    rows = to_document_summary_rows(overview)

    assert len(rows) >= 8, f"Expected >= 8 rows, got {len(rows)}"

    kinds = [row.kind for row in rows]
    assert len(kinds) == len(set(kinds)), f"Kinds are not unique: {kinds}"

    # Verify expected kinds are present
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
    actual_kinds = set(kinds)
    missing = expected_kinds - actual_kinds
    assert not missing, f"Missing expected kinds: {missing}"


# ---------------------------------------------------------------------------
# Test 7 — chat_id and document_id come from hierarchy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_document_overview_ids_from_hierarchy() -> None:
    """chat_id and document_id in DocumentOverview must match hierarchy."""
    blocks, hierarchy, section_enrichments = await _build_full_pipeline()
    provider = FixtureChatProvider(model="mock-chat")

    overview = await enrich_document_overview(
        hierarchy, section_enrichments, chat_provider=provider, blocks=blocks
    )

    assert overview.chat_id == hierarchy.chat_id, (
        f"chat_id mismatch: {overview.chat_id} != {hierarchy.chat_id}"
    )
    assert overview.document_id == hierarchy.document_id, (
        f"document_id mismatch: {overview.document_id} != {hierarchy.document_id}"
    )


# ---------------------------------------------------------------------------
# Test 8 — page_count derives from hierarchy when not specified
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_document_overview_page_count() -> None:
    """page_count must be >= 1 and derived from max(node.page_end)."""
    blocks, hierarchy, section_enrichments = await _build_full_pipeline()
    provider = FixtureChatProvider(model="mock-chat")

    overview = await enrich_document_overview(
        hierarchy, section_enrichments, chat_provider=provider, blocks=blocks
    )

    max_page = max(n.page_end for n in hierarchy.nodes)
    assert overview.page_count == max_page, (
        f"page_count {overview.page_count} != max(node.page_end) {max_page}"
    )


# ---------------------------------------------------------------------------
# Test 9 — page_count override
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_document_overview_page_count_override() -> None:
    """page_count override must be respected."""
    blocks, hierarchy, section_enrichments = await _build_full_pipeline()
    provider = FixtureChatProvider(model="mock-chat")

    overview = await enrich_document_overview(
        hierarchy,
        section_enrichments,
        chat_provider=provider,
        blocks=blocks,
        page_count=42,
    )

    assert overview.page_count == 42, (
        f"page_count override not respected: got {overview.page_count}"
    )


# ---------------------------------------------------------------------------
# Test 10 — FixtureChatProvider routing: section prompt still uses "default"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fixture_provider_section_still_uses_default() -> None:
    """Phase 5.1 section prompts must still use the 'default' fixture, not 'document_default'."""
    blocks, hierarchy, section_enrichments = await _build_full_pipeline()

    # Verify the section enrichments don't have any "document_default" content
    # (they should use the "default" fixture which returns a SectionEnrichment JSON)
    assert len(section_enrichments) > 0, "should have produced section enrichments"
    for se in section_enrichments:
        # detailed_summary should be non-empty (from "default" fixture)
        assert se.detailed_summary, f"section enrichment has empty detailed_summary for {se.title}"


# ---------------------------------------------------------------------------
# Test 11 — DocumentOverview model validation: extra fields rejected
# ---------------------------------------------------------------------------


def test_document_overview_forbids_extra_fields() -> None:
    """ConfigDict(extra='forbid') must reject unknown fields."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DocumentOverview(
            chat_id=_CHAT_ID,
            document_id=_DOC_ID,
            page_count=5,
            overview="Some overview text here",
            token_count_estimate=10,
            model_used="test",
            unknown_field="bad",  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# Test 12 — to_document_summary_rows: rows contain correct chat_id + document_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_to_document_summary_rows_ids() -> None:
    """All summary rows must carry the correct chat_id and document_id."""
    blocks, hierarchy, section_enrichments = await _build_full_pipeline()
    provider = FixtureChatProvider(model="mock-chat")

    overview = await enrich_document_overview(
        hierarchy, section_enrichments, chat_provider=provider, blocks=blocks
    )
    rows = to_document_summary_rows(overview)

    for row in rows:
        assert row.chat_id == _CHAT_ID, f"Wrong chat_id in row kind={row.kind}"
        assert row.document_id == _DOC_ID, f"Wrong document_id in row kind={row.kind}"
        assert row.source_node_id is None, (
            f"source_node_id should be None for document-level row kind={row.kind}"
        )
