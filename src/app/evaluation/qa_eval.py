"""qa_eval.py — Phase 9.1 Golden QA evaluation harness.

Runs the seven Golden QA case kinds defined in GUIDE §19 against the LangGraph
StateGraph with a chat-scoped mock RetrievalService.  No real DB, Vespa, or
paid LLM calls are made — the harness operates entirely at the agent-graph
boundary so it can run as a regular pytest under ``tests/evaluation``.

Case definitions live in ``data/fixtures/qa_cases.json``.  Each case maps
chat / document *labels* (e.g. ``"A"`` / ``"paper_a"``) to deterministic
``uuid5`` UUIDs so that re-runs produce identical IDs and assertions stay
stable.

Isolation invariants enforced by the runner
-------------------------------------------
- The mock retrieval service returns only hits whose ``chat_label`` maps to
  the *current* ``chat_id`` — exactly mirroring the production
  ``RetrievalService`` Vespa filter.
- The runner never inspects the answer text other than to detect the
  "not enough information" refusal phrase; all other assertions are
  structural (citations, documents_used, coverage).

The public functions are::

    load_qa_cases(path) -> list[QACaseSpec]
    run_case(spec) -> QACaseResult
    evaluate_corpus(specs) -> QACorpusReport

CLAUDE.md §12.1: the corresponding test file
``tests/evaluation/test_qa_eval.py`` is capped at ≤ 10 collect-only items.
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
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field

from app.agent.graph import build_graph
from app.agent.nodes.persist_messages import InMemoryMessageStore
from app.agent.state import AgentState, GenerationConfig
from app.agent.tools._invocation import ToolDeps
from app.providers.base import ChatCompletion, ChatMessage, Usage
from app.providers.mock import MockChatProvider
from app.retrieval.models import RetrievalRequest, RetrievalResponse, SearchHit

# ---------------------------------------------------------------------------
# Deterministic UUID derivation
# ---------------------------------------------------------------------------

_NS_CHAT = uuid.NAMESPACE_OID
_NS_DOC = uuid.NAMESPACE_OID
_NS_SESSION = uuid.NAMESPACE_OID


def chat_uuid(label: str) -> UUID:
    """Map a chat label (e.g. ``"A"``) to a deterministic UUID."""
    return uuid5(_NS_CHAT, f"qa_eval/chat/{label}")


def doc_uuid(label: str) -> UUID:
    """Map a document label (e.g. ``"paper_a"``) to a deterministic UUID."""
    return uuid5(_NS_DOC, f"qa_eval/document/{label}")


def session_uuid(chat_label: str, session_label: str) -> UUID:
    """Map a (chat_label, session_label) pair to a deterministic UUID."""
    return uuid5(_NS_SESSION, f"qa_eval/session/{chat_label}/{session_label}")


# ---------------------------------------------------------------------------
# Case schema
# ---------------------------------------------------------------------------


CaseKind = Literal[
    "global_summary",
    "method_explanation",
    "technical_comparison",
    "performance",
    "multi_document",
    "chat_isolation",
    "session_isolation",
]


class SearchHitSpec(BaseModel):
    """One mock search hit to seed in the retrieval service."""

    model_config = ConfigDict(extra="forbid")

    chat_label: str
    doc_label: str
    content: str
    page_start: int = Field(ge=0, default=1)
    page_end: int = Field(ge=0, default=1)
    source_type: str = "chunk"


class ExpectedOutcome(BaseModel):
    """Structural assertions for one case."""

    model_config = ConfigDict(extra="forbid")

    required_citation_doc_labels: list[str] = Field(default_factory=list)
    forbidden_citation_doc_labels: list[str] = Field(default_factory=list)
    no_answer_phrase_required: bool = False
    min_documents_used: int = Field(ge=0, default=0)
    max_documents_used: int | None = Field(ge=0, default=None)


class QACaseSpec(BaseModel):
    """One golden QA case loaded from ``qa_cases.json``."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    kind: CaseKind
    question: str
    target_chat_label: str
    target_session_label: str
    seeded_hits: list[SearchHitSpec]
    expected: ExpectedOutcome
    runner_note: str | None = None


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


class QACaseResult(BaseModel):
    """Outcome of running one ``QACaseSpec``."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    kind: CaseKind
    passed: bool
    failures: list[str] = Field(default_factory=list)
    answer_snippet: str
    citation_count: int = Field(ge=0)
    citation_doc_labels: list[str] = Field(default_factory=list)
    documents_used_count: int = Field(ge=0)
    coverage_state: Literal["pending", "incomplete", "complete"]
    retrieval_calls: int = Field(ge=0)


class QAKindStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0, default=0)
    passed: int = Field(ge=0, default=0)
    failed: int = Field(ge=0, default=0)


class QACorpusReport(BaseModel):
    """Summary report after running every case."""

    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    per_kind: dict[str, QAKindStats] = Field(default_factory=dict)
    results: list[QACaseResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# CitingMockChatProvider — emits [c1] [c2] ... markers when evidence is present
# ---------------------------------------------------------------------------


_MARKER_RE = re.compile(r"\[c(\d+)\]")


class CitingMockChatProvider(MockChatProvider):
    """Mock chat provider that scans the user prompt for ``[c<idx>]`` placeholders
    in the evidence block and emits an answer string containing every marker.

    The ``validate_citations`` node turns each marker into a ``CitationDraft``
    bound to the corresponding ``EvidenceItem`` (whose chat_id was already
    enforced upstream), so the runner can structurally assert citation scope
    without depending on real LLM output.

    When the user prompt has no markers (no evidence), the provider falls back
    to the deterministic hash answer produced by ``MockChatProvider`` — that
    answer never contains markers, so ``validate_citations`` yields zero
    citations and ``generate_answer`` triggers the no-info fallback.
    """

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> ChatCompletion:
        user_blob = "\n".join(m.content for m in messages if m.role == "user")
        markers = sorted({int(m) for m in _MARKER_RE.findall(user_blob)})
        if not markers:
            return await super().complete(
                messages, temperature=temperature, max_tokens=max_tokens, stop=stop
            )
        marker_str = " ".join(f"[c{i}]" for i in markers)
        content = (
            f"Based on the retrieved evidence, the answer is supported by the "
            f"following sources: {marker_str}."
        )
        return ChatCompletion(
            content=content,
            usage=Usage(prompt_tokens=len(user_blob) // 4, completion_tokens=len(content.split())),
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Mock RetrievalService — enforces chat_id filter
# ---------------------------------------------------------------------------


class _ScopedMockRetrieval:
    """Mock retrieval that enforces the production chat_id filter contract.

    Seeded with ``SearchHitSpec`` entries spanning *all* chats; at query time
    only those whose ``chat_label`` resolves to the request's ``chat_id`` are
    returned — mirroring the production Vespa ``chat_id contains`` filter.
    """

    def __init__(self, hits: list[SearchHitSpec], label_to_chat_id: dict[str, UUID]) -> None:
        self._hits = hits
        self._label_to_chat_id = label_to_chat_id
        self.call_count = 0
        self.last_request: RetrievalRequest | None = None

    async def search(self, request: RetrievalRequest) -> RetrievalResponse:
        self.call_count += 1
        self.last_request = request
        target_chat_id = request.chat_id
        out: list[SearchHit] = []
        for idx, spec in enumerate(self._hits):
            spec_chat_id = self._label_to_chat_id.get(spec.chat_label)
            if spec_chat_id is None or spec_chat_id != target_chat_id:
                continue
            out.append(
                SearchHit(
                    vespa_document_id=f"id::document_chunk::eval-{spec.doc_label}-{idx}",
                    chat_id=str(spec_chat_id),
                    document_id=str(doc_uuid(spec.doc_label)),
                    source_node_id=f"node-{spec.doc_label}-{idx}",
                    source_type=spec.source_type,
                    content=spec.content,
                    page_start=spec.page_start,
                    page_end=spec.page_end,
                    order_index=idx,
                    fusion_score=0.8 - 0.01 * idx,
                    final_score=0.8 - 0.01 * idx,
                    final_rank=idx + 1,
                )
            )
        return RetrievalResponse(hits=out)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_qa_cases(path: str | Path) -> list[QACaseSpec]:
    """Load and validate Golden QA cases from ``qa_cases.json``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_cases = data.get("cases", [])
    return [QACaseSpec.model_validate(c) for c in raw_cases]


# ---------------------------------------------------------------------------
# Single-case runner
# ---------------------------------------------------------------------------


_NO_ANSWER_PHRASE = "not enough information"


@asynccontextmanager
async def _null_session() -> AsyncIterator[Any]:
    """Yield a MagicMock — the graph does not perform DB I/O for these cases."""
    yield MagicMock()


def _all_chat_labels(spec: QACaseSpec) -> list[str]:
    labels = {spec.target_chat_label}
    for h in spec.seeded_hits:
        labels.add(h.chat_label)
    return sorted(labels)


async def run_case(spec: QACaseSpec) -> QACaseResult:
    """Replay one golden QA case through the LangGraph StateGraph.

    Builds a deterministic chat-scoped retrieval mock, invokes the graph, and
    converts the final ``AgentState`` into structural assertions.
    """
    label_to_chat = {label: chat_uuid(label) for label in _all_chat_labels(spec)}
    target_chat_id = label_to_chat[spec.target_chat_label]
    target_session_id = session_uuid(spec.target_chat_label, spec.target_session_label)

    retrieval = _ScopedMockRetrieval(spec.seeded_hits, label_to_chat)
    chat_provider = CitingMockChatProvider()
    # ToolDeps expects a RetrievalService Protocol; _ScopedMockRetrieval is a
    # structural match (async .search returning RetrievalResponse) — annotate
    # the assignment to silence mypy's nominal-type check.
    deps = ToolDeps(
        retrieval_service=retrieval,  # type: ignore[arg-type]
        chat_provider=chat_provider,
        session_factory=_null_session,
    )
    store = InMemoryMessageStore(expected_chat_id=target_chat_id)

    state = AgentState(
        chat_id=target_chat_id,
        session_id=target_session_id,
        question=spec.question,
        generation_config=GenerationConfig(),
    )
    graph = build_graph(deps=deps, chat_provider=chat_provider, message_store=store)
    container = await graph.ainvoke({"state": state.model_dump()})
    final = AgentState.model_validate(container["state"])

    return _assert(spec, final, retrieval, label_to_chat)


def _assert(
    spec: QACaseSpec,
    final: AgentState,
    retrieval: _ScopedMockRetrieval,
    label_to_chat: dict[str, UUID],
) -> QACaseResult:
    failures: list[str] = []

    answer_text = (final.answer or "").lower()
    citations = final.citations

    citation_doc_ids = {c.document_id for c in citations}
    label_for_doc = {doc_uuid(label): label for label in {h.doc_label for h in spec.seeded_hits}}
    citation_doc_labels = sorted({label_for_doc.get(did, str(did)) for did in citation_doc_ids})

    # Invariant: every citation must belong to the target chat
    for c in citations:
        if c.chat_id != label_to_chat[spec.target_chat_label]:
            failures.append(
                f"citation {c.citation_id} has cross-chat chat_id={c.chat_id}"
            )

    expected = spec.expected

    if expected.no_answer_phrase_required and _NO_ANSWER_PHRASE not in answer_text:
        failures.append(f"refusal phrase missing; got answer={final.answer!r}")

    for required_label in expected.required_citation_doc_labels:
        if required_label not in citation_doc_labels:
            failures.append(f"required citation doc {required_label!r} not present")

    for forbidden_label in expected.forbidden_citation_doc_labels:
        if forbidden_label in citation_doc_labels:
            failures.append(f"forbidden citation doc {forbidden_label!r} leaked")

    docs_used_count = len({c.document_id for c in citations})
    if docs_used_count < expected.min_documents_used:
        failures.append(
            f"documents_used={docs_used_count} < min={expected.min_documents_used}"
        )
    if expected.max_documents_used is not None and docs_used_count > expected.max_documents_used:
        failures.append(
            f"documents_used={docs_used_count} > max={expected.max_documents_used}"
        )

    return QACaseResult(
        case_id=spec.case_id,
        kind=spec.kind,
        passed=not failures,
        failures=failures,
        answer_snippet=(final.answer or "")[:200],
        citation_count=len(citations),
        citation_doc_labels=citation_doc_labels,
        documents_used_count=docs_used_count,
        coverage_state=final.coverage_state,
        retrieval_calls=retrieval.call_count,
    )


# ---------------------------------------------------------------------------
# Corpus runner
# ---------------------------------------------------------------------------


async def evaluate_corpus(specs: list[QACaseSpec]) -> QACorpusReport:
    """Run every case and aggregate a corpus-level report."""
    results: list[QACaseResult] = []
    for spec in specs:
        results.append(await run_case(spec))

    per_kind: dict[str, QAKindStats] = {}
    for r in results:
        stats = per_kind.setdefault(r.kind, QAKindStats())
        stats.total += 1
        if r.passed:
            stats.passed += 1
        else:
            stats.failed += 1

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    pass_rate = (passed / total) if total else 0.0

    return QACorpusReport(
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=pass_rate,
        per_kind=per_kind,
        results=results,
    )


__all__ = [
    "CaseKind",
    "ExpectedOutcome",
    "QACaseResult",
    "QACaseSpec",
    "QACorpusReport",
    "QAKindStats",
    "SearchHitSpec",
    "chat_uuid",
    "doc_uuid",
    "evaluate_corpus",
    "load_qa_cases",
    "run_case",
    "session_uuid",
]
