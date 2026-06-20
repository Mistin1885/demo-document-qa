"""Tests for deterministic grep_document_chunks retrieval."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.agent.state import AgentState
from app.agent.tools._invocation import ToolDeps
from app.agent.tools._models import GrepDocumentChunksParams
from app.agent.tools.grep_document_chunks import grep_document_chunks
from app.models.orm import DocumentNode

_CHAT_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_SESSION_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_DOC_ID = uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")


def _node(*, node_type: str, title: str | None, content: str, order_index: int) -> DocumentNode:
    return DocumentNode(
        id=uuid.uuid4(),
        chat_id=_CHAT_ID,
        document_id=_DOC_ID,
        parent_id=None,
        node_type=node_type,
        title=title,
        content=content,
        page_start=order_index,
        page_end=order_index,
        order_index=order_index,
        level=1,
        bbox=None,
        metadata_={},
    )


class _FakeScalarResult:
    def __init__(self, nodes: list[DocumentNode]) -> None:
        self._nodes = nodes

    def all(self) -> list[DocumentNode]:
        return self._nodes


class _FakeExecuteResult:
    def __init__(self, nodes: list[DocumentNode]) -> None:
        self._nodes = nodes

    def scalars(self) -> _FakeScalarResult:
        return _FakeScalarResult(self._nodes)


class _FakeSession:
    def __init__(self, nodes: list[DocumentNode]) -> None:
        self.nodes = nodes

    async def execute(self, stmt: Any) -> _FakeExecuteResult:  # noqa: ARG002
        return _FakeExecuteResult(self.nodes)


def _deps(nodes: list[DocumentNode]) -> ToolDeps:
    @asynccontextmanager
    async def session_factory() -> AsyncIterator[_FakeSession]:
        yield _FakeSession(nodes)

    return ToolDeps(
        retrieval_service=MagicMock(),  # type: ignore[arg-type]
        chat_provider=MagicMock(),  # type: ignore[arg-type]
        session_factory=session_factory,
    )


def _state(question: str) -> AgentState:
    return AgentState(chat_id=_CHAT_ID, session_id=_SESSION_ID, question=question)


@pytest.mark.asyncio
async def test_grep_prefers_exact_figure_label_over_wrong_figure() -> None:
    nodes = [
        _node(
            node_type="figure",
            title="Figure 1 architecture",
            content="Figure 1 shows the LightRAG architecture and graph retrieval.",
            order_index=1,
        ),
        _node(
            node_type="figure",
            title="Figure 2 cost",
            content=(
                "Figure 2 compares cost: GraphRAG uses 610 level-2 community reports "
                "with 1,000 tokens each; LightRAG uses fewer than 100 tokens and one API call."
            ),
            order_index=2,
        ),
    ]

    invocation = await grep_document_chunks(
        _state("What exact token and API-call contrast is shown in Figure 2?"),
        GrepDocumentChunksParams(
            query="What exact token and API-call contrast is shown in Figure 2?",
            source_types=["figure_caption"],
            required_terms=["Figure 2"],
            limit=2,
        ),
        deps=_deps(nodes),
    )

    assert invocation.record.status == "ok"
    assert invocation.evidence[0].content.startswith("Figure 2")
    assert "610" in invocation.evidence[0].content
    assert invocation.evidence[0].origin_tool == "grep_document_chunks"


@pytest.mark.asyncio
async def test_grep_preserves_html_table_chunks_when_requested() -> None:
    table_html = (
        "<table><caption>Table 2 Performance of ablated versions of LightRAG</caption>"
        "<tr><th>Variant</th><th>Legal Overall</th></tr>"
        "<tr><td>Full LightRAG</td><td>84.8%</td></tr>"
        "<tr><td>-High</td><td>78.0%</td></tr>"
        "<tr><td>-Low</td><td>81.2%</td></tr>"
        "<tr><td>-Origin</td><td>84.4%</td></tr></table>"
    )
    table_node = _node(node_type="table", title="Table 2", content="Table 2 caption only", order_index=2)
    table_node.metadata_ = {"html_body": table_html}
    nodes = [
        _node(node_type="paragraph", title=None, content="Ablation narrative without rows.", order_index=1),
        table_node,
    ]

    invocation = await grep_document_chunks(
        _state("Give the ablation table rows for Full LightRAG, -High, -Low, and -Origin."),
        GrepDocumentChunksParams(
            query="Table 2 ablation rows Full LightRAG -High -Low -Origin",
            source_types=["table_record"],
            required_terms=["Table 2", "-High", "-Low", "-Origin"],
            include_html=True,
            limit=1,
        ),
        deps=_deps(nodes),
    )

    assert invocation.record.status == "ok"
    assert invocation.evidence[0].source_type == "table"
    assert "<table>" in invocation.evidence[0].content
    assert "-High" in invocation.evidence[0].content
    assert "84.8%" in invocation.evidence[0].content
