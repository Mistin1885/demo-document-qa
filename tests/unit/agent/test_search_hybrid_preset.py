"""Unit tests for SearchHybridParams.preset and search_hybrid tool top_k scaling.

Test density cap: ≤ 10 items per file (CLAUDE.md §12.1).
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from app.agent.state import AgentState
from app.agent.tools._models import SearchHybridParams
from app.agent.tools.search_hybrid import search_hybrid
from app.retrieval.models import RetrievalRequest, RetrievalResponse

_CHAT_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_SESSION_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_DOC_ID = uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")


class _CapturingRetrievalService:
    """Records the RetrievalRequest it receives for assertion."""

    def __init__(self) -> None:
        self.last_request: RetrievalRequest | None = None

    async def search(self, request: RetrievalRequest) -> RetrievalResponse:
        self.last_request = request
        return RetrievalResponse(hits=[])


def _make_state() -> AgentState:
    return AgentState(
        chat_id=_CHAT_ID,
        session_id=_SESSION_ID,
        question="test question",
    )


def _make_deps(svc: _CapturingRetrievalService) -> MagicMock:
    from app.agent.tools._invocation import ToolDeps  # noqa: PLC0415

    deps = MagicMock(spec=ToolDeps)
    deps.retrieval_service = svc
    return deps


# 1. preset="broad" → final_top_k is top_k * 2 (up to 60)
@pytest.mark.asyncio
async def test_broad_preset_doubles_top_k() -> None:
    svc = _CapturingRetrievalService()
    params = SearchHybridParams(query="summary question", top_k=8, preset="broad")
    state = _make_state()
    await search_hybrid(state, params, deps=_make_deps(svc))
    assert svc.last_request is not None
    assert svc.last_request.final_top_k == 16


# 2. preset="default" → final_top_k stays as top_k
@pytest.mark.asyncio
async def test_default_preset_keeps_top_k() -> None:
    svc = _CapturingRetrievalService()
    params = SearchHybridParams(query="specific question", top_k=8, preset="default")
    state = _make_state()
    await search_hybrid(state, params, deps=_make_deps(svc))
    assert svc.last_request is not None
    assert svc.last_request.final_top_k == 8


# 3. extra="forbid": unknown field raises ValidationError
def test_unknown_field_raises() -> None:
    with pytest.raises(ValidationError):
        SearchHybridParams(query="q", unknown_field="bad")  # type: ignore[call-arg]
