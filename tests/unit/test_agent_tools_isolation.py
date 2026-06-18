"""Isolation tests for Phase 7.2 agent tools (CLAUDE.md §2, §8).

CLAUDE.md §12.1: ≤10 items per file.

Items:
  1.  All *Params models have no chat_id field (reflection check).
  2–8. (parametrize 7 cases) Injecting chat_id into any Params raises ValidationError.
  9.   search_hybrid passes state.chat_id to RetrievalService (not a fake from params).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from app.agent.state import AgentState
from app.agent.tools import ToolDeps
from app.agent.tools._models import (
    AggregateSourcesParams,
    ExpandEvidenceParams,
    FetchStructuralNodesParams,
    InspectChatParams,
    InspectDocumentParams,
    QueryStructuredFactsParams,
    SearchHybridParams,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHAT_ID = uuid.uuid4()
SESSION_ID = uuid.uuid4()
DOC_ID = uuid.uuid4()

_ALL_PARAMS_MODELS = [
    InspectChatParams,
    InspectDocumentParams,
    FetchStructuralNodesParams,
    SearchHybridParams,
    QueryStructuredFactsParams,
    AggregateSourcesParams,
    ExpandEvidenceParams,
]

_ALL_PARAMS_NAMES = [m.__name__ for m in _ALL_PARAMS_MODELS]


def _state() -> AgentState:
    return AgentState(
        chat_id=CHAT_ID,
        session_id=SESSION_ID,
        question="What is the result?",
    )


@asynccontextmanager
async def _fake_session_factory() -> AsyncGenerator[MagicMock, None]:
    session = MagicMock()
    session.execute = AsyncMock(
        return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))),
            scalar_one_or_none=MagicMock(return_value=None),
            scalar_one=MagicMock(return_value=0),
        )
    )
    yield session


def _fake_deps() -> ToolDeps:
    from app.retrieval.models import RetrievalResponse

    svc = MagicMock()
    svc.search = AsyncMock(return_value=RetrievalResponse(hits=[]))
    return ToolDeps(
        retrieval_service=svc,
        chat_provider=MagicMock(),
        session_factory=_fake_session_factory,
    )


# ---------------------------------------------------------------------------
# Test 1: No *Params model contains a chat_id field
# ---------------------------------------------------------------------------


def test_no_params_model_has_chat_id_field() -> None:
    """Reflection: model_fields must not include 'chat_id' in any *Params model."""
    for model_cls in _ALL_PARAMS_MODELS:
        assert "chat_id" not in model_cls.model_fields, (
            f"{model_cls.__name__} must not have a chat_id field (CLAUDE.md §8)"
        )


# ---------------------------------------------------------------------------
# Tests 2–8 (parametrize 7 cases): injecting chat_id triggers ValidationError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls", _ALL_PARAMS_MODELS, ids=_ALL_PARAMS_NAMES)
def test_chat_id_injection_raises_validation_error(model_cls: type) -> None:
    """extra='forbid' must reject any attempt to pass chat_id to *Params.

    This simulates an LLM attempting to inject chat_id into the tool params.
    """
    with pytest.raises(ValidationError):
        # Each model has different required fields — supply minimal valid kwargs
        # plus the forbidden chat_id.
        minimal: dict[str, object] = {"chat_id": str(uuid.uuid4())}

        # Supply required fields per model
        if model_cls is InspectDocumentParams:
            minimal["document_id"] = uuid.uuid4()
        elif model_cls is SearchHybridParams:
            minimal["query"] = "test"
        elif model_cls is ExpandEvidenceParams:
            minimal["evidence_id"] = "some-id"

        model_cls(**minimal)


# ---------------------------------------------------------------------------
# Test 9: search_hybrid uses state.chat_id, not a value from params
# ---------------------------------------------------------------------------


async def test_search_hybrid_uses_state_chat_id_not_params() -> None:
    """Verify the RetrievalRequest passed to RetrievalService carries state.chat_id."""
    from app.agent.tools.search_hybrid import search_hybrid

    captured_req: list = []

    async def _capture_search(req: object) -> object:
        captured_req.append(req)
        from app.retrieval.models import RetrievalResponse

        return RetrievalResponse(hits=[])

    attacker_chat_id = uuid.uuid4()  # a different UUID the LLM might try to use
    assert attacker_chat_id != CHAT_ID

    # The attacker cannot put chat_id in SearchHybridParams (extra=forbid).
    # Verify the call uses CHAT_ID from state, not attacker_chat_id.
    svc = MagicMock()
    svc.search = AsyncMock(side_effect=_capture_search)
    deps = ToolDeps(
        retrieval_service=svc,
        chat_provider=MagicMock(),
        session_factory=_fake_session_factory,
    )

    state = _state()
    params = SearchHybridParams(query="attention")
    await search_hybrid(state, params, deps=deps)

    assert len(captured_req) == 1
    req = captured_req[0]
    assert req.chat_id == CHAT_ID, f"Expected chat_id={CHAT_ID} from state, got {req.chat_id}"
    assert req.chat_id != attacker_chat_id
