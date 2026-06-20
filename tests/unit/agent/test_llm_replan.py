"""Phase C llm_replan tests (≤10 tests)."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest

from app.agent.budget import MAX_REPLAN_ROUNDS
from app.agent.nodes.llm_replan import llm_replan
from app.agent.state import AgentState, CoverageRequirement
from app.providers.base import (
    ChatChunk,
    ChatCompletion,
    ChatMessage,
    ChatProvider,
    ProviderTestResult,
    Usage,
)

_CHAT_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_SESSION_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


class _Provider(ChatProvider):
    def __init__(self, content: str) -> None:
        self.content = content

    @property
    def name(self) -> str:
        return "test"

    @property
    def model(self) -> str:
        return "test-model"

    @property
    def context_window(self) -> int:
        return 10000

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> ChatCompletion:
        return ChatCompletion(content=self.content, usage=Usage(), model=self.model)

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        if False:
            yield ChatChunk(delta="")

    async def test_connection(self) -> ProviderTestResult:
        return ProviderTestResult(ok=True, model=self.model)


def _state(**kwargs: object) -> AgentState:
    base: dict[str, object] = {
        "chat_id": _CHAT_ID,
        "session_id": _SESSION_ID,
        "question": "Compare LightRAG with GraphRAG",
        "coverage_requirements": [
            CoverageRequirement(requirement_id="r1", description="GraphRAG details")
        ],
    }
    base.update(kwargs)
    return AgentState(**base)


@pytest.mark.asyncio
async def test_llm_replan_happy_path_search_more() -> None:
    provider = _Provider(
        '{"action":"search_more","tool_calls":[{"tool":"search_hybrid","query":"GraphRAG architecture"}],"reasoning":"Need GraphRAG."}'
    )
    result = await llm_replan(_state(), provider)
    assert result["replan_rounds"] == 1
    assert result["replan_tool_calls"][0].tool == "search_hybrid"
    assert result["replan_tool_calls"][0].query == "GraphRAG architecture"


@pytest.mark.asyncio
async def test_llm_replan_schema_violation_falls_back() -> None:
    provider = _Provider(
        '{"action":"search_more","tool_calls":[{"tool":"search_hybrid","query":"x","chat_id":"bad"}],"reasoning":"bad"}'
    )
    result = await llm_replan(_state(), provider)
    assert result["replan_tool_calls"] == []
    assert result["errors"][-1].code == "LLM_REPLAN_FALLBACK"


@pytest.mark.asyncio
async def test_llm_replan_round_cap_skips() -> None:
    provider = _Provider('{"action":"search_more","tool_calls":[],"reasoning":"x"}')
    result = await llm_replan(_state(replan_rounds=MAX_REPLAN_ROUNDS), provider)
    assert result["replan_tool_calls"] == []
    assert "replan_rounds" not in result
