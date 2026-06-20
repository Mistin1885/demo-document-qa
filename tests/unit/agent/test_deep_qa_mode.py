"""Deep QA mode tests (≤10 tests)."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest

from app.agent.budget import ContextBudgetManager
from app.agent.nodes.check_context_budget import is_overflow
from app.agent.nodes.generate_answer import generate_answer
from app.agent.state import (
    AgentState,
    ConversationTurn,
    EvidenceItem,
    GenerationConfig,
    make_evidence_id,
)
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
_DOC_ID = uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")


class _CaptureProvider(ChatProvider):
    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    @property
    def name(self) -> str:
        return "capture"

    @property
    def model(self) -> str:
        return "capture-model"

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
        self.messages = messages
        return ChatCompletion(content="Answer [c1]", usage=Usage(), model=self.model)

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


def _evidence(content: str = "LightRAG has the best F1 score.") -> EvidenceItem:
    return EvidenceItem(
        evidence_id=make_evidence_id("search_hybrid", "node-1", _DOC_ID),
        source_type="chunk",
        document_id=_DOC_ID,
        source_node_id="node-1",
        page_start=1,
        page_end=1,
        content=content,
        origin_tool="search_hybrid",
    )


@pytest.mark.asyncio
async def test_deep_qa_mode_includes_session_memory_in_answer_prompt() -> None:
    provider = _CaptureProvider()
    state = AgentState(
        chat_id=_CHAT_ID,
        session_id=_SESSION_ID,
        question="結果來看哪個比較好?",
        conversation_history=[
            ConversationTurn(role="user", content="比較這篇 document 不同方法的 performance."),
            ConversationTurn(role="assistant", content="LightRAG and GraphRAG were compared."),
        ],
        evidence_items=[_evidence()],
        generation_config=GenerationConfig(deep_qa_mode=True),
    )

    await generate_answer(state, provider)

    user_prompt = next(m.content for m in provider.messages if m.role == "user")
    assert "Recent same-session conversation" in user_prompt
    assert "不同方法的 performance" in user_prompt
    assert "結果來看哪個比較好" in user_prompt


def test_deep_qa_budget_manager_ignores_soft_overflow() -> None:
    huge = _evidence("x " * 10000)
    state = AgentState(chat_id=_CHAT_ID, session_id=_SESSION_ID, question="q", evidence_items=[huge])
    assert is_overflow(state, ContextBudgetManager(default_context_window=1000)) is True
    assert is_overflow(state, ContextBudgetManager(default_context_window=1000, ignore_budget=True)) is False
