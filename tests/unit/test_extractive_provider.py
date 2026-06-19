from __future__ import annotations

import pytest

from app.providers.base import ChatMessage
from app.providers.extractive import ExtractiveEvidenceChatProvider


@pytest.mark.asyncio
async def test_extractive_provider_returns_cited_evidence_answer() -> None:
    provider = ExtractiveEvidenceChatProvider()
    prompt = """Evidence excerpts:
[c1] (page 1–1, Intro)
LightRAG integrates graph-based text indexing with a dual-level retrieval framework.

[c2] (page 2–2, Other)
The system also discusses unrelated implementation details.

Question: What is LightRAG?
"""

    result = await provider.complete([ChatMessage(role="user", content=prompt)])

    assert result.model == "extractive-evidence-v1"
    assert "LightRAG integrates graph-based text indexing" in result.content
    assert "[c1]" in result.content
    assert "mock" not in result.content.lower()


@pytest.mark.asyncio
async def test_extractive_provider_no_evidence_fallback() -> None:
    provider = ExtractiveEvidenceChatProvider()
    result = await provider.complete([ChatMessage(role="user", content="Question: What is it?")])
    assert "not enough information" in result.content.lower()
