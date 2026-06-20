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


@pytest.mark.asyncio
async def test_extractive_provider_answers_origin_ablation_not_always_hurt() -> None:
    provider = ExtractiveEvidenceChatProvider()
    prompt = """Evidence excerpts:
[c1] (page 1–1, Table 2)
<table><caption>Table 2 Performance of ablated versions of LightRAG</caption><tr><th>Variant</th><th>Agriculture Overall</th><th>Legal Overall</th></tr><tr><td>Full LightRAG</td><td>67.6%</td><td>84.8%</td></tr><tr><td>-Origin</td><td>74.4%</td><td>84.4%</td></tr></table>

Question: Does removing original text always hurt LightRAG in the reported ablation?
"""

    result = await provider.complete([ChatMessage(role="user", content=prompt)])

    assert "No." in result.content
    assert "-Origin" in result.content
    assert "74.4%" in result.content
    assert "84.4%" in result.content
