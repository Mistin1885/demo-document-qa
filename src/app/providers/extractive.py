"""Evidence-grounded fallback chat provider.

This provider is intentionally not an LLM and not a mock.  It is used when the
system has retrieved evidence but no external chat model is configured.  The
answer is extractive: it selects the most question-relevant evidence excerpts
from the prompt and returns short cited snippets using the same ``[cN]`` marker
format that the agent citation validator understands.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.providers.base import (
    ChatChunk,
    ChatCompletion,
    ChatMessage,
    ChatProvider,
    ProviderTestResult,
    Usage,
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_EVIDENCE_RE = re.compile(
    r"\[(c\d+)\]\s*\([^)]*\)\s*\n(?P<content>.*?)(?=\n\n\[c\d+\]|\n\nQuestion:|\Z)",
    re.DOTALL,
)
_QUESTION_RE = re.compile(r"\n\nQuestion:\s*(?P<question>.*)\Z", re.DOTALL)
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "what",
        "which",
        "does",
        "into",
        "about",
        "using",
        "used",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "how",
        "why",
        "who",
        "when",
        "where",
        "solve",
        "problem",
    }
)


@dataclass(frozen=True)
class _EvidenceExcerpt:
    marker: str
    content: str
    score: int


class ExtractiveEvidenceChatProvider(ChatProvider):
    """Deterministic evidence-only answer provider.

    The provider expects the standard ``generate_answer`` prompt containing
    evidence blocks like ``[c1] ...`` and a trailing ``Question: ...`` line.
    It never uses prior knowledge and returns the no-information fallback if no
    evidence is present.
    """

    def __init__(self, *, context_window: int = 16_384) -> None:
        self._context_window = context_window

    @property
    def name(self) -> str:
        return "extractive_evidence"

    @property
    def model(self) -> str:
        return "extractive-evidence-v1"

    @property
    def context_window(self) -> int:
        return self._context_window

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> ChatCompletion:
        prompt = "\n\n".join(m.content for m in messages if m.role == "user")
        answer = _build_extractive_answer(prompt, max_chars=max(800, max_tokens * 4))
        return ChatCompletion(
            content=answer,
            usage=Usage(
                prompt_tokens=max(1, len(prompt.split())),
                completion_tokens=max(1, len(answer.split())),
            ),
            model=self.model,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        completion = await self.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        words = completion.content.split()
        for i, word in enumerate(words):
            yield ChatChunk(
                delta=word if i == 0 else f" {word}",
                finish_reason="stop" if i == len(words) - 1 else None,
            )

    async def test_connection(self) -> ProviderTestResult:
        return ProviderTestResult(ok=True, model=self.model, latency_ms=0)


def _terms(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS}


def _compact_excerpt(text: str, *, max_chars: int = 360) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    # Prefer cutting at sentence-ish punctuation before the limit.
    cut = max(cleaned.rfind(". ", 0, max_chars), cleaned.rfind("; ", 0, max_chars))
    if cut < 120:
        cut = max_chars
    return cleaned[:cut].rstrip(" .;") + "…"


def _extract_evidence(prompt: str) -> tuple[str, list[_EvidenceExcerpt]]:
    q_match = _QUESTION_RE.search(prompt)
    question = q_match.group("question").strip() if q_match else ""
    q_terms = _terms(question)

    excerpts: list[_EvidenceExcerpt] = []
    for match in _EVIDENCE_RE.finditer(prompt):
        marker = match.group(1)
        content = match.group("content").strip()
        score = len(q_terms & _terms(content))
        excerpts.append(_EvidenceExcerpt(marker=marker, content=content, score=score))
    return question, excerpts


def _build_extractive_answer(prompt: str, *, max_chars: int) -> str:
    _question, excerpts = _extract_evidence(prompt)
    if not excerpts:
        return "There is not enough information in the current chat's documents to answer this question."

    ranked = sorted(excerpts, key=lambda ev: (ev.score, -int(ev.marker[1:])), reverse=True)
    selected = [ev for ev in ranked if ev.score > 0][:3] or ranked[:2]

    parts = ["Based on the retrieved document excerpts:"]
    budget_left = max_chars - len(parts[0])
    for ev in selected:
        snippet = _compact_excerpt(ev.content, max_chars=min(360, max(160, budget_left // 2)))
        sentence = f"- {snippet} [{ev.marker}]"
        if len(sentence) > budget_left and len(parts) > 1:
            break
        parts.append(sentence)
        budget_left -= len(sentence)
        if budget_left <= 120:
            break

    return "\n".join(parts)


__all__ = ["ExtractiveEvidenceChatProvider"]
