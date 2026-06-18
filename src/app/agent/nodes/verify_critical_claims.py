"""Node: verify_critical_claims — cross-check numeric tokens in answer draft.

If state.answer is already set, scans it for numeric tokens
(regex: \\b\\d+(\\.\\d+)?%?\\b) and verifies each against evidence / facts.

Violations are recorded as AgentError(code="UNVERIFIED_CLAIM") — they do NOT
hard-fail the graph (CLAUDE.md §8 spec).
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.state import AgentError, AgentState

_NUMERIC_RE = re.compile(r"\b\d+(\.\d+)?%?\b")


def _find_numeric_tokens(text: str) -> list[str]:
    """Return all unique numeric token strings from *text*."""
    return list({m.group(0) for m in _NUMERIC_RE.finditer(text)})


def _has_evidence_for_token(token: str, state: AgentState) -> bool:
    """Return True if *token* appears in any evidence item or structured fact."""
    for ev in state.evidence_items:
        if token in ev.content:
            return True
    for fact in state.structured_facts:
        if token in (fact.value_text or "") or token == str(fact.value_numeric or ""):
            return True
    return False


async def verify_critical_claims(state: AgentState) -> dict[str, Any]:
    """Check numeric claims in answer draft (if present) against evidence."""
    state.record_event("node_enter", "verify_critical_claims")

    source_text = state.answer or state.question
    numeric_tokens = _find_numeric_tokens(source_text)

    new_errors: list[AgentError] = list(state.errors)

    unverified_count = 0
    for token in numeric_tokens:
        if not _has_evidence_for_token(token, state):
            new_errors.append(
                AgentError(
                    code="UNVERIFIED_CLAIM",
                    detail=f"Numeric token '{token}' not backed by evidence/facts",
                )
            )
            unverified_count += 1

    state.record_event(
        "node_exit",
        "verify_critical_claims",
        numeric_tokens_checked=len(numeric_tokens),
        unverified=unverified_count,
    )

    return {
        "errors": new_errors,
        "debug_trace": state.debug_trace,
    }


__all__ = ["verify_critical_claims"]
