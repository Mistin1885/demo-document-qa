"""Node: check_coverage — assess whether information needs are satisfied.

For each CoverageRequirement with evidence_ids == []:
  - Compute simple token overlap between requirement.description and
    evidence content.
  - If at least one evidence item has non-zero overlap, mark satisfied=True
    and record matching evidence_ids.

Routing:
  - If any unsatisfied requirements remain AND iteration_count < 2 → "incomplete"
    so graph routes to plan_gap_retrieval.
  - Otherwise → "complete" (we've done our best).

Policy 7 (Phase 7.4):
  - When coverage_state would be "incomplete" AND iteration_count >= 2
    (the cap), we set coverage_state="complete" so the graph proceeds to
    verify_critical_claims.  The generate_answer node enforces policy 7
    (incomplete at cap → return no-info answer) via PolicyEngine.enforce_pre_answer.
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.budget import COVERAGE_SIMILARITY_THRESHOLD
from app.agent.state import AgentState, CoverageRequirement


def _token_overlap(text_a: str, text_b: str) -> int:
    """Return count of shared lowercase tokens between two strings."""
    tokens_a = set(re.sub(r"[^\w]", " ", text_a.lower()).split())
    tokens_b = set(re.sub(r"[^\w]", " ", text_b.lower()).split())
    # Ignore very short stop-words
    stop = {"the", "a", "an", "of", "in", "for", "and", "or", "to", "is", "are", "that"}
    return len((tokens_a - stop) & (tokens_b - stop))


_SEMANTIC_EQUIVALENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ablation", ("without component", "with and without", "removed", "ablated")),
    ("performance", ("result", "results", "score", "accuracy", "f1", "benchmark")),
    ("formula", ("equation", "latex", "mathematical expression")),
    ("image", ("figure", "plot", "diagram", "caption")),
    ("table", ("tabular", "row", "column")),
)


def _semantic_equivalent_match(requirement: str, content: str) -> bool:
    """Small deterministic fallback for common paper-QA synonyms."""
    req = requirement.lower()
    ev = content.lower()
    for anchor, variants in _SEMANTIC_EQUIVALENTS:
        if anchor in req and any(v in ev for v in variants):
            return True
        if any(v in req for v in variants) and anchor in ev:
            return True
    return False


def _contains_numeric_or_table_signal(text: str) -> bool:
    lower = text.lower()
    return bool(re.search(r"\b\d+(?:\.\d+)?%?\b", text)) or any(
        marker in lower
        for marker in (
            "table",
            "score",
            "scores",
            "win rate",
            "f1",
            "accuracy",
            "metric",
            "dataset",
            "performance decline",
        )
    )


def _comparison_targets_from_requirement(requirement: str) -> tuple[str, str] | None:
    patterns = (
        r"between\s+([A-Za-z][A-Za-z0-9_.+-]{1,80})\s+and\s+([A-Za-z][A-Za-z0-9_.+-]{1,80})",
        r"([A-Za-z][A-Za-z0-9_.+-]{1,80})\s+versus\s+([A-Za-z][A-Za-z0-9_.+-]{1,80})",
        r"([A-Za-z][A-Za-z0-9_.+-]{1,80})\s+vs\.?\s+([A-Za-z][A-Za-z0-9_.+-]{1,80})",
    )
    for pattern in patterns:
        match = re.search(pattern, requirement, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower(), match.group(2).lower()
    return None


def _comparison_targets_from_question(question: str) -> tuple[str, str] | None:
    patterns = (
        r"\bbetween\s+([A-Za-z][A-Za-z0-9_.+-]{1,80})\s+and\s+([A-Za-z][A-Za-z0-9_.+-]{1,80})",
        r"([A-Za-z][A-Za-z0-9_.+-]{1,80})\s+(?:vs\.?|versus)\s+([A-Za-z][A-Za-z0-9_.+-]{1,80})",
        r"\bcomp(?:are|ared|aring)\s+([A-Za-z][A-Za-z0-9_.+-]{1,80})\s+(?:with|and|to)\s+([A-Za-z][A-Za-z0-9_.+-]{1,80})",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower(), match.group(2).lower()
    return None


def _workspace_texts(state: AgentState) -> list[str]:
    texts = [ev.content for ev in state.evidence_items]
    for fact in state.structured_facts:
        value = str(fact.value_numeric) if fact.value_numeric is not None else fact.value_text or ""
        texts.append(
            " ".join(
                part
                for part in (
                    fact.kind,
                    fact.key,
                    value,
                    fact.unit or "",
                    fact.context_excerpt or "",
                )
                if part
            )
        )
    return texts


def _audit_specialized_coverage(state: AgentState) -> CoverageRequirement | None:
    """Return an unsatisfied audit requirement when retrieved evidence is too shallow."""
    if state.plan is None:
        return None

    rationale = state.plan.rationale.lower()
    texts = _workspace_texts(state)
    lowered = [text.lower() for text in texts]

    if "comparison question" in rationale:
        targets = _comparison_targets_from_question(state.plan.goal)
        if targets is None:
            return None
        first, second = targets
        has_first = any(first in text for text in lowered)
        has_second = any(second in text for text in lowered)
        has_direct = any(first in text and second in text for text in lowered)
        if not (has_first and has_second and has_direct):
            return CoverageRequirement(
                requirement_id="audit-comparison",
                description=f"direct comparison evidence containing both {first} and {second}",
            )

    if "ablation question" in rationale:
        has_ablation = any(
            any(marker in text for marker in ("ablat", "without", "removed", "-low", "-high", "-origin", "table 2"))
            for text in lowered
        )
        has_results = any(_contains_numeric_or_table_signal(text) for text in texts)
        if not (has_ablation and has_results):
            return CoverageRequirement(
                requirement_id="audit-ablation",
                description="ablation table or numeric performance results for ablated variants",
            )

    return None


def _score_requirement(requirement: str, content: str, vector_score: float | None) -> float:
    """Return a semantic coverage score for one requirement/evidence pair.

    Preferred signal is Vespa's native vector similarity.  When unavailable,
    fall back to lexical overlap and a compact synonym table for paper QA
    terminology such as ``ablation`` ↔ ``without component``.
    """
    req_lower = requirement.lower()
    content_lower = content.lower()

    comparison_targets = _comparison_targets_from_requirement(requirement)
    if comparison_targets is not None:
        first, second = comparison_targets
        if first in content_lower and second in content_lower:
            if "performance" in req_lower or "cost" in req_lower:
                return 1.0 if _contains_numeric_or_table_signal(content) else 0.0
            return 1.0
        return 0.0

    if "evidence about " in req_lower:
        target = req_lower.split("evidence about ", 1)[1].strip()
        if target and target in content_lower:
            return 1.0
        return 0.0

    if "ablation" in req_lower or "ablated" in req_lower:
        needs_numeric = any(
            marker in req_lower
            for marker in ("table", "numeric", "performance", "comparison", "versions")
        )
        if not needs_numeric and vector_score is not None and vector_score >= COVERAGE_SIMILARITY_THRESHOLD:
            return vector_score
        has_ablation_signal = any(
            marker in content_lower
            for marker in ("ablat", "without", "removed", "removing", "-low", "-high", "-origin", "table 2")
        )
        if not has_ablation_signal:
            return 0.0
        if needs_numeric and not _contains_numeric_or_table_signal(content):
            return 0.0
        return 1.0

    if vector_score is not None and vector_score >= COVERAGE_SIMILARITY_THRESHOLD:
        return vector_score
    if _token_overlap(requirement, content) > 0:
        return 0.4
    if _semantic_equivalent_match(requirement, content):
        return COVERAGE_SIMILARITY_THRESHOLD
    return 0.0


async def check_coverage(state: AgentState) -> dict[str, Any]:
    """Mark coverage requirements satisfied/unsatisfied based on evidence."""
    state.record_event("node_enter", "check_coverage")

    updated_reqs: list[CoverageRequirement] = []

    for req in state.coverage_requirements:
        if req.satisfied:
            updated_reqs.append(req)
            continue

        # Find evidence that overlaps with this requirement
        matched_ids: list[str] = []
        for ev in state.evidence_items:
            if _score_requirement(req.description, ev.content, ev.vector_score) > 0:
                matched_ids.append(ev.evidence_id)

        if matched_ids:
            updated_reqs.append(
                CoverageRequirement(
                    requirement_id=req.requirement_id,
                    description=req.description,
                    evidence_ids=matched_ids[:5],  # cap to 5 per requirement
                    satisfied=True,
                )
            )
        else:
            updated_reqs.append(req)

    # Determine new coverage state
    audit_req = _audit_specialized_coverage(
        state.model_copy(update={"coverage_requirements": updated_reqs})
    )
    if audit_req is not None and not any(r.requirement_id == audit_req.requirement_id for r in updated_reqs):
        updated_reqs.append(audit_req)

    has_unsatisfied = any(not r.satisfied for r in updated_reqs)

    # Route to gap retrieval only if we have budget (iteration_count < 2)
    if has_unsatisfied and state.iteration_count < 2:
        new_coverage_state = "incomplete"
    else:
        # At or past cap: force "complete" so graph proceeds.
        # Policy 7 in generate_answer will enforce the no-info fallback
        # if coverage is still insufficient.
        new_coverage_state = "complete"

    state.record_event(
        "node_exit",
        "check_coverage",
        satisfied=sum(1 for r in updated_reqs if r.satisfied),
        unsatisfied=sum(1 for r in updated_reqs if not r.satisfied),
        coverage_state=new_coverage_state,
    )

    return {
        "coverage_requirements": updated_reqs,
        "coverage_state": new_coverage_state,  # type: ignore[dict-item]
        "debug_trace": state.debug_trace,
    }


__all__ = ["check_coverage", "_score_requirement", "_token_overlap"]
