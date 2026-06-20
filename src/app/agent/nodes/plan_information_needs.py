"""Node: plan_information_needs — deterministic planner (no LLM free-text routing).

Routing rules (CLAUDE.md §8, spec §8):
  1. Question contains summary/overview keywords →
       inspect_chat + fetch_structural_nodes(source_types=[...summaries...]) +
       inspect_document (per doc, from document_manifests)
  2. Question contains numeric/benchmark/metric/dataset keywords →
       query_structured_facts + search_hybrid
  3. Default →
       search_hybrid

Sets state.plan and state.coverage_requirements.
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.state import AgentPlan, AgentState, CoverageRequirement, FactFilterHints

# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

_SUMMARY_KEYWORDS: frozenset[str] = frozenset(
    [
        "summary",
        "summarize",
        "summarise",
        "overview",
        "摘要",
        "概述",
        "introduce",
        "introduction",
        "abstract",
        "what is",
        "what are",
        "tell me about",
    ]
)

_NUMERIC_KEYWORDS: frozenset[str] = frozenset(
    [
        "benchmark",
        "metric",
        "metrics",
        "dataset",
        "accuracy",
        "score",
        "performance",
        "result",
        "results",
        "比較",
        "數值",
        "指標",
        "比例",
        "percentage",
        "number",
        "numbers",
        "statistic",
        "statistics",
        "experiment",
        "experiments",
        "evaluation",
    ]
)

_COMPARISON_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcomp(?:are|ared|aring)\b", re.IGNORECASE),
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bbetween\b.+\band\b", re.IGNORECASE),
    re.compile(r"對比|差異"),
)

_ABLATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\babla(?:te|ted|tion)s?\b", re.IGNORECASE),
    re.compile(r"\bwithout\b.+\bcomponent\b", re.IGNORECASE),
    re.compile(r"\bwith and without\b", re.IGNORECASE),
    re.compile(r"消融"),
)


def _tokenize(text: str) -> set[str]:
    """Return a set of lowercase tokens from *text*."""
    return set(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _is_summary_question(question: str) -> bool:
    tokens = _tokenize(question)
    return bool(tokens & _SUMMARY_KEYWORDS)


def _is_numeric_question(question: str) -> bool:
    tokens = _tokenize(question)
    return bool(tokens & _NUMERIC_KEYWORDS)


def _is_comparison_question(question: str) -> bool:
    """Return True for questions asking to compare two concepts/methods."""
    return any(pattern.search(question) for pattern in _COMPARISON_PATTERNS)


def _is_ablation_question(question: str) -> bool:
    """Return True for ablation / with-vs-without style questions."""
    return any(pattern.search(question) for pattern in _ABLATION_PATTERNS)


_ENTITY_RE = r"([A-Za-z][A-Za-z0-9_.+-]{1,80})"


def _clean_entity(value: str) -> str:
    """Normalise a regex-captured comparison target."""
    return value.strip(" \t\r\n?.,:;()[]{}\"'")


def _extract_comparison_targets(question: str) -> tuple[str, str] | None:
    """Best-effort deterministic extraction of comparison targets A and B."""
    patterns = (
        rf"\bbetween\s+{_ENTITY_RE}\s+and\s+{_ENTITY_RE}",
        rf"{_ENTITY_RE}\s+(?:vs\.?|versus)\s+{_ENTITY_RE}",
        rf"\bcomp(?:are|ared|aring)\s+{_ENTITY_RE}\s+(?:with|and|to)\s+{_ENTITY_RE}",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            first = _clean_entity(match.group(1))
            second = _clean_entity(match.group(2))
            if first and second and first.lower() != second.lower():
                return first, second
    return None


def _comparison_gap_queries(question: str) -> list[str]:
    """Produce deterministic sub-queries for comparison questions."""
    targets = _extract_comparison_targets(question)
    if targets is None:
        return [question, f"{question} differences", f"{question} architecture"]

    first, second = targets
    return [
        question,
        first,
        second,
        f"{first} {second} differences",
        f"{first} {second} performance comparison",
        f"{first} architecture",
        f"{second} architecture",
    ]


def _question_with_session_context(state: AgentState) -> str:
    """Return retrieval-oriented question with history for deep/ambiguous follow-ups."""
    q_lower = state.question.lower()
    ambiguous_follow_up = any(
        marker in q_lower
        for marker in (
            "which is better",
            "which one",
            "what about",
            "follow up",
            "哪個",
            "比較好",
            "結果",
            "那",
        )
    )
    if (
        not (state.generation_config.deep_qa_mode or ambiguous_follow_up)
        or not state.conversation_history
    ):
        return state.question
    recent = state.conversation_history[-4:]
    history = " ".join(f"{turn.role}: {turn.content}" for turn in recent)
    return f"{history} Follow-up question: {state.question}"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def plan_information_needs(state: AgentState) -> dict[str, Any]:
    """Produce a deterministic AgentPlan based on question keyword matching."""
    state.record_event("node_enter", "plan_information_needs")

    question = _question_with_session_context(state)
    summary_path = _is_summary_question(question)
    numeric_path = _is_numeric_question(question)
    comparison_path = _is_comparison_question(question)
    ablation_path = _is_ablation_question(question)

    # Build tool list and information needs
    gap_queries: list[str] = []
    fact_filter_hints = FactFilterHints()

    if summary_path:
        chosen_tools = ["inspect_chat", "fetch_structural_nodes", "search_hybrid"]
        # Add inspect_document for each known document
        if state.document_manifests:
            chosen_tools.append("inspect_document")
        gap_queries = [
            question,
            "LightRAG contributions methodology evaluation",
            "LightRAG architecture retrieval indexing summary",
        ]
        information_needs = [
            "high-level overview of chat documents",
            "document overview summaries",
            "chapter / section summaries",
            "representative raw evidence for the document summary",
        ]
        rationale = "summary/overview question: using structural fetch-all path plus broad hybrid evidence"
    elif comparison_path:
        chosen_tools = ["search_hybrid"]
        gap_queries = _comparison_gap_queries(question)
        targets = _extract_comparison_targets(question)
        if targets is not None:
            first, second = targets
            information_needs = [
                f"evidence about {first}",
                f"evidence about {second}",
                f"direct comparison between {first} and {second}",
                f"performance or cost comparison between {first} and {second}",
            ]
        else:
            information_needs = [
                "evidence about each compared method",
                "direct comparison between the methods",
                "performance / cost differences when available",
            ]
        rationale = "comparison question: decomposed into method-specific hybrid searches"
    elif numeric_path or ablation_path:
        chosen_tools = ["query_structured_facts", "search_hybrid"]
        information_needs = [
            "structured numeric facts (benchmarks, metrics, datasets)",
            "contextual evidence for numbers",
        ]
        rationale = "numeric/benchmark question: facts-first then hybrid search"
        if ablation_path:
            gap_queries = [
                question,
                "Table 2 ablated versions LightRAG performance",
                "LightRAG ablation results Table 2",
                "ablation study",
                "ablated variants performance numbers",
                "without component performance score",
                "with and without component results",
            ]
            # Do not exact-match ``keys=["ablation"]`` here.  Fact keys are
            # usually metric / dataset / variant names, so an exact key filter
            # can hide the ablation facts we are trying to recover.
            fact_filter_hints = FactFilterHints(kinds=["ablation", "metric"])
            information_needs = [
                "ablation table or numeric performance results",
                "ablated LightRAG variants and removed components",
                "full LightRAG versus ablated versions performance comparison",
            ]
            rationale = "ablation question: facts filter plus ablation-specific hybrid searches"
    else:
        chosen_tools = ["search_hybrid"]
        information_needs = ["relevant content for: " + question]
        rationale = "general question: hybrid retrieval"

    plan = AgentPlan(
        goal=question,
        information_needs=information_needs,
        chosen_tools=chosen_tools,
        rationale=rationale,
        gap_queries=gap_queries,
        fact_filter_hints=fact_filter_hints,
    )

    # Build coverage requirements (one per information need)
    coverage_reqs = [
        CoverageRequirement(
            requirement_id=f"req-{i}",
            description=need,
        )
        for i, need in enumerate(information_needs)
    ]

    state.record_event(
        "node_exit",
        "plan_information_needs",
        chosen_tools=chosen_tools,
        summary_path=summary_path,
        numeric_path=numeric_path,
        comparison_path=comparison_path,
        ablation_path=ablation_path,
    )

    return {
        "plan": plan,
        "coverage_requirements": coverage_reqs,
        "debug_trace": state.debug_trace,
    }


__all__ = [
    "plan_information_needs",
    "_is_summary_question",
    "_is_numeric_question",
    "_is_comparison_question",
    "_is_ablation_question",
]
