"""ContextBudgetManager — token-budget accounting for the Paper Notebook Agent.

Design rules (CLAUDE.md §8, §14, §12)
---------------------------------------
- Pure Python — no async, no network calls, no LangGraph imports.
- tiktoken (cl100k_base) is used when available; on ``ImportError`` the
  fallback is ``max(1, len(text) // 4)`` and ``last_was_estimate`` is set.
- All methods are stateless / functional-style; the manager holds only
  configuration, not mutable state.
- ``dict[str, Any]`` is NOT used; all parameters and returns use typed models.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agent.state import AgentState, ConversationTurn, EvidenceItem, ToolCallRecord

MAX_REPLAN_ROUNDS = 3
COVERAGE_SIMILARITY_THRESHOLD = 0.55


@runtime_checkable
class _TiktokenEncoder(Protocol):
    """Minimal protocol for a tiktoken Encoding object."""

    def encode(self, text: str, **kwargs: Any) -> list[int]: ...


# ---------------------------------------------------------------------------
# ContextAllocation — budget split configuration
# ---------------------------------------------------------------------------


class ContextAllocation(BaseModel):
    """Token budget allocation across different content categories.

    Default fields sum to 10,000.  When ``ContextBudgetManager`` is constructed
    with a different ``default_context_window`` and no explicit allocation,
    these defaults are scaled proportionally via :meth:`scaled_to`.

    The validator only ensures fields are non-negative.  Bespoke allocations
    may sum to any positive number — the manager caps queries against
    ``default_context_window`` independently.
    """

    model_config = ConfigDict(extra="forbid")

    system_and_tools: int = Field(default=1_200, ge=0)
    conversation: int = Field(default=1_000, ge=0)
    question_and_plan: int = Field(default=500, ge=0)
    document_evidence: int = Field(default=5_000, ge=0)
    answer_reserve: int = Field(default=2_000, ge=0)
    miscellaneous: int = Field(default=300, ge=0)

    @property
    def total(self) -> int:
        return (
            self.system_and_tools
            + self.conversation
            + self.question_and_plan
            + self.document_evidence
            + self.answer_reserve
            + self.miscellaneous
        )

    @model_validator(mode="after")
    def _check_positive(self) -> ContextAllocation:
        if self.total <= 0:
            raise ValueError(
                f"ContextAllocation fields must sum to a positive integer; got {self.total}."
            )
        return self

    @classmethod
    def scaled_to(cls, window: int) -> ContextAllocation:
        """Return the default allocation rescaled to sum to ``window``.

        Uses integer arithmetic and absorbs the rounding remainder into
        ``document_evidence`` so the total exactly matches ``window``.
        """
        if window <= 0:
            raise ValueError("window must be a positive integer")
        base = cls()
        base_total = base.total
        if window == base_total:
            return base
        ratio = window / base_total
        s = int(base.system_and_tools * ratio)
        c = int(base.conversation * ratio)
        q = int(base.question_and_plan * ratio)
        a = int(base.answer_reserve * ratio)
        m = int(base.miscellaneous * ratio)
        # document_evidence soaks the rounding remainder
        d = window - s - c - q - a - m
        return cls(
            system_and_tools=s,
            conversation=c,
            question_and_plan=q,
            document_evidence=d,
            answer_reserve=a,
            miscellaneous=m,
        )


# ---------------------------------------------------------------------------
# ContextBudgetManager
# ---------------------------------------------------------------------------


class ContextBudgetManager:
    """Token-budget manager for the Paper Notebook Agent.

    Parameters
    ----------
    default_context_window:
        Total token budget (from the provider profile).  Defaults to 10,000.
    allocation:
        Per-category budget split.  Defaults to ``ContextAllocation()``.

    Attributes
    ----------
    last_was_estimate:
        Set to ``True`` after a ``count_tokens`` call that fell back to the
        heuristic (tiktoken unavailable).  Callers should propagate this to
        ``AgentState.token_count_is_estimate``.
    """

    def __init__(
        self,
        default_context_window: int = 10_000,
        allocation: ContextAllocation | None = None,
        *,
        ignore_budget: bool = False,
    ) -> None:
        self.default_context_window = default_context_window
        self.ignore_budget = ignore_budget
        if allocation is None:
            self.allocation = ContextAllocation.scaled_to(default_context_window)
        else:
            self.allocation = allocation
        self.last_was_estimate: bool = False
        self._tiktoken_enc: _TiktokenEncoder | None = _load_tiktoken()

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Count tokens in *text*.

        Uses tiktoken ``cl100k_base`` when available; otherwise falls back to
        ``max(1, len(text) // 4)`` (heuristic) and sets ``last_was_estimate``.
        Empty string always returns 0 (no estimate flag set).
        """
        if not text:
            return 0

        if self._tiktoken_enc is not None:
            try:
                result = self._tiktoken_enc.encode(text)
                self.last_was_estimate = False
                return len(result)
            except Exception:
                pass

        # Fallback heuristic
        self.last_was_estimate = True
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # Estimation helpers
    # ------------------------------------------------------------------

    def estimate_tool_result(self, result: ToolCallRecord, evidence: list[EvidenceItem]) -> int:
        """Estimate tokens for one tool result.

        If ``result.token_estimate > 0`` it is used directly (avoids
        re-tokenising large evidence blobs).  Otherwise, the content of all
        evidence items matching ``result.call_id`` is summed.

        When *evidence* is empty or no items match, returns ``result.token_estimate``.
        """
        if result.token_estimate > 0:
            return result.token_estimate

        # Derive from evidence content
        relevant = [ev for ev in evidence if ev.origin_tool == result.tool_name]
        if not relevant:
            return 0
        return sum(self.count_tokens(ev.content) for ev in relevant)

    def estimate_conversation(self, turns: list[ConversationTurn]) -> int:
        """Estimate total tokens across all conversation turns."""
        if not turns:
            return 0
        return sum(self.count_tokens(t.content) for t in turns)

    # ------------------------------------------------------------------
    # Budget calculations
    # ------------------------------------------------------------------

    def calculate_available_evidence_budget(self, state: AgentState) -> int:
        """Return remaining token budget for document evidence.

        ``max(0, allocation.document_evidence - current_evidence_tokens)``
        """
        used = sum(self.count_tokens(ev.content) for ev in state.evidence_items)
        available = self.allocation.document_evidence - used
        return max(0, available)

    def detect_overflow(self, state: AgentState) -> bool:
        """Return True if current usage would exceed the safe generation budget.

        Overflow condition:
            evidence_tokens + history_tokens + plan_tokens
            > context_window − answer_reserve
        """
        if self.ignore_budget:
            return False
        evidence_tokens = sum(self.count_tokens(ev.content) for ev in state.evidence_items)
        history_tokens = self.estimate_conversation(state.conversation_history)
        plan_tokens = (
            self.count_tokens(state.plan.goal) + self.count_tokens(state.plan.rationale)
            if state.plan is not None
            else 0
        )
        used = evidence_tokens + history_tokens + plan_tokens
        safe_ceiling = self.default_context_window - self.allocation.answer_reserve
        return used > safe_ceiling

    def select_compact_sources(self, state: AgentState, target_budget: int) -> list[EvidenceItem]:
        """Greedy evidence compaction within *target_budget* tokens.

        Selects items in descending score order (None score treated as 0.0),
        then ascending token count as tiebreaker.

        Guarantees at least one item per unique ``document_id`` is retained,
        even if the per-document item alone exceeds the budget.
        """
        items = list(state.evidence_items)

        # Ensure at least-one-per-document by pre-selecting the
        # highest-score item for each document_id.
        by_doc: dict[str, EvidenceItem] = {}
        for item in sorted(items, key=lambda x: x.score or 0.0, reverse=True):
            doc_key = str(item.document_id)
            if doc_key not in by_doc:
                by_doc[doc_key] = item

        must_include = set(id(v) for v in by_doc.values())

        # Sort all items: score desc, token_count asc (proxy: len(content))
        sorted_items = sorted(
            items,
            key=lambda x: (-(x.score or 0.0), len(x.content)),
        )

        selected: list[EvidenceItem] = []
        used_tokens = 0

        # First pass: always add must-include items
        for item in sorted_items:
            if id(item) in must_include:
                selected.append(item)
                used_tokens += self.count_tokens(item.content)

        # Second pass: fill remaining budget greedily
        for item in sorted_items:
            if id(item) in must_include:
                continue  # already added
            item_tokens = self.count_tokens(item.content)
            if used_tokens + item_tokens <= target_budget:
                selected.append(item)
                used_tokens += item_tokens

        return selected

    def build_aggregation_groups(self, state: AgentState) -> list[list[EvidenceItem]]:
        """Group evidence items by ``(document_id, section_title)`` for aggregation.

        Used by the ``aggregate_sources`` tool when overflow is detected.
        Groups preserve insertion order within each bucket.
        """
        buckets: dict[tuple[str, str | None], list[EvidenceItem]] = {}
        for item in state.evidence_items:
            key = (str(item.document_id), item.section_title)
            buckets.setdefault(key, []).append(item)
        return list(buckets.values())


# ---------------------------------------------------------------------------
# Internal: tiktoken loader
# ---------------------------------------------------------------------------


def _load_tiktoken() -> _TiktokenEncoder | None:
    """Try to load tiktoken encoder; return None on ImportError."""
    try:
        import tiktoken  # noqa: PLC0415

        enc = tiktoken.get_encoding("cl100k_base")
        if isinstance(enc, _TiktokenEncoder):
            return enc
        return None
    except (ImportError, Exception):
        return None


__all__ = [
    "COVERAGE_SIMILARITY_THRESHOLD",
    "MAX_REPLAN_ROUNDS",
    "ContextAllocation",
    "ContextBudgetManager",
]
