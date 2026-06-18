"""AgentState — Pydantic v2 schema for the LangGraph Paper Notebook Agent.

Design rules (CLAUDE.md §8, §12, §13.1)
-----------------------------------------
- ``extra="forbid"`` on every model — no silent data loss.
- ``chat_id`` and ``session_id`` are **never** set by the LLM; they are
  injected by the service layer from the authenticated request context.
- ``tool_invocations_fingerprints`` is serialised as ``list[str]`` (JSON
  round-trip safe) and rebuilt as ``set[str]`` via a ``model_validator``.
- ``dict[str, Any]`` is allowed **only** in ``ToolCallRecord.params`` and
  ``TraceEvent.payload`` — these are contract-boundary fields whose schema
  cannot be statically defined (tool parameters vary per tool; trace events
  carry heterogeneous debug data).  The usage is documented here.
- No FastAPI, LangGraph, or network I/O imports.
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Namespace for deterministic evidence_id generation (uuid5)
# ---------------------------------------------------------------------------
_EVIDENCE_NS = uuid.NAMESPACE_OID


# ---------------------------------------------------------------------------
# ConversationTurn
# ---------------------------------------------------------------------------


class ConversationTurn(BaseModel):
    """One turn in the conversation history."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str
    message_id: UUID | None = None
    created_at: datetime | None = None


# ---------------------------------------------------------------------------
# ChatManifestSnapshot — agent-internal lightweight snapshot
# ---------------------------------------------------------------------------


class ChatManifestSnapshot(BaseModel):
    """Agent-internal snapshot of the chat manifest read at request time.

    Derived from ``ChatManifest.model_dump()`` but carries only the fields
    the agent needs for routing decisions — avoids dragging in heavy
    enrichment models.
    """

    model_config = ConfigDict(extra="forbid")

    document_count: int = Field(ge=0)
    document_titles: list[str] = Field(default_factory=list)
    main_topics: list[str] = Field(default_factory=list)
    total_token_estimate: int = Field(ge=0, default=0)
    source_types: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# DocumentManifest — per-document snapshot
# ---------------------------------------------------------------------------


class DocumentManifest(BaseModel):
    """Agent-internal snapshot of one document's manifest data."""

    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    title: str | None = None
    abstract_summary: str | None = None
    section_count: int = Field(ge=0, default=0)
    page_count: int | None = None
    source_types: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# AgentPlan
# ---------------------------------------------------------------------------


class AgentPlan(BaseModel):
    """The agent's information-retrieval plan produced by ``plan_information_needs``."""

    model_config = ConfigDict(extra="forbid")

    goal: str
    information_needs: list[str] = Field(default_factory=list)
    chosen_tools: list[str] = Field(default_factory=list)
    rationale: str = ""


# ---------------------------------------------------------------------------
# ToolCallRecord
# ---------------------------------------------------------------------------


class ToolCallRecord(BaseModel):
    """Record of one tool invocation.

    ``params`` uses ``dict[str, Any]`` because tool parameters vary per tool
    and cannot be statically schema-ized at the AgentState boundary.  This is
    an intentional, documented exception to the CLAUDE.md §12 no-unbounded-dict
    rule — all other state fields use typed Pydantic models.
    """

    model_config = ConfigDict(extra="forbid")

    call_id: str
    tool_name: str
    # Intentional dict[str, Any]: tool params are heterogeneous (see module docstring).
    params: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "overflow", "error", "empty"] = "ok"
    token_estimate: int = Field(ge=0, default=0)
    source_count: int = Field(ge=0, default=0)
    error: str | None = None


# ---------------------------------------------------------------------------
# EvidenceItem
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """One piece of retrieved evidence stored in the agent workspace.

    ``evidence_id`` is deterministic:
        ``str(uuid5(NAMESPACE_OID, f"{origin_tool}:{source_node_id}:{document_id}"))``
    This ensures ``add_evidence`` can de-duplicate by ID without extra bookkeeping.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    """Deterministic UUID-5 string: uuid5(NAMESPACE_OID, f'{origin_tool}:{source_node_id}:{document_id}')."""
    source_type: str
    document_id: UUID
    source_node_id: str
    page_start: int = Field(ge=0)
    page_end: int = Field(ge=0)
    content: str
    score: float | None = None
    section_title: str | None = None
    heading_path: str | None = None
    origin_tool: str


def make_evidence_id(origin_tool: str, source_node_id: str, document_id: str | UUID) -> str:
    """Return the canonical deterministic evidence_id string."""
    return str(uuid5(_EVIDENCE_NS, f"{origin_tool}:{source_node_id}:{document_id}"))


# ---------------------------------------------------------------------------
# StructuredFactSnapshot
# ---------------------------------------------------------------------------


class StructuredFactSnapshot(BaseModel):
    """Snapshot of a StructuredFact row read by the agent."""

    model_config = ConfigDict(extra="forbid")

    fact_id: UUID
    document_id: UUID
    kind: str
    key: str
    value_numeric: float | None = None
    value_text: str | None = None
    unit: str | None = None
    page: int | None = None
    context_excerpt: str | None = None


# ---------------------------------------------------------------------------
# CoverageRequirement
# ---------------------------------------------------------------------------


class CoverageRequirement(BaseModel):
    """One information-coverage requirement derived from the plan."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    description: str
    evidence_ids: list[str] = Field(default_factory=list)
    """Evidence IDs that satisfy this requirement; empty means not yet satisfied."""
    satisfied: bool = False


# ---------------------------------------------------------------------------
# CitationDraft
# ---------------------------------------------------------------------------


class CitationDraft(BaseModel):
    """Draft citation built by the agent before final validation.

    All citation_id / chat_id / document_id values are injected from retrieved
    evidence — never set by the LLM directly.
    """

    model_config = ConfigDict(extra="forbid")

    citation_id: str
    chat_id: UUID
    document_id: UUID
    document_name: str
    page_start: int = Field(ge=0)
    page_end: int = Field(ge=0)
    section_title: str | None = None
    source_node_id: str
    excerpt: str


# ---------------------------------------------------------------------------
# AgentError
# ---------------------------------------------------------------------------


class AgentError(BaseModel):
    """A non-fatal error recorded during agent execution."""

    model_config = ConfigDict(extra="forbid")

    code: str
    detail: str
    tool_name: str | None = None


# ---------------------------------------------------------------------------
# Debug trace
# ---------------------------------------------------------------------------


class GenerationConfig(BaseModel):
    """Per-request overrides for the LLM call in ``generate_answer``.

    All fields are optional; ``None`` means "use provider/env default".
    These are propagated by the service layer from the request body, never
    set by the LLM or the agent itself.
    """

    model_config = ConfigDict(extra="forbid")

    max_answer_tokens: int | None = Field(default=None, ge=1, le=32_768)
    """Cap on output tokens for the answer call (maps to ``complete(max_tokens=...)``)."""

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    """Sampling temperature for the answer call."""

    context_window: int | None = Field(default=None, ge=1_000, le=200_000)
    """Total input token budget — drives ``ContextBudgetManager`` size."""


class TraceEvent(BaseModel):
    """One append-only event in the agent debug trace.

    ``payload`` uses ``dict[str, Any]`` because trace events carry heterogeneous
    debug data (node names, token counts, policy results, etc.) whose schema
    differs per event kind.  This is a documented exception (see module docstring).
    """

    model_config = ConfigDict(extra="forbid")

    ts: datetime
    kind: Literal["node_enter", "node_exit", "tool_call", "policy_violation", "budget_check"]
    name: str
    # Intentional dict[str, Any]: trace payload is heterogeneous (see module docstring).
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentDebugTrace(BaseModel):
    """Append-only list of trace events for the current agent run."""

    model_config = ConfigDict(extra="forbid")

    events: list[TraceEvent] = Field(default_factory=list)

    def record(
        self,
        kind: Literal["node_enter", "node_exit", "tool_call", "policy_violation", "budget_check"],
        name: str,
        **payload: Any,
    ) -> None:
        """Append a new TraceEvent with the current UTC timestamp."""
        self.events.append(
            TraceEvent(
                ts=datetime.now(UTC),
                kind=kind,
                name=name,
                payload=dict(payload),
            )
        )


# ---------------------------------------------------------------------------
# AgentState — top-level state schema
# ---------------------------------------------------------------------------


class AgentState(BaseModel):
    """Canonical LangGraph agent state for the Paper Notebook Agent.

    Isolation contract (CLAUDE.md §2, §8)
    ---------------------------------------
    - ``chat_id`` is **always** injected by the service layer; the LLM/agent
      cannot set or override it.
    - ``session_id`` is likewise injected; Sessions are isolated from each other.
    - All tool results must be scoped to ``chat_id``; tools receive it from
      this state, not from LLM-supplied arguments.

    Serialisation notes
    -------------------
    - ``tool_invocations_fingerprints`` is stored as ``set[str]`` in-memory
      but serialised as ``list[str]`` (JSON round-trip safe).
    - Use ``model_dump()`` for serialisation; use ``AgentState.model_validate()``
      to deserialise (the ``model_validator`` rebuilds the set from the list).
    """

    model_config = ConfigDict(extra="forbid")

    # --- Core identity (service-injected; never LLM-controlled) ---
    chat_id: UUID
    session_id: UUID
    question: str = Field(min_length=1)

    # --- Conversation history ---
    conversation_history: list[ConversationTurn] = Field(default_factory=list)

    # --- Chat / document context ---
    chat_manifest: ChatManifestSnapshot | None = None
    document_manifests: list[DocumentManifest] = Field(default_factory=list)

    # --- Planning ---
    plan: AgentPlan | None = None

    # --- Tool execution records ---
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)

    # --- Evidence workspace ---
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    structured_facts: list[StructuredFactSnapshot] = Field(default_factory=list)

    # --- Coverage tracking ---
    coverage_requirements: list[CoverageRequirement] = Field(default_factory=list)
    coverage_state: Literal["pending", "incomplete", "complete"] = "pending"

    # --- Budget tracking ---
    context_token_estimate: int = Field(ge=0, default=0)
    remaining_budget: int = Field(ge=0, default=10_000)

    # --- Iteration control ---
    iteration_count: int = Field(ge=0, default=0)

    # --- Duplicate-tool-call detection ---
    # Stored in-memory as set[str]; serialised as list[str].
    # The model_validator below rebuilds the set after deserialisation.
    tool_invocations_fingerprints: set[str] = Field(default_factory=set)

    # --- Answer + citations ---
    answer: str | None = None
    citations: list[CitationDraft] = Field(default_factory=list)

    # --- Errors ---
    errors: list[AgentError] = Field(default_factory=list)

    # --- Debug trace ---
    debug_trace: AgentDebugTrace = Field(default_factory=AgentDebugTrace)

    # --- Token estimate flag (GUIDE §14) ---
    token_count_is_estimate: bool = False

    # --- Per-request generation overrides (from MessageRequest body) ---
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)

    # ------------------------------------------------------------------
    # Serialisation shim: set[str] ↔ list[str]
    # ------------------------------------------------------------------

    @field_validator("tool_invocations_fingerprints", mode="before")
    @classmethod
    def _coerce_fingerprints(cls, v: object) -> set[str]:
        """Accept list[str] (from JSON) or set[str] (in-memory)."""
        if isinstance(v, list):
            return set(v)
        if isinstance(v, set):
            return v
        raise ValueError(f"tool_invocations_fingerprints must be list or set, got {type(v)}")

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """Serialise: convert set[str] → sorted list[str] for JSON safety."""
        data = super().model_dump(**kwargs)
        data["tool_invocations_fingerprints"] = sorted(self.tool_invocations_fingerprints)
        return data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def clone(self) -> AgentState:
        """Return a deep copy of this state for use in LangGraph reducers."""
        return copy.deepcopy(self)

    def add_evidence(self, item: EvidenceItem) -> None:
        """Add *item* to the evidence workspace; silently skip duplicates by evidence_id."""
        existing_ids = {ev.evidence_id for ev in self.evidence_items}
        if item.evidence_id not in existing_ids:
            self.evidence_items.append(item)

    @staticmethod
    def _fingerprint(tool_name: str, params: dict[str, Any]) -> str:
        """Return a SHA-1 hex digest for ``(tool_name, params)``."""
        key = f"{tool_name}:{json.dumps(params, sort_keys=True, default=str)}"
        return hashlib.sha1(key.encode()).hexdigest()  # noqa: S324

    def record_tool_call(
        self,
        *,
        call_id: str,
        tool_name: str,
        params: dict[str, Any],
        status: Literal["ok", "overflow", "error", "empty"] = "ok",
        token_estimate: int = 0,
        source_count: int = 0,
        error: str | None = None,
    ) -> ToolCallRecord:
        """Append a ToolCallRecord and register the invocation fingerprint.

        Returns the new record so callers can chain or inspect it.
        """
        record = ToolCallRecord(
            call_id=call_id,
            tool_name=tool_name,
            params=params,
            status=status,
            token_estimate=token_estimate,
            source_count=source_count,
            error=error,
        )
        self.tool_calls.append(record)
        fp = self._fingerprint(tool_name, params)
        self.tool_invocations_fingerprints.add(fp)
        return record

    def is_duplicate_tool_call(self, tool_name: str, params: dict[str, Any]) -> bool:
        """Return True if ``(tool_name, params)`` has already been invoked."""
        return self._fingerprint(tool_name, params) in self.tool_invocations_fingerprints

    def record_error(
        self,
        *,
        code: str,
        detail: str,
        tool_name: str | None = None,
    ) -> None:
        """Append an AgentError to ``self.errors``."""
        self.errors.append(AgentError(code=code, detail=detail, tool_name=tool_name))

    def record_event(
        self,
        kind: Literal["node_enter", "node_exit", "tool_call", "policy_violation", "budget_check"],
        name: str,
        **payload: Any,
    ) -> None:
        """Delegate to ``debug_trace.record``."""
        self.debug_trace.record(kind, name, **payload)


__all__ = [
    "AgentDebugTrace",
    "AgentError",
    "AgentPlan",
    "AgentState",
    "ChatManifestSnapshot",
    "CitationDraft",
    "ConversationTurn",
    "CoverageRequirement",
    "DocumentManifest",
    "EvidenceItem",
    "GenerationConfig",
    "StructuredFactSnapshot",
    "ToolCallRecord",
    "TraceEvent",
    "make_evidence_id",
]
