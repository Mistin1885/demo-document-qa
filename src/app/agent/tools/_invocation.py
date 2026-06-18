"""Shared ToolInvocation wrapper and ToolDeps for all agent tools.

ToolInvocation is the canonical return type for every tool:
  - record: ToolCallRecord (for state.tool_calls)
  - evidence: list[EvidenceItem]  (to merge into state.evidence_items)
  - facts: list[StructuredFactSnapshot] (to merge into state.structured_facts)
  - errors: list[AgentError]
  - chat_manifest: optional ChatManifestSnapshot (only from inspect_chat)

Tools do NOT mutate AgentState directly; the Phase 7.3 graph nodes do the merge.

Design rules (CLAUDE.md §8, §12):
- No dict[str, Any] on typed fields.
- No FastAPI imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from app.agent.state import (
    AgentError,
    ChatManifestSnapshot,
    DocumentManifest,
    EvidenceItem,
    StructuredFactSnapshot,
    ToolCallRecord,
)

if TYPE_CHECKING:
    from app.providers.base import ChatProvider
    from app.retrieval.service import RetrievalService


# ---------------------------------------------------------------------------
# SessionFactory protocol — avoids complex generic typing
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionFactory(Protocol):
    """Protocol for an async context-manager factory that yields a DB session.

    Signature: ``() -> AsyncContextManager[Any]``.
    Implemented by production code as an ``asynccontextmanager`` decorated
    function; tests pass a simple ``@asynccontextmanager`` async function.
    """

    def __call__(self) -> Any:  # returns an async context manager
        ...


# ---------------------------------------------------------------------------
# ToolInvocation — typed return wrapper
# ---------------------------------------------------------------------------


class ToolInvocation(BaseModel):
    """Typed return value from every agent tool.

    The Phase 7.3 graph node is responsible for merging these fields into
    AgentState (reducer pattern).  Tools never mutate state directly.
    """

    model_config = ConfigDict(extra="forbid")

    record: ToolCallRecord
    evidence: list[EvidenceItem] = Field(default_factory=list)
    facts: list[StructuredFactSnapshot] = Field(default_factory=list)
    errors: list[AgentError] = Field(default_factory=list)

    # Only populated by inspect_chat
    chat_manifest: ChatManifestSnapshot | None = None

    # Only populated by inspect_document
    document_manifest: DocumentManifest | None = None


# ---------------------------------------------------------------------------
# ToolDeps — dependency injection carrier (passed as keyword-only dep=)
# ---------------------------------------------------------------------------


@dataclass
class ToolDeps:
    """Runtime dependencies injected into every tool.

    Constructed once per agent invocation by the service layer and passed as
    a keyword argument ``deps=`` to every tool function.  Tests substitute
    lightweight fakes (no real network / DB calls).

    Attributes
    ----------
    retrieval_service:
        The single RetrievalService instance (wraps Vespa).
    chat_provider:
        ChatProvider for the current session (used by aggregate_sources in
        Phase 9; placeholder in Phase 7.2 — not called).
    session_factory:
        Callable that returns an async context manager yielding AsyncSession.
        Signature: ``() -> AsyncContextManager[AsyncSession]``.
        In tests, pass a factory that yields an in-memory session.
    """

    retrieval_service: RetrievalService
    chat_provider: ChatProvider
    session_factory: SessionFactory  # async context-manager factory


__all__ = ["ToolDeps", "ToolInvocation"]
