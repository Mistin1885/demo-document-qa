"""LangGraph Agent package for the Paper Notebook Agent (Phase 7).

Exports the canonical AgentState and ContextBudgetManager.
Tool implementations, graph nodes, and policies live in sub-modules
built during phases 7.2–7.5.
"""

from app.agent.budget import ContextAllocation, ContextBudgetManager
from app.agent.state import (
    AgentDebugTrace,
    AgentError,
    AgentPlan,
    AgentState,
    ChatManifestSnapshot,
    CitationDraft,
    ConversationTurn,
    CoverageRequirement,
    DocumentManifest,
    EvidenceItem,
    StructuredFactSnapshot,
    ToolCallRecord,
    TraceEvent,
)

__all__ = [
    # state
    "AgentState",
    "AgentDebugTrace",
    "AgentError",
    "AgentPlan",
    "ChatManifestSnapshot",
    "CitationDraft",
    "ConversationTurn",
    "CoverageRequirement",
    "DocumentManifest",
    "EvidenceItem",
    "StructuredFactSnapshot",
    "ToolCallRecord",
    "TraceEvent",
    # budget
    "ContextAllocation",
    "ContextBudgetManager",
]
