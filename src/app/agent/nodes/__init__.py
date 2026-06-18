"""Agent graph nodes package — Phase 7.3.

Each node is an async function with signature:
    async def <name>(state: AgentState, deps: ToolDeps, ...) -> dict[str, Any]

Nodes return partial-update dicts; the graph assembles full state updates.
"""

from __future__ import annotations

__all__: list[str] = []
