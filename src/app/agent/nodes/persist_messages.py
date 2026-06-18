"""Node: persist_messages — persist user message and assistant answer.

Defines the MessageStore Protocol that the node depends on.
Phase 7.3 ships InMemoryMessageStore (no DB).
Phase 7.5 replaces with a real DB-backed implementation.

Security:
  - MessageStore.save_message asserts chat_id matches to prevent cross-chat writes.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from app.agent.state import AgentState, ConversationTurn

# ---------------------------------------------------------------------------
# MessageStore Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MessageStore(Protocol):
    """Protocol for message persistence.

    Phase 7.3 concrete: InMemoryMessageStore.
    Phase 7.5 concrete: DBMessageStore.
    """

    async def save_message(
        self,
        *,
        chat_id: UUID,
        session_id: UUID,
        role: str,
        content: str,
    ) -> None:
        """Persist one message.

        Raises:
            ValueError: if chat_id does not match the store's expected chat_id.
        """
        ...


# ---------------------------------------------------------------------------
# InMemoryMessageStore (Phase 7.3 implementation)
# ---------------------------------------------------------------------------


class InMemoryMessageStore:
    """In-memory message store for Phase 7.3 tests.

    Stores messages in a plain list; asserts chat_id consistency.
    """

    def __init__(self, expected_chat_id: UUID | None = None) -> None:
        self._expected_chat_id = expected_chat_id
        self.messages: list[dict[str, Any]] = []

    async def save_message(
        self,
        *,
        chat_id: UUID,
        session_id: UUID,
        role: str,
        content: str,
    ) -> None:
        if self._expected_chat_id is not None and chat_id != self._expected_chat_id:
            raise ValueError(
                f"MessageStore: chat_id mismatch: expected {self._expected_chat_id}, got {chat_id}"
            )
        self.messages.append(
            {
                "chat_id": chat_id,
                "session_id": session_id,
                "role": role,
                "content": content,
            }
        )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def persist_messages(
    state: AgentState,
    message_store: MessageStore,
) -> dict[str, Any]:
    """Save user question and assistant answer to the message store."""
    state.record_event("node_enter", "persist_messages")

    # Save user message
    await message_store.save_message(
        chat_id=state.chat_id,
        session_id=state.session_id,
        role="user",
        content=state.question,
    )

    # Save assistant answer (may be None if generation failed)
    answer = state.answer or (
        "There is not enough information in the current chat's documents to answer this question."
    )
    await message_store.save_message(
        chat_id=state.chat_id,
        session_id=state.session_id,
        role="assistant",
        content=answer,
    )

    # Update conversation history
    new_history = list(state.conversation_history) + [
        ConversationTurn(role="user", content=state.question),
        ConversationTurn(role="assistant", content=answer),
    ]

    state.record_event("node_exit", "persist_messages", messages_saved=2)

    return {
        "conversation_history": new_history,
        "debug_trace": state.debug_trace,
    }


__all__ = ["persist_messages", "MessageStore", "InMemoryMessageStore"]
