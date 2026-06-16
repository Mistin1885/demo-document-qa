"""Integration isolation tests — Session history (Message ORM) isolation.

Phase 7 has not yet implemented a Message API.  These tests validate the
ORM / DB layer contract so that Phase 7 can rely on it.

CLAUDE.md §2 isolation layers covered
--------------------------------------
- **Relational DB query**: SELECT scoped by ``session_id`` must never
  return rows that belong to a different session within the same chat.

Scenarios (n)
--------------
- Two sessions in the same chat each have their own Message rows.
- ORM SELECT filtered to session_a.id returns only session_a's messages.
- ORM SELECT filtered to session_b.id returns only session_b's messages.
- Session A belongs to Chat X; Session B belongs to Chat Y (different chats).
  SELECT filtered to session_a.id returns only session_a's messages.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import Message
from tests._helpers import make_chat, make_session

# ---------------------------------------------------------------------------
# (n) Two sessions in the SAME chat — messages must not cross-contaminate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_message_orm_same_chat_different_sessions_are_isolated(
    db_session: AsyncSession,
) -> None:
    """(n) Sessions A and B are in the same Chat.

    SELECT WHERE session_id = A returns only A's messages.
    SELECT WHERE session_id = B returns only B's messages.

    Layer: Relational DB query — regression gate for Phase 7.
    """
    chat = await make_chat(db_session, "same-chat-msg-iso")
    sess_a = await make_session(db_session, chat, name="history-A")
    sess_b = await make_session(db_session, chat, name="history-B")

    msg_a_1 = Message(session_id=sess_a.id, role="user", content="user query A")
    msg_a_2 = Message(session_id=sess_a.id, role="assistant", content="assistant reply A")
    msg_b_1 = Message(session_id=sess_b.id, role="user", content="user query B")
    db_session.add_all([msg_a_1, msg_a_2, msg_b_1])
    await db_session.flush()

    # Query session A's history
    result_a = await db_session.scalars(
        select(Message).where(Message.session_id == sess_a.id)
    )
    rows_a = list(result_a.all())
    assert len(rows_a) == 2, f"Expected 2 messages for session A, got {len(rows_a)}"
    session_ids_in_a = {r.session_id for r in rows_a}
    assert session_ids_in_a == {sess_a.id}, (
        "Session A's history must not contain messages from session B"
    )

    # Query session B's history
    result_b = await db_session.scalars(
        select(Message).where(Message.session_id == sess_b.id)
    )
    rows_b = list(result_b.all())
    assert len(rows_b) == 1, f"Expected 1 message for session B, got {len(rows_b)}"
    assert rows_b[0].session_id == sess_b.id
    assert rows_b[0].content == "user query B"


# ---------------------------------------------------------------------------
# (n) Two sessions in DIFFERENT chats — messages must not cross-contaminate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_message_orm_different_chats_sessions_are_isolated(
    db_session: AsyncSession,
) -> None:
    """(n) Sessions A and B are in different Chats.

    SELECT WHERE session_id = A must not return any message from B even
    though they share the same PostgreSQL table.

    Layer: Relational DB query.
    """
    chat_x = await make_chat(db_session, "chat-x-msg-iso")
    chat_y = await make_chat(db_session, "chat-y-msg-iso")
    sess_a = await make_session(db_session, chat_x, name="sess-in-X")
    sess_b = await make_session(db_session, chat_y, name="sess-in-Y")

    msg_a = Message(session_id=sess_a.id, role="user", content="message in X")
    msg_b = Message(session_id=sess_b.id, role="user", content="message in Y")
    db_session.add_all([msg_a, msg_b])
    await db_session.flush()

    result_a = await db_session.scalars(
        select(Message).where(Message.session_id == sess_a.id)
    )
    rows_a = list(result_a.all())

    assert len(rows_a) == 1
    assert rows_a[0].session_id == sess_a.id
    assert sess_b.id not in {r.session_id for r in rows_a}, (
        "Session B's messages (different chat) must not appear in session A's query"
    )


# ---------------------------------------------------------------------------
# (n) Empty session — no messages returned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_message_orm_empty_session_returns_no_rows(
    db_session: AsyncSession,
) -> None:
    """(n) A session with no messages returns an empty result set.

    Ensures the filter doesn't accidentally return rows from other sessions
    when the target session has no rows.

    Layer: Relational DB query.
    """
    chat = await make_chat(db_session, "chat-empty-sess")
    empty_sess = await make_session(db_session, chat, name="empty")
    other_sess = await make_session(db_session, chat, name="nonempty")

    msg_other = Message(session_id=other_sess.id, role="user", content="other msg")
    db_session.add(msg_other)
    await db_session.flush()

    result = await db_session.scalars(
        select(Message).where(Message.session_id == empty_sess.id)
    )
    rows = list(result.all())

    assert rows == [], (
        f"Empty session must return 0 messages, got {len(rows)}: "
        f"{[(r.session_id, r.content) for r in rows]}"
    )
