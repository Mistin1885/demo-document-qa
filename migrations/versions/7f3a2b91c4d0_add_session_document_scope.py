"""add session document scope

Revision ID: 7f3a2b91c4d0
Revises: b5b02bc9d209
Create Date: 2026-06-20 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "7f3a2b91c4d0"
down_revision: Union[str, None] = "b5b02bc9d209"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column("selected_document_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column(
            "document_scope_locked",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("sessions", "document_scope_locked")
    op.drop_column("sessions", "selected_document_ids")
