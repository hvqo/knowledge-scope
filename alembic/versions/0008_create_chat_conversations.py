"""Create persistent chat conversation tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    uuid = postgresql.UUID(as_uuid=True)
    jsonb = postgresql.JSONB()

    op.create_table(
        "chat_conversations",
        sa.Column("id", uuid, nullable=False),
        sa.Column("title", sa.String(length=200), server_default="新对话", nullable=False),
        sa.Column("knowledge_base_id", uuid, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "title = btrim(title) AND btrim(title) <> ''",
            name="ck_chat_conversations_title_trimmed_non_empty",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"],
            ["knowledge_bases.id"],
            name="fk_chat_conversations_knowledge_base_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chat_conversations_updated_at_id",
        "chat_conversations",
        ["updated_at", "id"],
    )
    op.create_index(
        "ix_chat_conversations_knowledge_base",
        "chat_conversations",
        ["knowledge_base_id"],
    )

    op.create_table(
        "chat_messages",
        sa.Column("id", uuid, nullable=False),
        sa.Column("conversation_id", uuid, nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="complete", nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", jsonb, server_default="[]", nullable=False),
        sa.Column("position", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="ck_chat_messages_role_known"),
        sa.CheckConstraint(
            "status IN ('complete', 'error')",
            name="ck_chat_messages_status_known",
        ),
        sa.CheckConstraint("position >= 0", name="ck_chat_messages_position_non_negative"),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["chat_conversations.id"],
            name="fk_chat_messages_conversation_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chat_messages_conversation_position",
        "chat_messages",
        ["conversation_id", "position"],
    )


def downgrade() -> None:
    op.drop_index("ix_chat_messages_conversation_position", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index("ix_chat_conversations_knowledge_base", table_name="chat_conversations")
    op.drop_index("ix_chat_conversations_updated_at_id", table_name="chat_conversations")
    op.drop_table("chat_conversations")
