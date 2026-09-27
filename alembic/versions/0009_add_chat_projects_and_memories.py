"""Add chat projects and long-term chat memories."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    uuid = postgresql.UUID(as_uuid=True)

    op.create_table(
        "chat_projects",
        sa.Column("id", uuid, nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
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
            name="ck_chat_projects_title_trimmed_non_empty",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_projects_updated_at_id", "chat_projects", ["updated_at", "id"])

    op.add_column(
        "chat_conversations",
        sa.Column("project_id", uuid, nullable=True),
    )
    op.create_foreign_key(
        "fk_chat_conversations_project_id",
        "chat_conversations",
        "chat_projects",
        ["project_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_chat_conversations_project",
        "chat_conversations",
        ["project_id"],
    )

    op.create_table(
        "chat_memories",
        sa.Column("id", uuid, nullable=False),
        sa.Column("knowledge_base_id", uuid, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_conversation_id", uuid, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "content = btrim(content) AND btrim(content) <> ''",
            name="ck_chat_memories_content_trimmed_non_empty",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"],
            ["knowledge_bases.id"],
            name="fk_chat_memories_knowledge_base_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_conversation_id"],
            ["chat_conversations.id"],
            name="fk_chat_memories_source_conversation_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chat_memories_knowledge_base_created",
        "chat_memories",
        ["knowledge_base_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_chat_memories_knowledge_base_created", table_name="chat_memories")
    op.drop_table("chat_memories")
    op.drop_index("ix_chat_conversations_project", table_name="chat_conversations")
    op.drop_constraint(
        "fk_chat_conversations_project_id",
        "chat_conversations",
        type_="foreignkey",
    )
    op.drop_column("chat_conversations", "project_id")
    op.drop_index("ix_chat_projects_updated_at_id", table_name="chat_projects")
    op.drop_table("chat_projects")
