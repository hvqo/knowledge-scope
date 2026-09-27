"""Persistent models for the knowledge-base chat workspace."""

from __future__ import annotations

from datetime import datetime
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import Mapped, mapped_column

from knowledge_scope.shared.database import Base

CHAT_CONVERSATION_TITLE_MAX_LENGTH: Final = 200
CHAT_CONVERSATION_DEFAULT_TITLE: Final = "新对话"
CHAT_PROJECT_TITLE_MAX_LENGTH: Final = 200
CHAT_PROJECT_DESCRIPTION_MAX_LENGTH: Final = 2_000
CHAT_MEMORY_CONTENT_MAX_LENGTH: Final = 500
CHAT_MESSAGE_CONTENT_MAX_LENGTH: Final = 100_000
CHAT_MESSAGE_CITATIONS_MAX_ITEMS: Final = 100
CHAT_MESSAGE_ROLES: Final = ("user", "assistant")
CHAT_MESSAGE_STATUSES: Final = ("complete", "error")


class ChatProject(Base):
    """A named group of conversations, comparable to a ChatGPT project."""

    __tablename__ = "chat_projects"
    __table_args__ = (
        CheckConstraint(
            "title = btrim(title) AND btrim(title) <> ''",
            name="ck_chat_projects_title_trimmed_non_empty",
        ),
        Index("ix_chat_projects_updated_at_id", "updated_at", "id"),
    )

    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(String(CHAT_PROJECT_TITLE_MAX_LENGTH), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ChatMemory(Base):
    """One durable fact the assistant should remember across conversations."""

    __tablename__ = "chat_memories"
    __table_args__ = (
        CheckConstraint(
            "content = btrim(content) AND btrim(content) <> ''",
            name="ck_chat_memories_content_trimmed_non_empty",
        ),
        Index("ix_chat_memories_knowledge_base_created", "knowledge_base_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    knowledge_base_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_conversation_id: Mapped[UUID | None] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("chat_conversations.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ChatConversation(Base):
    """One chat thread, optionally bound to a knowledge base and a project."""

    __tablename__ = "chat_conversations"
    __table_args__ = (
        CheckConstraint(
            "title = btrim(title) AND btrim(title) <> ''",
            name="ck_chat_conversations_title_trimmed_non_empty",
        ),
        Index("ix_chat_conversations_updated_at_id", "updated_at", "id"),
        Index("ix_chat_conversations_knowledge_base", "knowledge_base_id"),
        Index("ix_chat_conversations_project", "project_id"),
    )

    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(
        String(CHAT_CONVERSATION_TITLE_MAX_LENGTH),
        nullable=False,
        default=CHAT_CONVERSATION_DEFAULT_TITLE,
        server_default=CHAT_CONVERSATION_DEFAULT_TITLE,
    )
    knowledge_base_id: Mapped[UUID | None] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=True,
    )
    project_id: Mapped[UUID | None] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("chat_projects.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ChatMessage(Base):
    """One ordered question or answer inside a conversation."""

    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="ck_chat_messages_role_known"),
        CheckConstraint("status IN ('complete', 'error')", name="ck_chat_messages_status_known"),
        CheckConstraint("position >= 0", name="ck_chat_messages_position_non_negative"),
        Index("ix_chat_messages_conversation_position", "conversation_id", "position"),
    )

    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("chat_conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="complete")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
