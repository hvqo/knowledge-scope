"""Persistence helpers shared by the chat endpoints."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_scope.chat.models import ChatConversation, ChatMemory, ChatMessage, ChatProject
from knowledge_scope.knowledge_bases.models import KnowledgeBase

CHAT_TITLE_PREVIEW_LENGTH = 30
CHAT_TITLE_PREVIEW_ELLIPSIS = "…"


def title_from_question(question: str) -> str:
    """Derive a short conversation title from the first user question."""

    collapsed = " ".join(question.split())
    if len(collapsed) <= CHAT_TITLE_PREVIEW_LENGTH:
        return collapsed
    return f"{collapsed[:CHAT_TITLE_PREVIEW_LENGTH]}{CHAT_TITLE_PREVIEW_ELLIPSIS}"


async def ensure_knowledge_base(session: AsyncSession, knowledge_base_id: UUID) -> None:
    """Reject a conversation binding that points at a missing knowledge base."""

    exists = await session.scalar(
        select(func.count()).select_from(KnowledgeBase).where(KnowledgeBase.id == knowledge_base_id)
    )
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="知识库不存在",
        )


async def get_conversation(session: AsyncSession, conversation_id: UUID) -> ChatConversation:
    """Load one conversation or fail with a 404."""

    conversation = await session.get(ChatConversation, conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="对话不存在",
        )
    return conversation


async def get_project(session: AsyncSession, project_id: UUID) -> ChatProject:
    """Load one project or fail with a 404."""

    project = await session.get(ChatProject, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="项目不存在",
        )
    return project


async def count_project_conversations(session: AsyncSession, project_id: UUID) -> int:
    """Count the conversations currently grouped under one project."""

    total = await session.scalar(
        select(func.count())
        .select_from(ChatConversation)
        .where(ChatConversation.project_id == project_id)
    )
    return int(total or 0)


async def list_memories(
    session: AsyncSession,
    knowledge_base_id: UUID,
    *,
    limit: int,
    offset: int,
) -> tuple[tuple[ChatMemory, ...], int]:
    """Return one knowledge base's memories, newest first, with the total."""

    total = await session.scalar(
        select(func.count())
        .select_from(ChatMemory)
        .where(ChatMemory.knowledge_base_id == knowledge_base_id)
    )
    result = await session.scalars(
        select(ChatMemory)
        .where(ChatMemory.knowledge_base_id == knowledge_base_id)
        .order_by(ChatMemory.created_at.desc(), ChatMemory.id)
        .limit(limit)
        .offset(offset)
    )
    return tuple(result), int(total or 0)


async def load_recent_memories(
    session: AsyncSession,
    knowledge_base_id: UUID,
    *,
    limit: int,
) -> tuple[ChatMemory, ...]:
    """Return the newest memories of one knowledge base for prompt injection."""

    result = await session.scalars(
        select(ChatMemory)
        .where(ChatMemory.knowledge_base_id == knowledge_base_id)
        .order_by(ChatMemory.created_at.desc(), ChatMemory.id)
        .limit(limit)
    )
    return tuple(result)


async def count_messages(session: AsyncSession, conversation_id: UUID) -> int:
    """Count persisted messages of one conversation."""

    total = await session.scalar(
        select(func.count())
        .select_from(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
    )
    return int(total or 0)


async def load_messages(
    session: AsyncSession,
    conversation_id: UUID,
) -> tuple[ChatMessage, ...]:
    """Return every message of one conversation in reading order."""

    result = await session.scalars(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.position, ChatMessage.id)
    )
    return tuple(result)


async def load_message_counts(
    session: AsyncSession,
    conversation_ids: Sequence[UUID],
) -> dict[UUID, int]:
    """Return message counts for a bounded batch of conversations."""

    if not conversation_ids:
        return {}
    counts = (
        select(ChatMessage.conversation_id, func.count().label("message_count"))
        .where(ChatMessage.conversation_id.in_(list(conversation_ids)))
        .group_by(ChatMessage.conversation_id)
    )
    return {row.conversation_id: int(row.message_count) for row in await session.execute(counts)}


async def next_message_position(session: AsyncSession, conversation_id: UUID) -> int:
    """Return the position that keeps appended messages in reading order."""

    latest = await session.scalar(
        select(func.max(ChatMessage.position)).where(ChatMessage.conversation_id == conversation_id)
    )
    return int(latest) + 1 if latest is not None else 0


__all__ = [
    "CHAT_TITLE_PREVIEW_LENGTH",
    "count_messages",
    "count_project_conversations",
    "ensure_knowledge_base",
    "get_conversation",
    "get_project",
    "list_memories",
    "load_message_counts",
    "load_messages",
    "load_recent_memories",
    "next_message_position",
    "title_from_question",
]
