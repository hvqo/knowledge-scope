"""Conversation and message CRUD for the chat workspace."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_scope.chat.models import (
    CHAT_CONVERSATION_DEFAULT_TITLE,
    ChatConversation,
    ChatMessage,
)
from knowledge_scope.chat.schemas import (
    ChatConversationCreate,
    ChatConversationDetailResponse,
    ChatConversationListResponse,
    ChatConversationResponse,
    ChatConversationUpdate,
    ChatMessageCreate,
    ChatMessageResponse,
)
from knowledge_scope.chat.service import (
    count_messages,
    ensure_knowledge_base,
    get_conversation,
    get_project,
    load_message_counts,
    load_messages,
    next_message_position,
    title_from_question,
)
from knowledge_scope.shared.database import get_session

router = APIRouter(prefix="/chat/conversations", tags=["chat"])

SessionDependency = Annotated[AsyncSession, Depends(get_session)]


async def _conversation_response(
    session: AsyncSession,
    conversation: ChatConversation,
    *,
    message_count: int | None = None,
) -> ChatConversationResponse:
    count = (
        message_count
        if message_count is not None
        else await count_messages(session, conversation.id)
    )
    return ChatConversationResponse(
        id=conversation.id,
        title=conversation.title,
        knowledge_base_id=conversation.knowledge_base_id,
        project_id=conversation.project_id,
        message_count=count,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.get("", response_model=ChatConversationListResponse)
async def list_conversations(
    session: SessionDependency,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ChatConversationListResponse:
    """Return conversations ordered by their latest activity."""
    total = await session.scalar(select(func.count()).select_from(ChatConversation))
    conversations = tuple(
        await session.scalars(
            select(ChatConversation)
            .order_by(ChatConversation.updated_at.desc(), ChatConversation.id)
            .limit(limit)
            .offset(offset)
        )
    )
    counts = await load_message_counts(session, [item.id for item in conversations])
    return ChatConversationListResponse(
        items=[
            await _conversation_response(
                session,
                conversation,
                message_count=counts.get(conversation.id, 0),
            )
            for conversation in conversations
        ],
        total=int(total or 0),
        limit=limit,
        offset=offset,
    )


@router.post("", response_model=ChatConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ChatConversationCreate,
    session: SessionDependency,
) -> ChatConversationResponse:
    """Create an empty conversation, optionally bound to a knowledge base."""
    if payload.knowledge_base_id is not None:
        await ensure_knowledge_base(session, payload.knowledge_base_id)
    if payload.project_id is not None:
        await get_project(session, payload.project_id)
    conversation = ChatConversation(
        title=payload.title or CHAT_CONVERSATION_DEFAULT_TITLE,
        knowledge_base_id=payload.knowledge_base_id,
        project_id=payload.project_id,
    )
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return await _conversation_response(session, conversation, message_count=0)


@router.get("/{conversation_id}", response_model=ChatConversationDetailResponse)
async def read_conversation(
    conversation_id: UUID,
    session: SessionDependency,
) -> ChatConversationDetailResponse:
    """Return one conversation with every persisted message."""
    conversation = await get_conversation(session, conversation_id)
    messages = await load_messages(session, conversation_id)
    return ChatConversationDetailResponse(
        conversation=await _conversation_response(
            session,
            conversation,
            message_count=len(messages),
        ),
        messages=[ChatMessageResponse.model_validate(message) for message in messages],
    )


@router.patch("/{conversation_id}", response_model=ChatConversationResponse)
async def update_conversation(
    conversation_id: UUID,
    payload: ChatConversationUpdate,
    session: SessionDependency,
) -> ChatConversationResponse:
    """Rename a conversation or rebind it to another knowledge base."""
    conversation = await get_conversation(session, conversation_id)
    if "title" in payload.model_fields_set and payload.title is not None:
        conversation.title = payload.title
    if "knowledge_base_id" in payload.model_fields_set:
        if payload.knowledge_base_id is not None:
            await ensure_knowledge_base(session, payload.knowledge_base_id)
        conversation.knowledge_base_id = payload.knowledge_base_id
    if "project_id" in payload.model_fields_set:
        if payload.project_id is not None:
            await get_project(session, payload.project_id)
        conversation.project_id = payload.project_id
    await session.commit()
    await session.refresh(conversation)
    return await _conversation_response(session, conversation)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: UUID,
    session: SessionDependency,
) -> Response:
    """Delete a conversation and every message it owns."""
    conversation = await get_conversation(session, conversation_id)
    await session.delete(conversation)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{conversation_id}/messages",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def append_message(
    conversation_id: UUID,
    payload: ChatMessageCreate,
    session: SessionDependency,
) -> ChatMessageResponse:
    """Append one question or answer and keep the conversation title current."""
    conversation = await get_conversation(session, conversation_id)
    message = ChatMessage(
        conversation_id=conversation.id,
        role=payload.role,
        status=payload.status,
        content=payload.content,
        citations=[citation.model_dump(mode="json") for citation in payload.citations],
        position=await next_message_position(session, conversation.id),
    )
    session.add(message)
    if payload.role == "user" and conversation.title == CHAT_CONVERSATION_DEFAULT_TITLE:
        conversation.title = title_from_question(payload.content)
    # The sidebar is ordered by latest activity, so every appended message
    # advances the conversation timestamp even when the title stays the same.
    conversation.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(message)
    return ChatMessageResponse.model_validate(message)


__all__ = ["router"]
