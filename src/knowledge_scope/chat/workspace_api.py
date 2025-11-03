"""Projects and long-term memories for the chat workspace."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_scope.chat.memory import MemoryExtractionService
from knowledge_scope.chat.models import CHAT_MEMORY_CONTENT_MAX_LENGTH, ChatMemory, ChatProject
from knowledge_scope.chat.schemas import (
    ChatMemoryCreate,
    ChatMemoryListResponse,
    ChatMemoryResponse,
    ChatProjectCreate,
    ChatProjectListResponse,
    ChatProjectResponse,
    ChatProjectUpdate,
)
from knowledge_scope.chat.service import (
    count_project_conversations,
    ensure_knowledge_base,
    get_conversation,
    get_project,
    list_memories,
    load_messages,
    load_recent_memories,
)
from knowledge_scope.shared.config import Settings
from knowledge_scope.shared.database import get_session

router = APIRouter(tags=["chat"])

SessionDependency = Annotated[AsyncSession, Depends(get_session)]


async def _project_response(
    session: AsyncSession,
    project: ChatProject,
    *,
    conversation_count: int | None = None,
) -> ChatProjectResponse:
    count = (
        conversation_count
        if conversation_count is not None
        else await count_project_conversations(session, project.id)
    )
    return ChatProjectResponse(
        id=project.id,
        title=project.title,
        description=project.description,
        conversation_count=count,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


@router.get("/chat/projects", response_model=ChatProjectListResponse)
async def list_projects(
    session: SessionDependency,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ChatProjectListResponse:
    """Return projects ordered by their latest activity."""
    total = await session.scalar(select(func.count()).select_from(ChatProject))
    projects = tuple(
        await session.scalars(
            select(ChatProject)
            .order_by(ChatProject.updated_at.desc(), ChatProject.id)
            .limit(limit)
            .offset(offset)
        )
    )
    return ChatProjectListResponse(
        items=[await _project_response(session, project) for project in projects],
        total=int(total or 0),
        limit=limit,
        offset=offset,
    )


@router.post(
    "/chat/projects",
    response_model=ChatProjectResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_project(
    payload: ChatProjectCreate,
    session: SessionDependency,
) -> ChatProjectResponse:
    """Create an empty project that conversations can be grouped under."""
    project = ChatProject(title=payload.title, description=payload.description)
    session.add(project)
    await session.commit()
    await session.refresh(project)
    return await _project_response(session, project, conversation_count=0)


@router.patch("/chat/projects/{project_id}", response_model=ChatProjectResponse)
async def update_project(
    project_id: UUID,
    payload: ChatProjectUpdate,
    session: SessionDependency,
) -> ChatProjectResponse:
    """Rename a project or change its description."""
    project = await get_project(session, project_id)
    if "title" in payload.model_fields_set and payload.title is not None:
        project.title = payload.title
    if "description" in payload.model_fields_set:
        project.description = payload.description
    await session.commit()
    await session.refresh(project)
    return await _project_response(session, project)


@router.delete("/chat/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: UUID, session: SessionDependency) -> Response:
    """Delete a project; its conversations stay, just ungrouped."""
    project = await get_project(session, project_id)
    await session.delete(project)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/chat/memories", response_model=ChatMemoryListResponse)
async def list_memories_endpoint(
    session: SessionDependency,
    knowledge_base_id: UUID = Query(...),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ChatMemoryListResponse:
    """Return one knowledge base's long-term memories, newest first."""
    items, total = await list_memories(
        session,
        knowledge_base_id,
        limit=limit,
        offset=offset,
    )
    return ChatMemoryListResponse(
        items=[ChatMemoryResponse.model_validate(memory) for memory in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "/chat/memories",
    response_model=ChatMemoryResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_memory(
    payload: ChatMemoryCreate,
    session: SessionDependency,
) -> ChatMemoryResponse:
    """Manually add one memory the assistant should remember."""
    await ensure_knowledge_base(session, payload.knowledge_base_id)
    memory = ChatMemory(
        knowledge_base_id=payload.knowledge_base_id,
        content=payload.content,
    )
    session.add(memory)
    await session.commit()
    await session.refresh(memory)
    return ChatMemoryResponse.model_validate(memory)


@router.delete("/chat/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(memory_id: UUID, session: SessionDependency) -> Response:
    """Forget one memory."""
    memory = await session.get(ChatMemory, memory_id)
    if memory is None:
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    await session.delete(memory)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/chat/memories/extract", response_model=list[ChatMemoryResponse])
async def extract_memories(
    conversation_id: UUID,
    request: Request,
    session: SessionDependency,
) -> list[ChatMemoryResponse]:
    """Distill durable memories from one finished conversation."""
    settings: Settings = request.app.state.settings
    gateway = getattr(request.app.state, "llm_gateway", None)
    extraction = MemoryExtractionService(gateway, settings)  # type: ignore[arg-type]
    conversation = await get_conversation(session, conversation_id)
    if conversation.knowledge_base_id is None:
        return []
    messages = await load_messages(session, conversation_id)
    existing = await load_recent_memories(
        session,
        conversation.knowledge_base_id,
        limit=settings.chat_memory_max_items,
    )
    extracted = await extraction.extract(
        tuple((message.role, message.content) for message in messages),
        tuple(memory.content for memory in existing),
    )
    created: list[ChatMemory] = []
    for content in extracted:
        memory = ChatMemory(
            knowledge_base_id=conversation.knowledge_base_id,
            content=content[:CHAT_MEMORY_CONTENT_MAX_LENGTH],
            source_conversation_id=conversation.id,
        )
        session.add(memory)
        created.append(memory)
    if created:
        await session.commit()
        for memory in created:
            await session.refresh(memory)
    return [ChatMemoryResponse.model_validate(memory) for memory in created]


__all__ = ["router"]
