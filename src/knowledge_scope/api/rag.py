"""Streaming HTTP endpoint for retrieval-augmented question answering."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from knowledge_scope.chat.service import load_recent_memories
from knowledge_scope.rag.schemas import RAGQueryRequest, RAGStreamEvent
from knowledge_scope.rag.service import RAGService
from knowledge_scope.shared.config import Settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


def encode_sse(event: RAGStreamEvent) -> str:
    """Encode one validated RAG event using the SSE wire format."""
    data = json.dumps(event.data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event.event}\ndata: {data}\n\n"


async def _load_memories(
    settings: Settings,
    session_factory: object | None,
    knowledge_base_id: UUID | None,
) -> tuple[str, ...]:
    """Return the knowledge base's recent memories for prompt injection.

    Memories are an enhancement, never a requirement: a missing or unavailable
    database must not stop a question from being answered.
    """

    if knowledge_base_id is None or settings.rag_memory_inject_items == 0:
        return ()
    if session_factory is None:
        return ()
    try:
        async with session_factory() as session:
            memories = await load_recent_memories(
                session,
                knowledge_base_id,
                limit=settings.rag_memory_inject_items,
            )
    except Exception as error:  # degradation beats failure
        logger.debug("memory injection skipped (%s)", type(error).__name__)
        return ()
    kept: list[str] = []
    budget = settings.rag_memory_inject_max_chars
    for memory in memories:
        content = memory.content.strip()
        if not content:
            continue
        kept.append(content)
        budget -= min(len(content), budget)
        if budget <= 0:
            break
    return tuple(kept)


async def _event_stream(
    service: RAGService,
    payload: RAGQueryRequest,
) -> AsyncIterator[str]:
    async with aclosing(service.stream(payload)) as event_stream:
        async for event in event_stream:
            yield encode_sse(event)


@router.post("/query", response_class=StreamingResponse)
async def query(payload: RAGQueryRequest, request: Request) -> StreamingResponse:
    """Stream a grounded answer and application-generated citation metadata."""
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not initialized",
        )
    settings: Settings = request.app.state.settings
    memories = await _load_memories(
        settings,
        getattr(request.app.state, "db_session_factory", None),
        payload.knowledge_base_id,
    )
    if memories:
        payload = payload.model_copy(update={"memories": memories})
    return StreamingResponse(
        _event_stream(service, payload),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


__all__ = ["encode_sse", "router"]
