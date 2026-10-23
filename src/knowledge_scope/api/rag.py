# ruff: noqa: RUF001
"""Streaming HTTP endpoint for retrieval-augmented question answering."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from knowledge_scope.agent.service import AgentChatService
from knowledge_scope.agent.tools import DocumentLoader
from knowledge_scope.chat.service import load_recent_memories
from knowledge_scope.documents.models import Document
from knowledge_scope.rag.schemas import (
    RAGCompleteData,
    RAGErrorData,
    RAGQueryRequest,
    RAGStreamEvent,
)
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


def _document_loader(request: Request) -> DocumentLoader:
    """Build a request-scoped document resolver for agent tool execution."""

    session_factory = getattr(request.app.state, "db_session_factory", None)

    async def load(knowledge_base_id: UUID, document_id: UUID) -> Document | None:
        if session_factory is None:
            return None
        async with session_factory() as session:
            document = await session.get(Document, document_id)
            if document is None or document.knowledge_base_id != knowledge_base_id:
                return None
            return document

    return load


def _agent_timeout(settings: Settings) -> float:
    """Return the wall-clock guard for one agentic turn."""

    return settings.agent_timeout_seconds


async def _agent_event_stream(
    service: AgentChatService,
    payload: RAGQueryRequest,
    request: Request,
) -> AsyncIterator[str]:
    settings: Settings = request.app.state.settings
    timeout_seconds = _agent_timeout(settings)
    started = time.perf_counter()
    try:
        async with asyncio.timeout(timeout_seconds):
            async with aclosing(
                service.stream(payload, document_loader=_document_loader(request))
            ) as event_stream:
                async for event in event_stream:
                    yield encode_sse(event)
    except TimeoutError:
        logger.warning("agent turn timed out after %ss", timeout_seconds)
        yield encode_sse(
            RAGStreamEvent(
                event="error",
                data=RAGErrorData(
                    category="agent",
                    message=(
                        f"智能体执行超过 {int(timeout_seconds)} 秒，已终止本轮问答。"
                        "请重试或切换为标准问答。"
                    ),
                ).model_dump(mode="json"),
            )
        )
        yield encode_sse(
            RAGStreamEvent(
                event="complete",
                data=RAGCompleteData(
                    status="error",
                    prompt_version="agent-rag-v1",
                    retrieval_mode="unified",
                    route="unified",
                    answer_mode="agent",
                    tool_calls=0,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                ).model_dump(mode="json"),
            )
        )


@router.post("/query", response_class=StreamingResponse)
async def query(payload: RAGQueryRequest, request: Request) -> StreamingResponse:
    """Stream a grounded answer and application-generated citation metadata."""
    settings: Settings = request.app.state.settings
    memories = await _load_memories(
        settings,
        getattr(request.app.state, "db_session_factory", None),
        payload.knowledge_base_id,
    )
    if memories:
        payload = payload.model_copy(update={"memories": memories})
    if payload.answer_mode == "agent":
        agent_service = getattr(request.app.state, "agent_service", None)
        if agent_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent service is not initialized",
            )
        return StreamingResponse(
            _agent_event_stream(agent_service, payload, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not initialized",
        )
    return StreamingResponse(
        _event_stream(service, payload),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


__all__ = ["encode_sse", "router"]
