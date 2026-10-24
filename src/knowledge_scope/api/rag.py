# ruff: noqa: RUF001
"""Streaming HTTP endpoint for retrieval-augmented question answering."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from knowledge_scope.agent.service import AgentChatService
from knowledge_scope.agent.tools import DocumentLoader
from knowledge_scope.chat.models import ChatConversation
from knowledge_scope.chat.service import load_recent_memories
from knowledge_scope.chat.trust import record_streamed_answer
from knowledge_scope.documents.models import Document
from knowledge_scope.rag.schemas import (
    RAGCitation,
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


async def _events(
    service: RAGService,
    payload: RAGQueryRequest,
) -> AsyncIterator[RAGStreamEvent]:
    async with aclosing(service.stream(payload)) as event_stream:
        async for event in event_stream:
            yield event


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


async def _agent_events(
    service: AgentChatService,
    payload: RAGQueryRequest,
    request: Request,
) -> AsyncIterator[RAGStreamEvent]:
    settings: Settings = request.app.state.settings
    timeout_seconds = _agent_timeout(settings)
    started = time.perf_counter()
    try:
        async with asyncio.timeout(timeout_seconds):
            async with aclosing(
                service.stream(payload, document_loader=_document_loader(request))
            ) as event_stream:
                async for event in event_stream:
                    yield event
    except TimeoutError:
        logger.warning("agent turn timed out after %ss", timeout_seconds)
        yield RAGStreamEvent(
            event="error",
            data=RAGErrorData(
                category="agent",
                message=(
                    f"智能体执行超过 {int(timeout_seconds)} 秒，已终止本轮问答。"
                    "请重试或切换为标准问答。"
                ),
            ).model_dump(mode="json"),
        )
        yield RAGStreamEvent(
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


async def _trusted_event_stream(
    events: AsyncIterator[RAGStreamEvent],
    *,
    settings: Settings,
    session_factory: object | None,
    payload: RAGQueryRequest,
    answer_mode: Literal["agent", "rag"],
) -> AsyncIterator[str]:
    """Encode events and attach a server-signed trust record to completed turns.

    The record is written before the final ``complete`` event leaves the server,
    so the client can rely on it for report insertion. A missing conversation or
    database failure only disables that record; the answer stream itself is not
    blocked.
    """

    answer_parts: list[str] = []
    citations: list[RAGCitation] = []
    complete_data: RAGCompleteData | None = None
    async with aclosing(events) as event_stream:
        async for event in event_stream:
            if event.event == "answer_delta":
                text = event.data.get("text")
                if isinstance(text, str):
                    answer_parts.append(text)
                yield encode_sse(event)
            elif event.event == "citations":
                for item in event.data.get("items") or []:
                    try:
                        citations.append(RAGCitation.model_validate(item))
                    except ValidationError:
                        logger.warning("citations event contains an invalid citation")
                yield encode_sse(event)
            elif event.event == "complete":
                try:
                    complete_data = RAGCompleteData.model_validate(event.data)
                except ValidationError:
                    logger.warning("complete event failed validation; trust record skipped")
                    yield encode_sse(event)
                    complete_data = None
            else:
                yield encode_sse(event)
    if complete_data is None:
        return
    data = complete_data.model_dump(mode="json")
    trust_id: str | None = None
    if (
        complete_data.status == "completed"
        and payload.conversation_id is not None
        and payload.knowledge_base_id is not None
        and session_factory is not None
    ):
        try:
            async with session_factory() as session:
                conversation = await session.get(ChatConversation, payload.conversation_id)
                if (
                    conversation is not None
                    and conversation.knowledge_base_id == payload.knowledge_base_id
                ):
                    trust_id = await asyncio.to_thread(
                        record_streamed_answer,
                        settings,
                        conversation_id=payload.conversation_id,
                        knowledge_base_id=payload.knowledge_base_id,
                        content="".join(answer_parts),
                        citations=citations,
                        answer_mode=answer_mode,
                        complete=complete_data,
                    )
        except Exception as error:  # recording is best-effort for the stream
            logger.warning("agent trust record skipped (%s)", type(error).__name__)
    data["trust_id"] = trust_id
    data["answer_generated_by"] = (
        "safety_gate"
        if complete_data.safety_gate != "none"
        else ("agent" if answer_mode == "agent" else "rag")
    )
    yield encode_sse(RAGStreamEvent(event="complete", data=data))


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
            _trusted_event_stream(
                _agent_events(agent_service, payload, request),
                settings=settings,
                session_factory=getattr(request.app.state, "db_session_factory", None),
                payload=payload,
                answer_mode="agent",
            ),
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
        _trusted_event_stream(
            _events(service, payload),
            settings=settings,
            session_factory=getattr(request.app.state, "db_session_factory", None),
            payload=payload,
            answer_mode="rag",
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


__all__ = ["encode_sse", "router"]
