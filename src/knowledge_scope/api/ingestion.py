"""Read/retry endpoints for the PDF auto-ingestion pipeline."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_scope.ingestion.models import status_entry
from knowledge_scope.ingestion.worker import IngestionError, IngestionWorker
from knowledge_scope.shared.database import get_session

from .documents import _get_document, _get_knowledge_base

router = APIRouter(tags=["ingestion"])


def _worker(request: Request) -> IngestionWorker:
    worker = getattr(request.app.state, "ingestion_worker", None)
    if worker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="自动入库任务未启用",
        )
    return worker


def _task_payload(task: object | None) -> dict[str, object] | None:
    if task is None:
        return None
    entry = status_entry(task)  # type: ignore[arg-type]
    return {"task": task.model_dump(mode="json"), "summary": entry.model_dump(mode="json")}  # type: ignore[attr-defined]


@router.get("/knowledge-bases/{knowledge_base_id}/documents/{document_id}/ingestion")
async def get_ingestion_status(
    knowledge_base_id: UUID,
    document_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    await _get_knowledge_base(session, knowledge_base_id)
    await _get_document(session, knowledge_base_id, document_id)
    worker = _worker(request)
    task = worker.status(document_id)
    if task is not None and task.knowledge_base_id != knowledge_base_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="入库任务不存在")
    return {"task": None, "summary": None} if task is None else _task_payload(task)  # type: ignore[return-value]


@router.post(
    "/knowledge-bases/{knowledge_base_id}/documents/{document_id}/ingestion/retry",
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_ingestion(
    knowledge_base_id: UUID,
    document_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    await _get_knowledge_base(session, knowledge_base_id)
    await _get_document(session, knowledge_base_id, document_id)
    worker = _worker(request)
    try:
        task = worker.retry(document_id)
    except IngestionError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from None
    if task.knowledge_base_id != knowledge_base_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="入库任务不存在")
    return _task_payload(task)  # type: ignore[return-value]


@router.get("/knowledge-bases/{knowledge_base_id}/ingestion")
async def list_ingestion_tasks(
    knowledge_base_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    await _get_knowledge_base(session, knowledge_base_id)
    worker = _worker(request)
    tasks = worker.list_tasks(knowledge_base_id)
    return {"tasks": [status_entry(task).model_dump(mode="json") for task in tasks]}


__all__ = ["router"]
