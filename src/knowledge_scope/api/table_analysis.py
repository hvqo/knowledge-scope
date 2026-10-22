"""Read-only HTTP API for deterministic PDF table analysis.

The endpoint accepts explicit table Evidence references and structured
operation parameters only.  It never accepts code, SQL or free-form
expressions, and every document/evidence lookup is scoped to the requested
knowledge base before any table content is read.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_scope.documents.assets import AssetAccessError
from knowledge_scope.documents.models import Document
from knowledge_scope.shared.database import get_session
from knowledge_scope.table_analysis.models import (
    TableAccessError,
    TableAnalysisRequest,
    TableAnalysisResult,
)
from knowledge_scope.table_analysis.service import analyze_table_evidence

from .documents import _get_document
from .knowledge_bases import _get_knowledge_base

router = APIRouter(prefix="/table-analysis", tags=["table-analysis"])


@router.post(
    "/analyze",
    response_model=TableAnalysisResult,
    summary="对 PDF 表格证据执行确定性计算",
)
async def analyze_table(
    request: Request,
    payload: TableAnalysisRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> TableAnalysisResult:
    """Run one deterministic operation over one or two explicitly selected tables.

    Missing/stale/wrong-scope Evidence answers 404.  Table-level problems are
    reported in the returned result with ``status="not_computable"`` because the
    request itself was valid.
    """
    await _get_knowledge_base(session, payload.knowledge_base_id)
    document = await _get_document(
        session,
        payload.knowledge_base_id,
        payload.table.document_id,
    )
    compare_document: Document | None = None
    if payload.compare_table is not None:
        compare_document = await _get_document(
            session,
            payload.knowledge_base_id,
            payload.compare_table.document_id,
        )

    settings = request.app.state.settings
    try:
        return await asyncio.to_thread(
            analyze_table_evidence,
            settings,
            payload,
            document=document,
            compare_document=compare_document,
        )
    except (AssetAccessError, TableAccessError) as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
