"""Read-only vision analysis endpoints scoped to one evidence image."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_scope.documents.assets import AssetAccessError
from knowledge_scope.shared.database import get_session
from knowledge_scope.vision.errors import (
    VisionImageTooLargeError,
    VisionRequestError,
)
from knowledge_scope.vision.schemas import VisionAnalysisRequest, VisionAnalysisResult
from knowledge_scope.vision.service import VisionAnalysisService

from .documents import _asset_access_http_error, _get_document
from .knowledge_bases import _get_knowledge_base

router = APIRouter(prefix="/vision", tags=["vision"])


@router.post("/analyze", response_model=VisionAnalysisResult)
async def analyze_evidence_image(
    payload: VisionAnalysisRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> VisionAnalysisResult:
    """Analyze one image already registered as evidence inside a knowledge base."""
    await _get_knowledge_base(session, payload.knowledge_base_id)
    document = await _get_document(session, payload.knowledge_base_id, payload.document_id)
    service = VisionAnalysisService(request.app.state.settings, request.app.state.vlm_provider)
    try:
        return await service.analyze(
            document,
            evidence_id=payload.evidence_id,
            asset_index=payload.asset_index,
            task_type=payload.task_type,
            question=payload.question,
        )
    except AssetAccessError as error:
        raise _asset_access_http_error(error) from None
    except VisionImageTooLargeError as error:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=error.message,
        ) from None
    except VisionRequestError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=error.message,
        ) from None


__all__ = ["router"]
