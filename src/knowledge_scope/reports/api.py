"""REST API for the persistent report workspace."""

# Chinese user-facing error text intentionally uses Chinese punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_scope.chatbi.execution import redact_sql_literals
from knowledge_scope.chatbi.models import ChatBIDataSourceRecord
from knowledge_scope.documents.models import Document
from knowledge_scope.knowledge_bases.models import KnowledgeBase
from knowledge_scope.shared.database import get_session

from .models import (
    REPORT_SECTION_CONTENT_MAX_LENGTH,
    Report,
    ReportSection,
    ReportSourceReference,
)
from .schemas import (
    ReportChatBISourceCreate,
    ReportCreate,
    ReportDetailResponse,
    ReportListResponse,
    ReportRagSourceCreate,
    ReportSectionCreate,
    ReportSectionResponse,
    ReportSectionUpdate,
    ReportSourceCreate,
    ReportSourceInsertCreate,
    ReportSourceReferenceResponse,
    ReportSummary,
    ReportUpdate,
)

router = APIRouter(prefix="/reports", tags=["reports"])


async def _get_report(session: AsyncSession, report_id: UUID) -> Report:
    report = await session.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="报告不存在")
    return report


async def _get_section(
    session: AsyncSession,
    report_id: UUID,
    section_id: UUID,
) -> ReportSection:
    section = await session.scalar(
        select(ReportSection).where(
            ReportSection.id == section_id,
            ReportSection.report_id == report_id,
        )
    )
    if section is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="报告章节不存在")
    return section


async def _validate_report_scope(
    session: AsyncSession,
    knowledge_base_id: UUID | None,
    datasource_id: UUID | None,
) -> None:
    if knowledge_base_id is not None:
        knowledge_base_exists = await session.scalar(
            select(KnowledgeBase.id).where(KnowledgeBase.id == knowledge_base_id)
        )
        if knowledge_base_exists is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识库不存在")
    if datasource_id is not None:
        datasource_exists = await session.scalar(
            select(ChatBIDataSourceRecord.id).where(ChatBIDataSourceRecord.id == datasource_id)
        )
        if datasource_exists is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="数据源不存在")


async def _validate_reference_scope(
    session: AsyncSession,
    report_id: UUID,
    knowledge_base_id: UUID | None,
    datasource_id: UUID | None,
) -> None:
    references = await session.scalars(
        select(ReportSourceReference).where(ReportSourceReference.report_id == report_id)
    )
    for reference in references:
        if (
            reference.source_kind == "rag_evidence"
            and reference.knowledge_base_id != knowledge_base_id
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="报告已有知识库证据，不能切换到其他知识库",
            )
        if reference.source_kind == "chatbi_result" and reference.datasource_id != datasource_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="报告已有数据分析来源，不能切换到其他数据源",
            )


def _summary(report: Report) -> ReportSummary:
    return ReportSummary(
        id=report.id,
        title=report.title,
        knowledge_base_id=report.knowledge_base_id,
        datasource_id=report.datasource_id,
        created_at=report.created_at.isoformat(),
        updated_at=report.updated_at.isoformat(),
    )


def _source_response(reference: ReportSourceReference) -> ReportSourceReferenceResponse:
    return ReportSourceReferenceResponse(
        id=reference.id,
        section_id=reference.section_id,
        kind=reference.source_kind,  # type: ignore[arg-type]
        title=reference.title,
        knowledge_base_id=reference.knowledge_base_id,
        datasource_id=reference.datasource_id,
        document_id=reference.document_id,
        document_title=reference.document_title,
        evidence_id=reference.evidence_id,
        chunk_id=reference.chunk_id,
        page_start=reference.page_start,
        page_end=reference.page_end,
        evidence_type=reference.evidence_type,
        snippet=reference.snippet,
        section_path=list(reference.section_path or []),
        source_block_ids=list(reference.source_block_ids or []),
        question=reference.question,
        answer=reference.answer,
        columns=list(reference.columns or []),
        rows=list(reference.rows or []),
        row_count=reference.row_count,
        truncated=reference.truncated,
        truncation_reason=reference.truncation_reason,
        redacted_sql=reference.redacted_sql,
        chart_spec=reference.chart_spec,
        created_at=reference.created_at.isoformat(),
    )


async def _detail(session: AsyncSession, report: Report) -> ReportDetailResponse:
    sections = list(
        await session.scalars(
            select(ReportSection)
            .where(ReportSection.report_id == report.id)
            .order_by(ReportSection.position, ReportSection.created_at, ReportSection.id)
        )
    )
    references = list(
        await session.scalars(
            select(ReportSourceReference)
            .where(ReportSourceReference.report_id == report.id)
            .order_by(ReportSourceReference.created_at, ReportSourceReference.id)
        )
    )
    references_by_section: dict[UUID, list[ReportSourceReferenceResponse]] = {}
    for reference in references:
        references_by_section.setdefault(reference.section_id, []).append(
            _source_response(reference)
        )
    return ReportDetailResponse(
        id=report.id,
        title=report.title,
        knowledge_base_id=report.knowledge_base_id,
        datasource_id=report.datasource_id,
        sections=[
            ReportSectionResponse(
                id=section.id,
                report_id=section.report_id,
                title=section.title,
                content=section.content,
                position=section.position,
                sources=references_by_section.get(section.id, []),
                created_at=section.created_at.isoformat(),
                updated_at=section.updated_at.isoformat(),
            )
            for section in sections
        ],
        created_at=report.created_at.isoformat(),
        updated_at=report.updated_at.isoformat(),
    )


async def _ordered_sections(session: AsyncSession, report_id: UUID) -> list[ReportSection]:
    return list(
        await session.scalars(
            select(ReportSection)
            .where(ReportSection.report_id == report_id)
            .order_by(ReportSection.position, ReportSection.created_at, ReportSection.id)
        )
    )


def _renumber(sections: list[ReportSection]) -> None:
    for position, section in enumerate(sections):
        section.position = position


@router.get("", response_model=ReportListResponse)
async def list_reports(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> ReportListResponse:
    total = int(await session.scalar(select(func.count()).select_from(Report)) or 0)
    reports = list(
        await session.scalars(
            select(Report)
            .order_by(Report.updated_at.desc(), Report.id.desc())
            .offset(offset)
            .limit(limit)
        )
    )
    return ReportListResponse(
        items=[_summary(report) for report in reports],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("", response_model=ReportDetailResponse, status_code=status.HTTP_201_CREATED)
async def create_report(
    payload: ReportCreate,
    session: AsyncSession = Depends(get_session),
) -> ReportDetailResponse:
    await _validate_report_scope(session, payload.knowledge_base_id, payload.datasource_id)
    report = Report(
        title=payload.title,
        knowledge_base_id=payload.knowledge_base_id,
        datasource_id=payload.datasource_id,
    )
    session.add(report)
    await session.flush()
    session.add(ReportSection(report_id=report.id, title="概览", content="", position=0))
    await session.commit()
    await session.refresh(report)
    return await _detail(session, report)


@router.get("/{report_id}", response_model=ReportDetailResponse)
async def get_report(
    report_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> ReportDetailResponse:
    return await _detail(session, await _get_report(session, report_id))


@router.patch("/{report_id}", response_model=ReportDetailResponse)
async def update_report(
    report_id: UUID,
    payload: ReportUpdate,
    session: AsyncSession = Depends(get_session),
) -> ReportDetailResponse:
    report = await _get_report(session, report_id)
    changes = payload.model_dump(exclude_unset=True)
    knowledge_base_id = changes.get("knowledge_base_id", report.knowledge_base_id)
    datasource_id = changes.get("datasource_id", report.datasource_id)
    await _validate_report_scope(session, knowledge_base_id, datasource_id)
    await _validate_reference_scope(session, report.id, knowledge_base_id, datasource_id)
    for field_name, value in changes.items():
        setattr(report, field_name, value)
    report.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(report)
    return await _detail(session, report)


@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_report(
    report_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    report = await _get_report(session, report_id)
    await session.delete(report)
    await session.commit()


@router.post(
    "/{report_id}/sections",
    response_model=ReportSectionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_section(
    report_id: UUID,
    payload: ReportSectionCreate,
    session: AsyncSession = Depends(get_session),
) -> ReportSectionResponse:
    report = await _get_report(session, report_id)
    sections = await _ordered_sections(session, report_id)
    position = len(sections) if payload.position is None else min(payload.position, len(sections))
    section = ReportSection(
        report_id=report_id,
        title=payload.title,
        content=payload.content,
        position=position,
    )
    sections.insert(position, section)
    _renumber(sections)
    session.add(section)
    report.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(section)
    return ReportSectionResponse(
        id=section.id,
        report_id=section.report_id,
        title=section.title,
        content=section.content,
        position=section.position,
        sources=[],
        created_at=section.created_at.isoformat(),
        updated_at=section.updated_at.isoformat(),
    )


@router.patch("/{report_id}/sections/{section_id}", response_model=ReportSectionResponse)
async def update_section(
    report_id: UUID,
    section_id: UUID,
    payload: ReportSectionUpdate,
    session: AsyncSession = Depends(get_session),
) -> ReportSectionResponse:
    section = await _get_section(session, report_id, section_id)
    report = await _get_report(session, report_id)
    changes = payload.model_dump(exclude_unset=True)
    if "position" in changes:
        sections = await _ordered_sections(session, report_id)
        sections.remove(section)
        position = min(changes["position"], len(sections))
        sections.insert(position, section)
        _renumber(sections)
    for field_name, value in changes.items():
        if field_name != "position":
            setattr(section, field_name, value)
    section.updated_at = datetime.now(UTC)
    report.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(section)
    references = list(
        await session.scalars(
            select(ReportSourceReference)
            .where(ReportSourceReference.section_id == section.id)
            .order_by(ReportSourceReference.created_at, ReportSourceReference.id)
        )
    )
    return ReportSectionResponse(
        id=section.id,
        report_id=section.report_id,
        title=section.title,
        content=section.content,
        position=section.position,
        sources=[_source_response(reference) for reference in references],
        created_at=section.created_at.isoformat(),
        updated_at=section.updated_at.isoformat(),
    )


async def _build_source_reference(
    session: AsyncSession,
    report_id: UUID,
    section_id: UUID,
    payload: ReportSourceCreate,
) -> tuple[Report, ReportSection, ReportSourceReference]:
    report = await _get_report(session, report_id)
    section = await _get_section(session, report_id, section_id)
    fields: dict[str, object] = {
        "report_id": report_id,
        "section_id": section_id,
        "source_kind": payload.kind,
        "title": payload.title,
        "section_path": [],
        "source_block_ids": [],
        "columns": [],
        "rows": [],
    }
    if isinstance(payload, ReportRagSourceCreate):
        if (
            report.knowledge_base_id is None
            or report.knowledge_base_id != payload.knowledge_base_id
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="请先为报告选择与证据一致的知识库",
            )
        document = await session.scalar(
            select(Document).where(
                Document.id == payload.document_id,
                Document.knowledge_base_id == payload.knowledge_base_id,
            )
        )
        if document is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="证据文档不存在")
        fields.update(
            knowledge_base_id=payload.knowledge_base_id,
            document_id=payload.document_id,
            document_title=payload.document_title,
            evidence_id=payload.evidence_id,
            chunk_id=payload.chunk_id,
            page_start=payload.page_start,
            page_end=payload.page_end,
            evidence_type=payload.evidence_type,
            snippet=payload.snippet,
            section_path=payload.section_path,
            source_block_ids=payload.source_block_ids,
        )
    elif isinstance(payload, ReportChatBISourceCreate):
        if report.datasource_id is None or report.datasource_id != payload.datasource_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="请先为报告选择与结果一致的数据源",
            )
        datasource = await session.scalar(
            select(ChatBIDataSourceRecord).where(ChatBIDataSourceRecord.id == payload.datasource_id)
        )
        if datasource is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="数据源不存在")
        payload_data = payload.model_dump(mode="json")
        fields.update(
            datasource_id=payload.datasource_id,
            document_title=payload.datasource_name,
            question=payload.question,
            answer=payload.answer,
            columns=payload_data["columns"],
            rows=payload_data["rows"],
            row_count=payload.row_count,
            truncated=payload.truncated,
            truncation_reason=payload.truncation_reason,
            redacted_sql=(
                redact_sql_literals(payload.redacted_sql)
                if payload.redacted_sql is not None
                else None
            ),
            chart_spec=payload_data["chart_spec"],
        )
    else:  # pragma: no cover - the discriminated union is exhaustive.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="来源类型无效",
        )
    return report, section, ReportSourceReference(**fields)


def _append_source_content(content: str, content_append: str) -> str:
    normalized_content = content.strip()
    normalized_append = content_append.strip()
    combined = (
        f"{normalized_content}\n\n{normalized_append}" if normalized_content else normalized_append
    )
    if len(combined) > REPORT_SECTION_CONTENT_MAX_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="章节正文过长，无法插入来源",
        )
    return combined


@router.delete("/{report_id}/sections/{section_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_section(
    report_id: UUID,
    section_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    section = await _get_section(session, report_id, section_id)
    report = await _get_report(session, report_id)
    await session.delete(section)
    await session.flush()
    _renumber(await _ordered_sections(session, report_id))
    report.updated_at = datetime.now(UTC)
    await session.commit()


@router.post(
    "/{report_id}/sections/{section_id}/sources",
    response_model=ReportSourceReferenceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_source_reference(
    report_id: UUID,
    section_id: UUID,
    payload: Annotated[ReportSourceCreate, Body(...)],
    session: AsyncSession = Depends(get_session),
) -> ReportSourceReferenceResponse:
    report, _section, reference = await _build_source_reference(
        session, report_id, section_id, payload
    )
    session.add(reference)
    report.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(reference)
    return _source_response(reference)


@router.post(
    "/{report_id}/sections/{section_id}/sources/insert",
    response_model=ReportSourceReferenceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def insert_source_reference(
    report_id: UUID,
    section_id: UUID,
    payload: ReportSourceInsertCreate,
    session: AsyncSession = Depends(get_session),
) -> ReportSourceReferenceResponse:
    report, section, reference = await _build_source_reference(
        session, report_id, section_id, payload.source
    )
    section.content = _append_source_content(section.content, payload.content_append)
    now = datetime.now(UTC)
    section.updated_at = now
    report.updated_at = now
    session.add(reference)
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    await session.refresh(reference)
    return _source_response(reference)


@router.delete(
    "/{report_id}/sources/{source_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_source_reference(
    report_id: UUID,
    source_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    reference = await session.scalar(
        select(ReportSourceReference).where(
            ReportSourceReference.id == source_id,
            ReportSourceReference.report_id == report_id,
        )
    )
    if reference is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="来源引用不存在")
    report = await _get_report(session, report_id)
    await session.delete(reference)
    report.updated_at = datetime.now(UTC)
    await session.commit()


__all__ = ["router"]
