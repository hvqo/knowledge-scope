"""ChatBI datasource and bounded read-only analysis API."""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_scope.reports.models import Report
from knowledge_scope.shared.config import Settings
from knowledge_scope.shared.database import get_session

from .agent import ChatBIResult
from .discovery import SchemaDiscoveryService, create_postgres_schema_discovery_service
from .errors import ChatBIError, ChatBIErrorCategory
from .models import ChatBIDataSourceRecord
from .policy import default_query_policy
from .schema_models import SchemaDiscoveryResult
from .schemas import (
    ColumnMetadata,
    DataSource,
    DataSourceCreate,
    DataSourceListResponse,
    DataSourcePublic,
    DataSourceUpdate,
    QueryEligibilityReasonCode,
    QueryLifecycleState,
    QueryTruncationReason,
    ScalarValue,
)

router = APIRouter(prefix="/chatbi/data-sources", tags=["chatbi"])


class ChatBIQuestionRequest(BaseModel):
    """Untrusted user intent for one registered datasource."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    question: StrictStr = Field(min_length=1, max_length=10_000)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must contain non-whitespace characters")
        return normalized


class ChatBIEligibilityPublic(BaseModel):
    """Product-safe eligibility state without gate or schema provenance."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["eligible", "clarify", "refuse", "unavailable"]
    reason_code: QueryEligibilityReasonCode
    user_message: str | None = Field(default=None, max_length=1_000)
    clarification_question: str | None = Field(default=None, max_length=1_000)


class ChatBIAnalysisResponse(BaseModel):
    """Bounded product projection of one ChatBI agent result."""

    model_config = ConfigDict(extra="forbid")

    query_id: UUID
    datasource_id: UUID
    execution_status: QueryLifecycleState
    eligibility: ChatBIEligibilityPublic | None
    answer: str | None = Field(default=None, max_length=20_000)
    columns: list[ColumnMetadata]
    rows: list[list[ScalarValue]]
    row_count: StrictInt = Field(ge=0)
    truncated: bool
    truncation_reason: QueryTruncationReason | None
    redacted_sql: str | None = Field(default=None, max_length=100_000)
    warnings: list[str]
    error_message: str | None = Field(default=None, max_length=2_000)
    elapsed_ms: StrictFloat = Field(ge=0)


def _public_eligibility(result: ChatBIResult) -> ChatBIEligibilityPublic | None:
    if result.eligibility is not None:
        decision = result.eligibility
        return ChatBIEligibilityPublic(
            decision=decision.decision,
            reason_code=decision.reason_code,
            user_message=decision.user_message,
            clarification_question=decision.clarification_question,
        )
    if result.execution_status is QueryLifecycleState.SUCCEEDED:
        return ChatBIEligibilityPublic(
            decision="eligible",
            reason_code=QueryEligibilityReasonCode.ELIGIBLE_ANALYTICAL,
        )
    return None


def _to_analysis_response(result: ChatBIResult, elapsed_ms: float) -> ChatBIAnalysisResponse:
    return ChatBIAnalysisResponse(
        query_id=result.query_id,
        datasource_id=result.datasource_id,
        execution_status=result.execution_status,
        eligibility=_public_eligibility(result),
        answer=result.answer,
        columns=result.columns,
        rows=result.rows,
        row_count=result.row_count,
        truncated=result.truncated,
        truncation_reason=result.truncation_reason,
        redacted_sql=result.redacted_sql,
        warnings=result.warnings,
        error_message=result.error_message,
        elapsed_ms=max(elapsed_ms, 0.0),
    )


def _analysis_http_error(error: ChatBIError) -> HTTPException:
    if error.category is ChatBIErrorCategory.DATASOURCE_NOT_FOUND:
        error_status = status.HTTP_404_NOT_FOUND
    elif error.category in {
        ChatBIErrorCategory.INVALID_QUERY,
        ChatBIErrorCategory.UNSUPPORTED_DIALECT,
        ChatBIErrorCategory.POLICY_VIOLATION,
    }:
        error_status = status.HTTP_422_UNPROCESSABLE_CONTENT
    elif error.category is ChatBIErrorCategory.DATASOURCE_DISABLED:
        error_status = status.HTTP_409_CONFLICT
    else:
        error_status = status.HTTP_502_BAD_GATEWAY
    return HTTPException(
        status_code=error_status,
        detail={"category": error.category.value, "message": error.safe_message},
    )


def _to_public(record: ChatBIDataSourceRecord) -> DataSourcePublic:
    """Project a record without exposing its opaque connection reference."""
    return DataSourcePublic(
        id=record.id,
        display_name=record.display_name,
        dialect=record.dialect,
        enabled=record.enabled,
        default_database=record.default_database,
        default_schema=record.default_schema,
        connection_configured=bool(record.connection_ref),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _to_internal(record: ChatBIDataSourceRecord) -> DataSource:
    """Build the internal datasource contract without exposing it in a response."""
    return DataSource.model_validate(
        {
            "id": record.id,
            "display_name": record.display_name,
            "dialect": record.dialect,
            "enabled": record.enabled,
            "connection_ref": record.connection_ref,
            "default_database": record.default_database,
            "default_schema": record.default_schema,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }
    )


def _build_schema_discovery_service(settings: Settings) -> SchemaDiscoveryService:
    """Create a stateless discovery service for one API request."""
    return create_postgres_schema_discovery_service(
        connection_timeout_seconds=settings.chatbi_schema_connection_timeout_seconds,
        statement_timeout_ms=settings.chatbi_statement_timeout_ms,
    )


def _schema_discovery_http_error(error: ChatBIError) -> HTTPException:
    if error.category is ChatBIErrorCategory.DATASOURCE_DISABLED:
        error_status = status.HTTP_409_CONFLICT
    elif error.category is ChatBIErrorCategory.UNSUPPORTED_DIALECT:
        error_status = status.HTTP_422_UNPROCESSABLE_CONTENT
    else:
        error_status = status.HTTP_502_BAD_GATEWAY
    return HTTPException(
        status_code=error_status,
        detail={"category": error.category.value, "message": error.safe_message},
    )


async def _get_data_source(session: AsyncSession, datasource_id: UUID) -> ChatBIDataSourceRecord:
    record = await session.scalar(
        select(ChatBIDataSourceRecord).where(ChatBIDataSourceRecord.id == datasource_id)
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found",
        )
    return record


@router.post("", response_model=DataSourcePublic, status_code=status.HTTP_201_CREATED)
async def create_data_source(
    payload: DataSourceCreate,
    session: AsyncSession = Depends(get_session),
) -> DataSourcePublic:
    """Register safe metadata and an opaque external connection reference."""
    record = ChatBIDataSourceRecord(
        display_name=payload.display_name,
        dialect=payload.dialect.value,
        enabled=payload.enabled,
        connection_ref=payload.connection_ref,
        default_database=payload.default_database,
        default_schema=payload.default_schema,
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return _to_public(record)


@router.get("", response_model=DataSourceListResponse)
async def list_data_sources(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> DataSourceListResponse:
    total = int(await session.scalar(select(func.count()).select_from(ChatBIDataSourceRecord)) or 0)
    result = await session.scalars(
        select(ChatBIDataSourceRecord)
        .order_by(ChatBIDataSourceRecord.created_at.desc(), ChatBIDataSourceRecord.id.desc())
        .offset(offset)
        .limit(limit)
    )
    return DataSourceListResponse(
        items=[_to_public(record) for record in result.all()],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{datasource_id}/schema", response_model=SchemaDiscoveryResult)
async def discover_data_source_schema(
    datasource_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> SchemaDiscoveryResult:
    """Discover allow-listed external schema metadata without executing SQL."""
    record = await _get_data_source(session, datasource_id)
    settings = request.app.state.settings
    service = _build_schema_discovery_service(settings)
    try:
        return await service.discover(
            _to_internal(record),
            default_query_policy(settings),
            max_chars=settings.chatbi_schema_context_max_chars,
        )
    except ChatBIError as error:
        raise _schema_discovery_http_error(error) from None


@router.post(
    "/{datasource_id}/ask",
    response_model=ChatBIAnalysisResponse,
    status_code=status.HTTP_200_OK,
)
async def ask_data_source(
    datasource_id: UUID,
    payload: ChatBIQuestionRequest,
    request: Request,
) -> ChatBIAnalysisResponse:
    """Run the registered-datasource ChatBI Agent without exposing raw SQL input."""
    agent = getattr(request.app.state, "chatbi_agent_service", None)
    if not callable(getattr(agent, "ask", None)):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ChatBI analysis service is not initialized",
        )

    started = perf_counter()
    settings: Settings = request.app.state.settings
    try:
        result = await agent.ask(
            datasource_id,
            payload.question,
            policy=default_query_policy(settings),
            max_chars=settings.chatbi_schema_context_max_chars,
        )
    except ChatBIError as error:
        raise _analysis_http_error(error) from None
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "category": ChatBIErrorCategory.DATASOURCE_UNAVAILABLE.value,
                "message": "ChatBI analysis is temporarily unavailable",
            },
        ) from None
    return _to_analysis_response(result, (perf_counter() - started) * 1_000)


@router.get("/{datasource_id}", response_model=DataSourcePublic)
async def get_data_source(
    datasource_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> DataSourcePublic:
    return _to_public(await _get_data_source(session, datasource_id))


@router.patch("/{datasource_id}", response_model=DataSourcePublic)
async def update_data_source(
    datasource_id: UUID,
    payload: DataSourceUpdate,
    session: AsyncSession = Depends(get_session),
) -> DataSourcePublic:
    record = await _get_data_source(session, datasource_id)
    for field_name, value in payload.model_dump(exclude_unset=True).items():
        setattr(record, field_name, value)
    record.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(record)
    return _to_public(record)


@router.delete("/{datasource_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_data_source(
    datasource_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> Response:
    record = await _get_data_source(session, datasource_id)
    report_id = await session.scalar(
        select(Report.id).where(Report.datasource_id == record.id).limit(1)
    )
    if report_id is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="数据源仍被报告引用, 请先删除报告",
        )
    await session.delete(record)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
