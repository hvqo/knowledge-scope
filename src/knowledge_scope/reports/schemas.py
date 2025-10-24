"""Bounded API contracts for the report workspace."""

from __future__ import annotations

import json
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from knowledge_scope.chatbi.schemas import ScalarValue

from .models import (
    REPORT_SECTION_CONTENT_MAX_LENGTH,
    REPORT_SECTION_TITLE_MAX_LENGTH,
    REPORT_SOURCE_ANSWER_MAX_LENGTH,
    REPORT_SOURCE_DOCUMENT_TITLE_MAX_LENGTH,
    REPORT_SOURCE_ID_MAX_LENGTH,
    REPORT_SOURCE_LINEAGE_ITEM_MAX_LENGTH,
    REPORT_SOURCE_QUESTION_MAX_LENGTH,
    REPORT_SOURCE_SNIPPET_MAX_LENGTH,
    REPORT_SOURCE_SQL_MAX_LENGTH,
    REPORT_SOURCE_TITLE_MAX_LENGTH,
    REPORT_TITLE_MAX_LENGTH,
)


class _ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


def _trimmed(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must contain non-whitespace characters")
    return normalized


def _optional_trimmed(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _trimmed(value, field_name)


class ReportSummary(_ReportModel):
    id: UUID
    title: str
    knowledge_base_id: UUID | None
    datasource_id: UUID | None
    created_at: str
    updated_at: str


class ReportCreate(_ReportModel):
    title: str = Field(min_length=1, max_length=REPORT_TITLE_MAX_LENGTH)
    knowledge_base_id: UUID | None = None
    datasource_id: UUID | None = None

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        return _trimmed(value, "title")


class ReportUpdate(_ReportModel):
    title: str | None = Field(default=None, max_length=REPORT_TITLE_MAX_LENGTH)
    knowledge_base_id: UUID | None = None
    datasource_id: UUID | None = None

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        return _trimmed(value, "title") if value is not None else None

    @model_validator(mode="after")
    def require_update(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("at least one report field must be provided")
        if "title" in self.model_fields_set and self.title is None:
            raise ValueError("title cannot be null")
        return self


class ReportSectionCreate(_ReportModel):
    title: str = Field(default="新章节", min_length=1, max_length=REPORT_SECTION_TITLE_MAX_LENGTH)
    content: str = Field(default="", max_length=REPORT_SECTION_CONTENT_MAX_LENGTH)
    position: int | None = Field(default=None, ge=0, le=1_000)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        return _trimmed(value, "title")


class ReportSectionUpdate(_ReportModel):
    title: str | None = Field(default=None, max_length=REPORT_SECTION_TITLE_MAX_LENGTH)
    content: str | None = Field(default=None, max_length=REPORT_SECTION_CONTENT_MAX_LENGTH)
    position: int | None = Field(default=None, ge=0, le=1_000)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        return _trimmed(value, "title") if value is not None else None

    @model_validator(mode="after")
    def require_update(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("at least one section field must be provided")
        if "title" in self.model_fields_set and self.title is None:
            raise ValueError("title cannot be null")
        if "content" in self.model_fields_set and self.content is None:
            raise ValueError("content cannot be null")
        return self


class ReportColumn(_ReportModel):
    name: str = Field(min_length=1, max_length=255)
    data_type: str = Field(min_length=1, max_length=255)
    nullable: bool
    ordinal: int = Field(ge=0, le=100)

    @field_validator("name", "data_type")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return _trimmed(value, "column value")


class ReportChartPoint(_ReportModel):
    label: str = Field(min_length=1, max_length=255)
    value: float


class ReportChartSpec(_ReportModel):
    kind: Literal["bar", "line", "pie"]
    category_column: str = Field(min_length=1, max_length=255)
    value_column: str = Field(min_length=1, max_length=255)
    points: list[ReportChartPoint] = Field(max_length=24)


class ReportRagSourceCreate(_ReportModel):
    kind: Literal["rag_evidence"]
    title: str = Field(min_length=1, max_length=REPORT_SOURCE_TITLE_MAX_LENGTH)
    knowledge_base_id: UUID
    document_id: UUID
    evidence_id: str | None = Field(default=None, max_length=REPORT_SOURCE_ID_MAX_LENGTH)
    chunk_id: str | None = Field(default=None, max_length=REPORT_SOURCE_ID_MAX_LENGTH)
    document_title: str = Field(max_length=REPORT_SOURCE_DOCUMENT_TITLE_MAX_LENGTH)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    evidence_type: Literal["text", "image", "table", "formula"] = "text"
    snippet: str | None = Field(default=None, max_length=REPORT_SOURCE_SNIPPET_MAX_LENGTH)
    section_path: list[Annotated[str, Field(max_length=REPORT_SOURCE_LINEAGE_ITEM_MAX_LENGTH)]] = (
        Field(
            default_factory=list,
            max_length=32,
        )
    )
    source_block_ids: list[
        Annotated[str, Field(max_length=REPORT_SOURCE_LINEAGE_ITEM_MAX_LENGTH)]
    ] = Field(default_factory=list, max_length=64)

    @field_validator("title", "document_title")
    @classmethod
    def normalize_titles(cls, value: str) -> str:
        return _trimmed(value, "source title")

    @field_validator("evidence_id", "chunk_id")
    @classmethod
    def normalize_ids(cls, value: str | None) -> str | None:
        return _optional_trimmed(value, "source id")

    @field_validator("snippet")
    @classmethod
    def normalize_snippet(cls, value: str | None) -> str | None:
        return _optional_trimmed(value, "snippet")

    @field_validator("section_path", "source_block_ids")
    @classmethod
    def validate_reference_values(cls, values: list[str]) -> list[str]:
        normalized = [_trimmed(value, "reference value") for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("reference values must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_page_range(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError("page_end must not be less than page_start")
        if self.evidence_id is None and self.chunk_id is None:
            raise ValueError("RAG source requires evidence_id or chunk_id")
        return self


class ReportChatBISourceCreate(_ReportModel):
    kind: Literal["chatbi_result"]
    title: str = Field(min_length=1, max_length=REPORT_SOURCE_TITLE_MAX_LENGTH)
    datasource_id: UUID
    datasource_name: str = Field(min_length=1, max_length=REPORT_SOURCE_DOCUMENT_TITLE_MAX_LENGTH)
    question: str = Field(min_length=1, max_length=REPORT_SOURCE_QUESTION_MAX_LENGTH)
    answer: str | None = Field(default=None, max_length=REPORT_SOURCE_ANSWER_MAX_LENGTH)
    columns: list[ReportColumn] = Field(max_length=50)
    rows: list[list[ScalarValue]] = Field(max_length=100)
    row_count: int = Field(ge=0)
    truncated: bool = False
    truncation_reason: Literal["row_limit", "payload_bytes"] | None = None
    redacted_sql: str | None = Field(default=None, max_length=REPORT_SOURCE_SQL_MAX_LENGTH)
    chart_spec: ReportChartSpec | None = None

    @field_validator("title", "datasource_name", "question")
    @classmethod
    def normalize_titles(cls, value: str) -> str:
        return _trimmed(value, "source value")

    @field_validator("answer", "redacted_sql")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        return _optional_trimmed(value, "source value")

    @model_validator(mode="after")
    def validate_result_snapshot(self) -> Self:
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("ChatBI source rows must match the column contract")
        if self.truncated != (self.truncation_reason is not None):
            raise ValueError("ChatBI source truncation metadata is inconsistent")
        try:
            encoded = json.dumps(
                {"columns": [column.model_dump() for column in self.columns], "rows": self.rows},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("ChatBI source snapshot is not JSON-safe") from error
        if len(encoded.encode("utf-8")) > 200_000:
            raise ValueError("ChatBI source snapshot is too large")
        return self


ReportSourceCreate = Annotated[
    ReportRagSourceCreate | ReportChatBISourceCreate,
    Field(discriminator="kind"),
]


class ReportSourceInsertCreate(_ReportModel):
    source: ReportSourceCreate
    content_append: str = Field(
        min_length=1,
        max_length=REPORT_SECTION_CONTENT_MAX_LENGTH,
    )

    @field_validator("content_append")
    @classmethod
    def normalize_content_append(cls, value: str) -> str:
        return _trimmed(value, "content_append")


class ReportSourceReferenceResponse(_ReportModel):
    id: UUID
    section_id: UUID
    kind: Literal["rag_evidence", "chatbi_result"]
    title: str
    knowledge_base_id: UUID | None
    datasource_id: UUID | None
    document_id: UUID | None
    document_title: str | None
    evidence_id: str | None
    chunk_id: str | None
    page_start: int | None
    page_end: int | None
    evidence_type: str | None
    snippet: str | None
    section_path: list[str]
    source_block_ids: list[str]
    question: str | None
    answer: str | None
    columns: list[ReportColumn]
    rows: list[list[ScalarValue]]
    row_count: int | None
    truncated: bool | None
    truncation_reason: str | None
    redacted_sql: str | None
    chart_spec: ReportChartSpec | None
    created_at: str


class ReportSectionResponse(_ReportModel):
    id: UUID
    report_id: UUID
    title: str
    content: str
    position: int
    sources: list[ReportSourceReferenceResponse]
    created_at: str
    updated_at: str


class ReportDetailResponse(_ReportModel):
    id: UUID
    title: str
    knowledge_base_id: UUID | None
    datasource_id: UUID | None
    sections: list[ReportSectionResponse]
    created_at: str
    updated_at: str


class ReportListResponse(_ReportModel):
    items: list[ReportSummary]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


REPORT_AI_MAX_SOURCE_REFERENCES = 12
REPORT_AI_INSTRUCTION_MAX_LENGTH = 2_000


class ReportAIContextRequest(_ReportModel):
    """Bounded source selection shared by report-generation previews."""

    source_ids: list[UUID] = Field(
        default_factory=list,
        max_length=REPORT_AI_MAX_SOURCE_REFERENCES,
    )
    instruction: str | None = Field(default=None, max_length=REPORT_AI_INSTRUCTION_MAX_LENGTH)

    @field_validator("instruction")
    @classmethod
    def normalize_instruction(cls, value: str | None) -> str | None:
        return _optional_trimmed(value, "instruction")

    @field_validator("source_ids")
    @classmethod
    def require_unique_sources(cls, value: list[UUID]) -> list[UUID]:
        if len(value) != len(set(value)):
            raise ValueError("source_ids must be unique")
        return value


class ReportAIOutlineRequest(ReportAIContextRequest):
    topic: str | None = Field(default=None, max_length=REPORT_TITLE_MAX_LENGTH)

    @field_validator("topic")
    @classmethod
    def normalize_topic(cls, value: str | None) -> str | None:
        return _optional_trimmed(value, "topic")


class ReportAIEditRequest(ReportAIContextRequest):
    operation: Literal["rewrite", "expand", "summarize"]


class ReportAIOutlineItem(_ReportModel):
    title: str = Field(min_length=1, max_length=REPORT_SECTION_TITLE_MAX_LENGTH)
    summary: str = Field(min_length=1, max_length=1_000)

    @field_validator("title", "summary")
    @classmethod
    def normalize_outline_text(cls, value: str) -> str:
        return _trimmed(value, "outline value")


class ReportAIOutlineResponse(_ReportModel):
    items: list[ReportAIOutlineItem] = Field(min_length=1, max_length=12)


class ReportAICitation(_ReportModel):
    marker: str = Field(pattern=r"^S-[0-9A-F]{12}$")
    title: str = Field(min_length=1, max_length=REPORT_SOURCE_TITLE_MAX_LENGTH)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)


class ReportAIDraftResponse(_ReportModel):
    operation: Literal["generate", "rewrite", "expand", "summarize"]
    content: str = Field(min_length=1, max_length=REPORT_SECTION_CONTENT_MAX_LENGTH)
    citations: list[ReportAICitation] = Field(default_factory=list, max_length=12)

    @field_validator("content")
    @classmethod
    def normalize_generated_content(cls, value: str) -> str:
        return _trimmed(value, "generated content")


__all__ = [
    "ReportAICitation",
    "ReportAIContextRequest",
    "ReportAIDraftResponse",
    "ReportAIEditRequest",
    "ReportAIOutlineItem",
    "ReportAIOutlineRequest",
    "ReportAIOutlineResponse",
    "ReportChatBISourceCreate",
    "ReportCreate",
    "ReportDetailResponse",
    "ReportListResponse",
    "ReportRagSourceCreate",
    "ReportSectionCreate",
    "ReportSectionResponse",
    "ReportSectionUpdate",
    "ReportSourceCreate",
    "ReportSourceInsertCreate",
    "ReportSourceReferenceResponse",
    "ReportUpdate",
]
