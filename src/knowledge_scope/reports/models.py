"""Persistent report workspace models."""

from __future__ import annotations

from datetime import datetime
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import Mapped, mapped_column

from knowledge_scope.shared.database import Base

REPORT_TITLE_MAX_LENGTH: Final = 200
REPORT_SECTION_TITLE_MAX_LENGTH: Final = 200
REPORT_SECTION_CONTENT_MAX_LENGTH: Final = 100_000
REPORT_SOURCE_KIND_MAX_LENGTH: Final = 32
REPORT_SOURCE_TITLE_MAX_LENGTH: Final = 255
REPORT_SOURCE_DOCUMENT_TITLE_MAX_LENGTH: Final = 512
REPORT_SOURCE_EVIDENCE_TYPE_MAX_LENGTH: Final = 32
REPORT_SOURCE_ID_MAX_LENGTH: Final = 255
REPORT_SOURCE_LINEAGE_ITEM_MAX_LENGTH: Final = 255
REPORT_SOURCE_SNIPPET_MAX_LENGTH: Final = 4_000
REPORT_SOURCE_QUESTION_MAX_LENGTH: Final = 4_000
REPORT_SOURCE_ANSWER_MAX_LENGTH: Final = 8_000
REPORT_SOURCE_SQL_MAX_LENGTH: Final = 100_000


class Report(Base):
    """A user-authored report with optional KB and datasource context."""

    __tablename__ = "reports"
    __table_args__ = (
        CheckConstraint(
            "title = btrim(title) AND btrim(title) <> ''",
            name="ck_reports_title_trimmed_non_empty",
        ),
    )

    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(String(REPORT_TITLE_MAX_LENGTH), nullable=False)
    knowledge_base_id: Mapped[UUID | None] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="RESTRICT"),
        nullable=True,
    )
    datasource_id: Mapped[UUID | None] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("chatbi_data_sources.id", ondelete="RESTRICT"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ReportSection(Base):
    """One ordered, editable section in a report."""

    __tablename__ = "report_sections"
    __table_args__ = (
        CheckConstraint(
            "title = btrim(title) AND btrim(title) <> ''",
            name="ck_report_sections_title_trimmed_non_empty",
        ),
        CheckConstraint("position >= 0", name="ck_report_sections_position_non_negative"),
        Index("ix_report_sections_report_position", "report_id", "position"),
    )

    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    report_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("reports.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(REPORT_SECTION_TITLE_MAX_LENGTH), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ReportSourceReference(Base):
    """A bounded reference rendered in one report section."""

    __tablename__ = "report_source_references"
    __table_args__ = (
        CheckConstraint(
            "source_kind IN ('rag_evidence', 'chatbi_result')",
            name="ck_report_source_references_kind_known",
        ),
        CheckConstraint(
            "page_start IS NULL OR page_start >= 1",
            name="ck_report_source_references_page_start_positive",
        ),
        CheckConstraint(
            "page_end IS NULL OR page_end >= 1",
            name="ck_report_source_references_page_end_positive",
        ),
        Index("ix_report_source_references_section", "section_id"),
    )

    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    report_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("reports.id", ondelete="CASCADE"),
        nullable=False,
    )
    section_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("report_sections.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_kind: Mapped[str] = mapped_column(String(REPORT_SOURCE_KIND_MAX_LENGTH), nullable=False)
    title: Mapped[str] = mapped_column(String(REPORT_SOURCE_TITLE_MAX_LENGTH), nullable=False)
    knowledge_base_id: Mapped[UUID | None] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="SET NULL"),
        nullable=True,
    )
    datasource_id: Mapped[UUID | None] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("chatbi_data_sources.id", ondelete="SET NULL"),
        nullable=True,
    )
    document_id: Mapped[UUID | None] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    document_title: Mapped[str | None] = mapped_column(
        String(REPORT_SOURCE_DOCUMENT_TITLE_MAX_LENGTH), nullable=True
    )
    evidence_id: Mapped[str | None] = mapped_column(
        String(REPORT_SOURCE_ID_MAX_LENGTH), nullable=True
    )
    chunk_id: Mapped[str | None] = mapped_column(String(REPORT_SOURCE_ID_MAX_LENGTH), nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    evidence_type: Mapped[str | None] = mapped_column(
        String(REPORT_SOURCE_EVIDENCE_TYPE_MAX_LENGTH), nullable=True
    )
    snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    section_path: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    source_block_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    columns: Mapped[list[dict[str, object]]] = mapped_column(JSONB, nullable=False, default=list)
    rows: Mapped[list[list[object]]] = mapped_column(JSONB, nullable=False, default=list)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    truncated: Mapped[bool | None] = mapped_column(nullable=True)
    truncation_reason: Mapped[str | None] = mapped_column(
        String(REPORT_SOURCE_EVIDENCE_TYPE_MAX_LENGTH), nullable=True
    )
    redacted_sql: Mapped[str | None] = mapped_column(Text, nullable=True)
    chart_spec: Mapped[dict[str, object] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
