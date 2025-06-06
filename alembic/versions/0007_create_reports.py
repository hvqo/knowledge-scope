"""Create persistent report workspace tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    uuid = postgresql.UUID(as_uuid=True)
    jsonb = postgresql.JSONB()

    op.create_table(
        "reports",
        sa.Column("id", uuid, nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("knowledge_base_id", uuid, nullable=True),
        sa.Column("datasource_id", uuid, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "title = btrim(title) AND btrim(title) <> ''",
            name="ck_reports_title_trimmed_non_empty",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"],
            ["knowledge_bases.id"],
            name="fk_reports_knowledge_base_id",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["datasource_id"],
            ["chatbi_data_sources.id"],
            name="fk_reports_datasource_id",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "report_sections",
        sa.Column("id", uuid, nullable=False),
        sa.Column("report_id", uuid, nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("content", sa.Text(), server_default="", nullable=False),
        sa.Column("position", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "title = btrim(title) AND btrim(title) <> ''",
            name="ck_report_sections_title_trimmed_non_empty",
        ),
        sa.CheckConstraint("position >= 0", name="ck_report_sections_position_non_negative"),
        sa.ForeignKeyConstraint(
            ["report_id"],
            ["reports.id"],
            name="fk_report_sections_report_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_report_sections_report_position",
        "report_sections",
        ["report_id", "position"],
    )

    op.create_table(
        "report_source_references",
        sa.Column("id", uuid, nullable=False),
        sa.Column("report_id", uuid, nullable=False),
        sa.Column("section_id", uuid, nullable=False),
        sa.Column("source_kind", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("knowledge_base_id", uuid, nullable=True),
        sa.Column("datasource_id", uuid, nullable=True),
        sa.Column("document_id", uuid, nullable=True),
        sa.Column("document_title", sa.String(length=512), nullable=True),
        sa.Column("evidence_id", sa.String(length=255), nullable=True),
        sa.Column("chunk_id", sa.String(length=255), nullable=True),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("evidence_type", sa.String(length=32), nullable=True),
        sa.Column("snippet", sa.Text(), nullable=True),
        sa.Column("section_path", jsonb, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column(
            "source_block_ids",
            jsonb,
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("columns", jsonb, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("rows", jsonb, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("truncated", sa.Boolean(), nullable=True),
        sa.Column("truncation_reason", sa.String(length=32), nullable=True),
        sa.Column("redacted_sql", sa.Text(), nullable=True),
        sa.Column("chart_spec", jsonb, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "source_kind IN ('rag_evidence', 'chatbi_result')",
            name="ck_report_source_references_kind_known",
        ),
        sa.CheckConstraint(
            "page_start IS NULL OR page_start >= 1",
            name="ck_report_source_references_page_start_positive",
        ),
        sa.CheckConstraint(
            "page_end IS NULL OR page_end >= 1",
            name="ck_report_source_references_page_end_positive",
        ),
        sa.ForeignKeyConstraint(
            ["report_id"],
            ["reports.id"],
            name="fk_report_source_references_report_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["section_id"],
            ["report_sections.id"],
            name="fk_report_source_references_section_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"],
            ["knowledge_bases.id"],
            name="fk_report_source_references_knowledge_base_id",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["datasource_id"],
            ["chatbi_data_sources.id"],
            name="fk_report_source_references_datasource_id",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_report_source_references_document_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_report_source_references_section",
        "report_source_references",
        ["section_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_report_source_references_section", table_name="report_source_references")
    op.drop_table("report_source_references")
    op.drop_index("ix_report_sections_report_position", table_name="report_sections")
    op.drop_table("report_sections")
    op.drop_table("reports")
