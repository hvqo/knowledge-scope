"""Evidence-backed table analysis service.

The service is the reusable core behind both the read-only HTTP API (M2) and the
future Agent tool contract (M4).  Callers never provide code or SQL: every
operation is a fixed, deterministic function over a parsed table, and every
result carries the Evidence lineage, the applied formula and the key inputs.
"""
# ruff: noqa: RUF001

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from knowledge_scope.documents.assets import (
    AssetAccessError,
    get_evidence,
)
from knowledge_scope.documents.models import Document
from knowledge_scope.evidence.models import MultimodalEvidence
from knowledge_scope.shared.config import Settings
from knowledge_scope.table_analysis.models import (
    MAX_TABLE_COLUMNS,
    MAX_TABLE_ROWS,
    TABLE_ANALYSIS_SCHEMA_VERSION,
    AggregateParameters,
    AnalysisParameters,
    CompareParameters,
    FilterParameters,
    GrowthParameters,
    SelectParameters,
    SortParameters,
    StructuredTable,
    TableAccessError,
    TableAnalysisError,
    TableAnalysisRequest,
    TableAnalysisResult,
    TableNotComputableError,
    TableOperation,
    TableOperationError,
    TableReference,
    TableSource,
    TableSummary,
)
from knowledge_scope.table_analysis.operations import (
    OperationOutcome,
    run_aggregate,
    run_compare,
    run_filter,
    run_growth,
    run_select,
    run_sort,
)
from knowledge_scope.table_analysis.parsing import (
    ParsedTableData,
    detect_unit_hint,
    parse_table_content,
)

_CONTEXT_CHAR_LIMIT = 500


@dataclass(frozen=True, slots=True)
class _LoadedTable:
    table: StructuredTable
    summary: TableSummary
    source: TableSource


def _representation_content(
    evidence: MultimodalEvidence,
    representation_type: str,
) -> str | None:
    for representation in evidence.representations:
        if (
            representation.representation_type == representation_type
            and representation.content is not None
            and representation.searchable
        ):
            return representation.content
    return None


def _context_text(evidence: MultimodalEvidence) -> str | None:
    parts: list[str] = []
    for representation in evidence.representations:
        if representation.representation_type != "context":
            continue
        if representation.content is None:
            continue
        parts.append(representation.content)
    if not parts:
        return None
    return " ".join(parts)[:_CONTEXT_CHAR_LIMIT]


def _table_representations(
    evidence: MultimodalEvidence,
) -> tuple[str | None, str | None, str | None]:
    html = _representation_content(evidence, "html")
    markdown = _representation_content(evidence, "markdown")
    caption = _representation_content(evidence, "caption")
    return html, markdown, caption


def _parse_evidence_table(
    evidence: MultimodalEvidence,
) -> tuple[ParsedTableData, str | None, str | None]:
    html, markdown, caption = _table_representations(evidence)
    representation_type: Literal["html", "markdown"] | None = None
    content: str | None = None
    if html is not None:
        representation_type, content = "html", html
    elif markdown is not None:
        representation_type, content = "markdown", markdown

    if content is None:
        if evidence.lineage.asset_refs:
            raise TableNotComputableError(
                "table_only_image",
                "该表格证据仅保留图片素材，当前版本不支持图片表格计算",
            )
        raise TableNotComputableError(
            "table_content_missing",
            "表格证据缺少可解析的 HTML/Markdown 文本内容",
        )

    context = _context_text(evidence)
    unit_hint = detect_unit_hint(caption, context)
    parsed = parse_table_content(
        content,
        representation_type,
        caption=caption,
        unit_hint=unit_hint,
    )
    return parsed, caption, unit_hint


def _validate_evidence_document(
    evidence: MultimodalEvidence,
    document: Document,
    knowledge_base_id: object,
) -> None:
    lineage = evidence.lineage
    if lineage.document_id != document.id:
        raise TableAccessError("证据不属于指定文档")
    if lineage.knowledge_base_id != document.knowledge_base_id:
        raise TableAccessError("证据不属于指定知识库")
    if lineage.knowledge_base_id != knowledge_base_id:
        raise TableAccessError("证据不属于请求的知识库")


class TableAnalysisService:
    """Load table Evidence and run one explicit deterministic operation."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def analyze(
        self,
        request: TableAnalysisRequest,
        *,
        document: Document,
        compare_document: Document | None = None,
    ) -> TableAnalysisResult:
        """Analyze one table evidence reference, optionally comparing two tables.

        Access and staleness problems raise :class:`TableAccessError` so the HTTP
        layer can answer with 404.  Parse/compute problems that are inherent to
        the table are reported inside the returned result instead of raising.
        """
        tables: list[TableSummary] = []
        sources: list[TableSource] = []
        try:
            primary = self._load_table(
                request,
                document=document,
                reference=request.table,
                role="primary",
            )
        except TableNotComputableError as error:
            return self._failure(request, error=error, status="not_computable")
        tables.append(primary.summary)
        sources.append(primary.source)

        compare: _LoadedTable | None = None
        if request.operation == "compare":
            if request.compare_table is None or compare_document is None:
                raise TableAccessError("对比计算缺少第二张表格")
            try:
                compare = self._load_table(
                    request,
                    document=compare_document,
                    reference=request.compare_table,
                    role="compare",
                )
            except TableNotComputableError as error:
                return self._failure(
                    request,
                    error=error,
                    status="not_computable",
                    tables=tables,
                    sources=sources,
                )
            if compare.table.evidence_id == primary.table.evidence_id:
                raise TableOperationError(
                    "same_table",
                    "对比计算需要两张不同的表格证据",
                )
            tables.append(compare.summary)
            sources.append(compare.source)

        try:
            outcome = self._run_operation(
                request.operation,
                primary.table,
                compare.table if compare is not None else None,
                request.parameters,
            )
        except TableNotComputableError as error:
            return self._failure(
                request,
                error=error,
                status="not_computable",
                tables=tables,
                sources=sources,
            )
        except TableOperationError as error:
            return self._failure(
                request,
                error=error,
                status="error",
                tables=tables,
                sources=sources,
            )
        if not primary.table.computable:
            return self._failure(
                request,
                error=TableNotComputableError(
                    primary.table.not_computable_reason or "not_computable",
                    "该表格无法进行确定性计算",
                ),
                status="not_computable",
                tables=tables,
                sources=sources,
            )

        return TableAnalysisResult(
            operation=request.operation,
            status=outcome.status,
            error_code=outcome.error_code,
            message=outcome.message,
            columns=outcome.columns,
            rows=outcome.rows,
            formula=outcome.formula,
            inputs=outcome.inputs,
            units=outcome.units,
            sources=sources,
            warnings=outcome.warnings,
            tables=tables,
        )

    def _load_table(
        self,
        request: TableAnalysisRequest,
        *,
        document: Document,
        reference: TableReference,
        role: str,
    ) -> _LoadedTable:
        if reference.document_id != document.id:
            raise TableAccessError("证据所属文档与请求不一致")
        snapshot = get_evidence(self._settings, document, reference.evidence_id)
        evidence = snapshot.evidence
        _validate_evidence_document(evidence, document, request.knowledge_base_id)

        if evidence.modality != "table":
            raise TableNotComputableError(
                "table_missing",
                f"证据类型为 {evidence.modality}，不是表格证据",
            )
        parsed, caption, unit_hint = _parse_evidence_table(evidence)
        if len(parsed.rows) > MAX_TABLE_ROWS:
            raise TableNotComputableError(
                "table_too_large",
                f"表格数据行数 {len(parsed.rows)} 超过可分析上限 {MAX_TABLE_ROWS}",
            )
        if len(parsed.columns) > MAX_TABLE_COLUMNS:
            raise TableNotComputableError(
                "table_too_large",
                f"表格列数 {len(parsed.columns)} 超过可分析上限 {MAX_TABLE_COLUMNS}",
            )

        lineage = evidence.lineage
        table = StructuredTable(
            schema_version=TABLE_ANALYSIS_SCHEMA_VERSION,
            knowledge_base_id=lineage.knowledge_base_id,
            document_id=lineage.document_id,
            evidence_id=evidence.evidence_id,
            page_start=lineage.page_start,
            page_end=lineage.page_end,
            source_block_ids=list(lineage.source_block_ids),
            section_path=list(lineage.section_path),
            source_fingerprint=lineage.source_fingerprint,
            caption=caption,
            unit_hint=unit_hint,
            columns=parsed.columns,
            rows=parsed.rows,
            row_labels=parsed.row_labels,
            warnings=list(parsed.warnings),
            computable=True,
            not_computable_reason=None,
        )
        table_role: str = "table_a" if role == "primary" else "table_b"
        source = TableSource(
            table_role=table_role,  # type: ignore[arg-type]
            document_id=lineage.document_id,
            evidence_id=evidence.evidence_id,
            page_start=lineage.page_start,
            page_end=lineage.page_end,
            source_block_ids=list(lineage.source_block_ids),
        )
        summary = TableSummary(
            table_role=table_role,  # type: ignore[arg-type]
            document_id=lineage.document_id,
            evidence_id=evidence.evidence_id,
            caption=caption,
            page_start=lineage.page_start,
            page_end=lineage.page_end,
            source_block_ids=list(lineage.source_block_ids),
            columns=list(parsed.columns),
            row_count=len(parsed.rows),
            computable=True,
            warnings=list(parsed.warnings),
            not_computable_reason=None,
        )
        return _LoadedTable(table=table, summary=summary, source=source)

    def _run_operation(
        self,
        operation: TableOperation,
        table: StructuredTable,
        compare_table: StructuredTable | None,
        parameters: AnalysisParameters,
    ) -> OperationOutcome:
        if operation == "select":
            if not isinstance(parameters, SelectParameters):
                raise TableOperationError("invalid_parameter", "select 参数类型不匹配")
            return run_select(table, parameters)
        if operation == "filter":
            if not isinstance(parameters, FilterParameters):
                raise TableOperationError("invalid_parameter", "filter 参数类型不匹配")
            return run_filter(table, parameters)
        if operation == "sort":
            if not isinstance(parameters, SortParameters):
                raise TableOperationError("invalid_parameter", "sort 参数类型不匹配")
            return run_sort(table, parameters)
        if operation == "aggregate":
            if not isinstance(parameters, AggregateParameters):
                raise TableOperationError("invalid_parameter", "aggregate 参数类型不匹配")
            return run_aggregate(table, parameters)
        if operation == "growth":
            if not isinstance(parameters, GrowthParameters):
                raise TableOperationError("invalid_parameter", "growth 参数类型不匹配")
            return run_growth(table, parameters)
        if not isinstance(parameters, CompareParameters):
            raise TableOperationError("invalid_parameter", "compare 参数类型不匹配")
        if compare_table is None:
            raise TableOperationError("invalid_parameter", "compare 缺少第二张表格")
        return run_compare(table, compare_table, parameters)

    @staticmethod
    def _failure(
        request: TableAnalysisRequest,
        *,
        error: TableAnalysisError,
        status: str,
        tables: list[TableSummary] | None = None,
        sources: list[TableSource] | None = None,
    ) -> TableAnalysisResult:
        return TableAnalysisResult(
            operation=request.operation,
            status=status,  # type: ignore[arg-type]
            error_code=error.code,
            message=str(error),
            sources=list(sources or []),
            tables=list(tables or []),
        )


def analyze_table_evidence(
    settings: Settings,
    request: TableAnalysisRequest,
    *,
    document: Document,
    compare_document: Document | None = None,
) -> TableAnalysisResult:
    """Convenience wrapper used by the HTTP layer and the future Agent tool."""
    return TableAnalysisService(settings).analyze(
        request,
        document=document,
        compare_document=compare_document,
    )


__all__ = [
    "AssetAccessError",
    "TableAnalysisService",
    "analyze_table_evidence",
]
