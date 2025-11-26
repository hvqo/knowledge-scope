"""Structured PDF tables and deterministic table-analysis contracts.

The models in this module are the stable tool contract that a later agent
(M4) can reuse.  They intentionally contain no free-form code or expression
fields: an analysis request only selects evidence items and names one of the
supported deterministic operations.
"""

from __future__ import annotations

from typing import Final, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

TABLE_ANALYSIS_SCHEMA_VERSION: Final = "1.0"
EVIDENCE_ID_PATTERN: Final = r"^evidence_v1_[0-9a-f]{64}$"

MAX_TABLE_ROWS: Final = 500
MAX_TABLE_COLUMNS: Final = 50
MAX_TABLE_HTML_CHARS: Final = 2_000_000
MAX_CELL_CHARS: Final = 2_000
MAX_RESULT_ROWS: Final = 200
MAX_WARNINGS: Final = 30
MAX_MARKDOWN_TABLE_CHARS: Final = 500_000

TableOperation = Literal["select", "filter", "sort", "aggregate", "growth", "compare"]
ColumnRef = int | str
RowRef = int | str
CellKind = Literal["number", "percent", "missing", "text"]
ColumnKind = Literal["number", "percent", "text", "mixed"]
OperationStatus = Literal["ok", "not_computable", "error"]
TableRole = Literal["table_a", "table_b"]


class TableAnalysisError(RuntimeError):
    """Base class for controlled table-analysis failures."""

    code: str = "table_analysis_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class TableAccessError(TableAnalysisError):
    """The requested evidence/table cannot be accessed (missing, stale, wrong scope)."""

    code = "table_access_error"


class TableNotComputableError(TableAnalysisError):
    """The table cannot be parsed into reliable structured data."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class TableOperationError(TableAnalysisError):
    """An operation parameter or data state makes the operation fail."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class TableReference(BaseModel):
    """One explicitly selected table evidence item inside a knowledge base."""

    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    evidence_id: str = Field(..., pattern=EVIDENCE_ID_PATTERN)


class TableCell(BaseModel):
    """One normalized cell of a structured table."""

    text: str
    value: str | None = None
    kind: CellKind = "text"
    row_span: int = Field(default=1, ge=1, le=MAX_TABLE_ROWS)
    col_span: int = Field(default=1, ge=1, le=MAX_TABLE_COLUMNS)
    derived: bool = False


class TableColumn(BaseModel):
    """Header, unit and value-kind facts of one physical table column."""

    index: int = Field(..., ge=0)
    header: str
    raw_headers: list[str] = Field(default_factory=list)
    unit: str | None = None
    kind: ColumnKind
    year: int | None = None
    numeric_count: int = Field(default=0, ge=0)
    missing_count: int = Field(default=0, ge=0)


class StructuredTable(BaseModel):
    """A deterministic, provenance-carrying structured table."""

    schema_version: Literal["1.0"] = TABLE_ANALYSIS_SCHEMA_VERSION
    knowledge_base_id: UUID
    document_id: UUID
    evidence_id: str = Field(..., pattern=EVIDENCE_ID_PATTERN)
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    source_block_ids: list[str]
    section_path: list[str]
    source_fingerprint: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    caption: str | None = None
    unit_hint: str | None = None
    columns: list[TableColumn]
    rows: list[list[TableCell]]
    row_labels: list[str | None] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    computable: bool
    not_computable_reason: str | None = None


class SelectParameters(BaseModel):
    """Read explicit rows and columns, keeping the parsed numeric values."""

    model_config = ConfigDict(extra="forbid")

    columns: list[ColumnRef] = Field(default_factory=list, max_length=MAX_TABLE_COLUMNS)
    rows: list[RowRef] | None = Field(default=None, max_length=MAX_TABLE_ROWS)


class FilterParameters(BaseModel):
    """Keep rows whose value in one column satisfies a comparison."""

    model_config = ConfigDict(extra="forbid")

    column: ColumnRef
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains"] = "eq"
    value: str = Field(..., min_length=1, max_length=MAX_CELL_CHARS)
    columns: list[ColumnRef] = Field(default_factory=list, max_length=MAX_TABLE_COLUMNS)


class SortParameters(BaseModel):
    """Sort rows by one column value before returning them."""

    model_config = ConfigDict(extra="forbid")

    column: ColumnRef
    direction: Literal["asc", "desc"] = "asc"
    limit: int = Field(default=50, ge=1, le=MAX_RESULT_ROWS)
    columns: list[ColumnRef] = Field(default_factory=list, max_length=MAX_TABLE_COLUMNS)


class AggregateParameters(BaseModel):
    """Aggregate one numeric column, optionally grouped by another column."""

    model_config = ConfigDict(extra="forbid")

    column: ColumnRef
    functions: list[Literal["sum", "avg", "min", "max"]] = Field(
        ...,
        min_length=1,
        max_length=4,
    )
    group_by: ColumnRef | None = None


class GrowthParameters(BaseModel):
    """Compute absolute delta and relative growth between two values of one metric."""

    model_config = ConfigDict(extra="forbid")

    axis: Literal["rows", "columns"] = "rows"
    column: ColumnRef | None = None
    base_row: RowRef | None = None
    target_row: RowRef | None = None
    row: RowRef | None = None
    base_column: ColumnRef | None = None
    target_column: ColumnRef | None = None

    @model_validator(mode="after")
    def validate_axis_parameters(self) -> GrowthParameters:
        if self.axis == "rows":
            if self.column is None or self.base_row is None or self.target_row is None:
                raise ValueError("rows growth requires column, base_row and target_row")
            if (
                self.row is not None
                or self.base_column is not None
                or self.target_column is not None
            ):
                raise ValueError("rows growth cannot mix column-axis parameters")
        else:
            if self.row is None or self.base_column is None or self.target_column is None:
                raise ValueError("columns growth requires row, base_column and target_column")
            if self.column is not None or self.base_row is not None or self.target_row is not None:
                raise ValueError("columns growth cannot mix row-axis parameters")
        return self


class CompareParameters(BaseModel):
    """Compare the same metric of two tables after aligning their key rows."""

    model_config = ConfigDict(extra="forbid")

    key_column: str = Field(..., min_length=1, max_length=MAX_CELL_CHARS)
    value_column: str = Field(..., min_length=1, max_length=MAX_CELL_CHARS)
    keys: list[str] | None = Field(default=None, max_length=MAX_TABLE_ROWS)


AnalysisParameters = (
    SelectParameters
    | FilterParameters
    | SortParameters
    | AggregateParameters
    | GrowthParameters
    | CompareParameters
)

_PARAMETER_TYPES: dict[str, type[BaseModel]] = {
    "select": SelectParameters,
    "filter": FilterParameters,
    "sort": SortParameters,
    "aggregate": AggregateParameters,
    "growth": GrowthParameters,
    "compare": CompareParameters,
}


class TableAnalysisRequest(BaseModel):
    """One deterministic table operation over one or two explicitly selected tables."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = TABLE_ANALYSIS_SCHEMA_VERSION
    knowledge_base_id: UUID
    operation: TableOperation
    table: TableReference
    compare_table: TableReference | None = None
    parameters: AnalysisParameters

    @model_validator(mode="after")
    def validate_operation_parameters(self) -> TableAnalysisRequest:
        expected_type = _PARAMETER_TYPES[self.operation]
        if not isinstance(self.parameters, expected_type):
            raise ValueError(
                f"parameters for operation {self.operation!r} must be {expected_type.__name__}"
            )
        if self.operation == "compare":
            if self.compare_table is None:
                raise ValueError("compare requires compare_table")
            if self.compare_table == self.table:
                raise ValueError("compare requires two different tables")
        elif self.compare_table is not None:
            raise ValueError("compare_table is only valid for the compare operation")
        return self


class ResultColumn(BaseModel):
    """One output column of a table-analysis result."""

    name: str
    kind: ColumnKind
    unit: str | None = None


class ResultRow(BaseModel):
    """One output row; every cell is JSON serializable (numbers are decimal strings)."""

    cells: list[str | None]


class TableSource(BaseModel):
    """Deterministic provenance of one input table."""

    table_role: TableRole
    document_id: UUID
    evidence_id: str = Field(..., pattern=EVIDENCE_ID_PATTERN)
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    source_block_ids: list[str]


class TableSummary(BaseModel):
    """Compact public metadata about one input table, including parse limits."""

    table_role: TableRole
    document_id: UUID
    evidence_id: str = Field(..., pattern=EVIDENCE_ID_PATTERN)
    caption: str | None = None
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    source_block_ids: list[str]
    columns: list[TableColumn] = Field(default_factory=list)
    row_count: int = Field(default=0, ge=0)
    computable: bool
    warnings: list[str] = Field(default_factory=list)
    not_computable_reason: str | None = None


class TableAnalysisResult(BaseModel):
    """Deterministic tool result with provenance, formula and controlled status."""

    schema_version: Literal["1.0"] = TABLE_ANALYSIS_SCHEMA_VERSION
    operation: TableOperation
    status: OperationStatus
    error_code: str | None = None
    message: str | None = None
    columns: list[ResultColumn] = Field(default_factory=list)
    rows: list[ResultRow] = Field(default_factory=list)
    formula: str | None = None
    inputs: dict[str, str] = Field(default_factory=dict)
    units: list[str] = Field(default_factory=list)
    sources: list[TableSource] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tables: list[TableSummary] = Field(default_factory=list)


__all__ = [
    "MAX_CELL_CHARS",
    "MAX_MARKDOWN_TABLE_CHARS",
    "MAX_RESULT_ROWS",
    "MAX_TABLE_COLUMNS",
    "MAX_TABLE_HTML_CHARS",
    "MAX_TABLE_ROWS",
    "MAX_WARNINGS",
    "AggregateParameters",
    "AnalysisParameters",
    "ColumnRef",
    "CompareParameters",
    "FilterParameters",
    "GrowthParameters",
    "ResultColumn",
    "ResultRow",
    "SelectParameters",
    "SortParameters",
    "StructuredTable",
    "TableAccessError",
    "TableAnalysisError",
    "TableAnalysisRequest",
    "TableAnalysisResult",
    "TableColumn",
    "TableNotComputableError",
    "TableOperationError",
    "TableReference",
    "TableSource",
    "TableSummary",
]
