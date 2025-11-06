"""Parse MinerU HTML or Markdown table representations into structured tables.

Only first-party standard-library parsers are used.  The parser never executes
or trusts markup: HTML is inspected through :class:`html.parser.HTMLParser`
(script/style contents are skipped), merged cells are normalized only when the
grid can be reconstructed unambiguously, and anything else is reported as a
controlled ``not_computable`` condition instead of guessed data.
"""
# ruff: noqa: RUF001, RUF002

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from typing import Final, Literal

from knowledge_scope.table_analysis.models import (
    MAX_CELL_CHARS,
    MAX_MARKDOWN_TABLE_CHARS,
    MAX_TABLE_COLUMNS,
    MAX_TABLE_HTML_CHARS,
    MAX_TABLE_ROWS,
    MAX_WARNINGS,
    ColumnKind,
    TableCell,
    TableColumn,
    TableNotComputableError,
)

MISSING_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "",
        "-",
        "--",
        "---",
        "—",
        "–",
        "－",
        "/",
        "\\",
        "n/a",
        "na",
        "nan",
        "null",
        "none",
        "无",
        "暂无",
        "缺失",
        "不详",
    }
)

NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
YEAR_RE: Final[re.Pattern[str]] = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
YEAR_ONLY_RE: Final[re.Pattern[str]] = re.compile(r"^((?:19|20)\d{2})\s*年?$")
UNIT_PAREN_RE: Final[re.Pattern[str]] = re.compile(r"[（(]([^()（）]{1,32})[)）]")
UNIT_LABEL_RE: Final[re.Pattern[str]] = re.compile(r"单位\s*[:：]\s*([^;；,，。\s]{1,16})")
UNIT_TOKENS: Final[tuple[str, ...]] = (
    "元",
    "美元",
    "欧元",
    "日元",
    "英镑",
    "港元",
    "亿元",
    "万元",
    "亿",
    "万",
    "吨",
    "千克",
    "公斤",
    "克",
    "米",
    "千米",
    "公里",
    "公顷",
    "平方米",
    "立方米",
    "升",
    "毫升",
    "人",
    "人次",
    "户",
    "辆",
    "台",
    "件",
    "个",
    "百分点",
    "%",
)
_ENGLISH_UNIT_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z][A-Za-z0-9 .,/%_\-]{0,39}$")
_UNIT_OPERATOR_CHARS: Final[frozenset[str]] = frozenset("^=+*/\\<>")

MARKDOWN_SEPARATOR_CELL_RE: Final[re.Pattern[str]] = re.compile(r"^:?-{3,}:?$")


@dataclass(frozen=True, slots=True)
class _RawCell:
    """One parsed ``td``/``th`` cell before grid normalization."""

    text: str
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False


@dataclass(frozen=True, slots=True)
class _PlacedCell:
    """A cell mapped onto one grid position, including span-derived copies."""

    text: str
    row_span: int
    col_span: int
    derived: bool


@dataclass(frozen=True, slots=True)
class ParsedTableData:
    """Deterministic structured result of parsing one table representation."""

    columns: list[TableColumn]
    rows: list[list[TableCell]]
    row_labels: list[str | None]
    warnings: list[str]
    caption: str | None
    unit_hint: str | None


def _normalize_text(value: str) -> str:
    return " ".join(value.replace("\u00a0", " ").split()).strip()


def _append_warning(warnings: list[str], message: str) -> None:
    if message not in warnings and len(warnings) < MAX_WARNINGS:
        warnings.append(message)


def _span_value(value: str | None, *, limit: int) -> int:
    try:
        parsed = int(value or "1")
    except (TypeError, ValueError):
        return 1
    if parsed < 1:
        return 1
    return min(parsed, limit)


def _parse_number(text: str) -> tuple[Decimal | None, str]:
    """Return ``(decimal_value, kind)`` with percent values kept as percent points."""
    raw = text.strip()
    if raw.casefold() in MISSING_TOKENS:
        return None, "missing"
    compact = raw.replace("\u00a0", " ").replace("，", ",").replace(" ", "")
    percent = compact.endswith(("%", "％"))
    if percent:
        compact = compact[:-1]
    negative = compact.startswith("(") and compact.endswith(")")
    if negative:
        compact = compact[1:-1]
    compact = compact.replace(",", "").replace("−", "-")
    if not NUMBER_RE.fullmatch(compact):
        return None, "text"
    try:
        value = Decimal(compact)
    except InvalidOperation:
        return None, "text"
    if negative:
        value = -value
    return value, ("percent" if percent else "number")


def _is_year_only(text: str) -> bool:
    return YEAR_ONLY_RE.fullmatch(_normalize_text(text)) is not None


def _extract_unit_and_year(header: str) -> tuple[str | None, int | None]:
    year_match = YEAR_RE.search(header)
    year = int(year_match.group(1)) if year_match else None
    for match in UNIT_PAREN_RE.finditer(header):
        candidate = _normalize_text(match.group(1))
        if not candidate or _is_year_only(candidate):
            continue
        if any(character in _UNIT_OPERATOR_CHARS for character in candidate):
            continue
        if any(token in candidate for token in UNIT_TOKENS):
            return candidate, year
        if _ENGLISH_UNIT_RE.fullmatch(candidate) is not None:
            return candidate, year
    if header.rstrip().endswith(("%", "％")):
        return "%", year
    return None, year


def _detect_unit_hint(caption: str | None, context: str | None) -> str | None:
    for text in (caption, context):
        if not text:
            continue
        label_match = UNIT_LABEL_RE.search(text)
        if label_match:
            return _normalize_text(label_match.group(1))
        for match in UNIT_PAREN_RE.finditer(text):
            candidate = _normalize_text(match.group(1))
            if candidate and "单位" in candidate:
                return _normalize_text(candidate.replace("单位", "").strip(" :："))
    return None


class _HTMLTableCollector(HTMLParser):
    """Collect rows/cells of the first ``<table>`` without trusting its content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[_RawCell]]] = []
        self._table_depth = 0
        self._skip_depth = 0
        self._rows: list[list[_RawCell]] = []
        self._row: list[_RawCell] | None = None
        self._cell_text: list[str] | None = None
        self._cell_is_header = False
        self._cell_rowspan = 1
        self._cell_colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._rows = []
                self._row = None
                self._cell_text = None
            return
        if self._table_depth != 1:
            return
        if tag == "tr":
            self._finish_row()
            self._row = []
        elif tag in {"td", "th"}:
            self._finish_cell()
            attributes = {key.lower(): value for key, value in attrs}
            self._cell_text = []
            self._cell_is_header = tag == "th"
            self._cell_rowspan = _span_value(attributes.get("rowspan"), limit=MAX_TABLE_ROWS)
            self._cell_colspan = _span_value(attributes.get("colspan"), limit=MAX_TABLE_COLUMNS)
        elif tag == "br" and self._cell_text is not None:
            self._cell_text.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "table":
            if self._table_depth == 0:
                return
            self._table_depth -= 1
            if self._table_depth == 0:
                self._finish_row()
                if self._rows:
                    self.tables.append(self._rows)
                self._rows = []
                self._row = None
                self._cell_text = None
            return
        if self._table_depth != 1:
            return
        if tag in {"td", "th"}:
            self._finish_cell()
        elif tag == "tr":
            self._finish_row()

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._cell_text is None:
            return
        self._cell_text.append(data)

    def _finish_cell(self) -> None:
        if self._cell_text is None:
            return
        text = _normalize_text(" ".join(self._cell_text))
        if self._row is None:
            self._row = []
        self._row.append(
            _RawCell(
                text=text,
                rowspan=self._cell_rowspan,
                colspan=self._cell_colspan,
                is_header=self._cell_is_header,
            )
        )
        self._cell_text = None

    def _finish_row(self) -> None:
        if self._row:
            self._rows.append(self._row)
        self._row = None


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.replace("\\|", "|") for cell in stripped.split("|")]
    return [_normalize_text(cell) for cell in cells]


def _is_markdown_separator(cells: list[str]) -> bool:
    return bool(cells) and all(MARKDOWN_SEPARATOR_CELL_RE.fullmatch(cell) for cell in cells)


def _collect_markdown_rows(content: str) -> list[list[_RawCell]]:
    lines = [line for line in content.splitlines() if line.strip()]
    if not lines:
        raise TableNotComputableError("empty_table", "表格内容为空")
    header_index = -1
    for index in range(min(len(lines) - 1, 20)):
        cells = _split_markdown_row(lines[index])
        next_cells = _split_markdown_row(lines[index + 1])
        if "|" in lines[index] and _is_markdown_separator(next_cells):
            header_index = index
            break
    if header_index < 0:
        raise TableNotComputableError("unsupported_table_format", "无法识别的 Markdown 表格结构")
    rows: list[list[_RawCell]] = []
    header_cells = _split_markdown_row(lines[header_index])
    rows.append([_RawCell(text=cell, is_header=True) for cell in header_cells])
    for line in lines[header_index + 2 :]:
        if "|" not in line:
            break
        cells = _split_markdown_row(line)
        if _is_markdown_separator(cells):
            break
        rows.append([_RawCell(text=cell) for cell in cells])
    return rows


def _build_grid(
    raw_rows: list[list[_RawCell]],
    warnings: list[str],
) -> tuple[list[list[_PlacedCell]], set[tuple[int, int]]]:
    if not raw_rows:
        raise TableNotComputableError("empty_table", "表格内容为空")
    if len(raw_rows) > MAX_TABLE_ROWS:
        raise TableNotComputableError("table_too_large", "表格行数超过可分析上限")
    grid: dict[tuple[int, int], _PlacedCell] = {}
    origin_positions: set[tuple[int, int]] = set()
    width = 0
    for row_index, row in enumerate(raw_rows):
        column_index = 0
        for cell in row:
            while (row_index, column_index) in grid:
                column_index += 1
            placed = _PlacedCell(
                text=cell.text,
                row_span=cell.rowspan,
                col_span=cell.colspan,
                derived=False,
            )
            origin_positions.add((row_index, column_index))
            for row_offset in range(cell.rowspan):
                for column_offset in range(cell.colspan):
                    position = (row_index + row_offset, column_index + column_offset)
                    if position in grid:
                        raise TableNotComputableError(
                            "complex_merge",
                            "表格合并单元格存在冲突，无法可靠还原",
                        )
                    grid[position] = _PlacedCell(
                        text=cell.text,
                        row_span=cell.rowspan,
                        col_span=cell.colspan,
                        derived=bool(row_offset or column_offset),
                    )
            width = max(width, column_index + cell.colspan)
            column_index += cell.colspan
    if width > MAX_TABLE_COLUMNS:
        raise TableNotComputableError("table_too_large", "表格列数超过可分析上限")
    matrix: list[list[_PlacedCell]] = []
    for row_index in range(len(raw_rows)):
        row_cells: list[_PlacedCell] = []
        for column_index in range(width):
            placed = grid.get((row_index, column_index))
            if placed is None:
                row_cells.append(_PlacedCell(text="", row_span=1, col_span=1, derived=False))
            else:
                row_cells.append(placed)
        matrix.append(row_cells)
    return matrix, origin_positions


def _row_numeric_score(row: list[_PlacedCell]) -> int:
    score = 0
    for cell in row:
        if cell.derived:
            continue
        value, kind = _parse_number(cell.text)
        if value is None:
            continue
        if kind == "number" and _is_year_only(cell.text):
            continue
        score += 1
    return score


def _column_kind(kinds: list[str]) -> ColumnKind:
    numeric = sum(1 for kind in kinds if kind in {"number", "percent"})
    missing = sum(1 for kind in kinds if kind == "missing")
    text = len(kinds) - numeric - missing
    has_number = any(kind == "number" for kind in kinds)
    has_percent = any(kind == "percent" for kind in kinds)
    if numeric == 0:
        return "text"
    if text > 0 or (has_number and has_percent):
        return "mixed"
    return "percent" if has_percent and not has_number else "number"


def _resolve_columns_from_grid(
    matrix: list[list[_PlacedCell]],
    header_rows: list[list[_PlacedCell]],
    data_start: int,
    unit_hint: str | None,
    warnings: list[str],
) -> tuple[list[TableColumn], list[list[TableCell]], list[str | None]]:
    width = len(matrix[0]) if matrix else 0
    columns: list[TableColumn] = []
    for column_index in range(width):
        header_parts: list[str] = []
        for header_row in header_rows:
            text = header_row[column_index].text
            if text and text not in header_parts:
                header_parts.append(text)
        header = " / ".join(header_parts) if header_parts else f"列{column_index + 1}"
        if len(header) > MAX_CELL_CHARS:
            header = header[:MAX_CELL_CHARS]
        unit, year = _extract_unit_and_year(header)
        kinds: list[str] = []
        for row_index in range(data_start, len(matrix)):
            cell = matrix[row_index][column_index]
            _, kind = _parse_number(cell.text)
            kinds.append(kind)
        kind = _column_kind(kinds)
        columns.append(
            TableColumn(
                index=column_index,
                header=header,
                raw_headers=header_parts,
                unit=unit,
                kind=kind,
                year=year,
                numeric_count=sum(1 for item in kinds if item in {"number", "percent"}),
                missing_count=sum(1 for item in kinds if item == "missing"),
            )
        )
    if unit_hint is not None and any(
        column.kind in {"number", "percent"} and column.unit is None for column in columns
    ):
        applied = False
        for column in columns:
            if column.kind in {"number", "percent"} and column.unit is None:
                column.unit = unit_hint
                applied = True
        if applied:
            _append_warning(warnings, f"列单位取自表格说明：{unit_hint}")

    data_rows: list[list[TableCell]] = []
    for row_index in range(data_start, len(matrix)):
        row_cells: list[TableCell] = []
        for column_index in range(width):
            placed = matrix[row_index][column_index]
            text = placed.text[:MAX_CELL_CHARS]
            value, kind = _parse_number(text)
            row_cells.append(
                TableCell(
                    text=text,
                    value=str(value) if value is not None else None,
                    kind=kind,  # type: ignore[arg-type]
                    row_span=placed.row_span,
                    col_span=placed.col_span,
                    derived=placed.derived,
                )
            )
        data_rows.append(row_cells)

    row_labels: list[str | None] = []
    if data_rows and columns and columns[0].kind == "text":
        labels = [row[0].text for row in data_rows]
        if all(label for label in labels):
            row_labels = labels
    if not row_labels:
        row_labels = [None] * len(data_rows)
    return columns, data_rows, row_labels


def assemble_structured_table(
    raw_rows: list[list[_RawCell]],
    *,
    source_kind: Literal["html", "markdown"],
    caption: str | None,
    unit_hint: str | None,
    initial_warnings: list[str] | None = None,
) -> ParsedTableData:
    """Normalize raw rows (with spans) into a deterministic structured table."""
    warnings: list[str] = list(initial_warnings or [])
    matrix, origin_positions = _build_grid(raw_rows, warnings)
    if not matrix or not matrix[0]:
        raise TableNotComputableError("empty_table", "表格内容为空")

    data_start: int | None = None
    for row_index, row in enumerate(matrix):
        numeric_count = _row_numeric_score(row)
        if numeric_count >= 1 and numeric_count >= max(1, (len(row) * 2 + 4) // 5):
            data_start = row_index
            break
    if data_start is None:
        raise TableNotComputableError("no_numeric_data", "表格中未找到可计算的数值数据")

    if data_start == 0:
        header_rows = matrix[:1]
        _append_warning(warnings, "未识别到独立表头，已将首行同时作为列名与数据")
    else:
        header_rows = matrix[:data_start]
        if len(header_rows) > 1:
            _append_warning(warnings, "表格存在多层表头，已合并为列名")
    if len(header_rows) > 5:
        raise TableNotComputableError("complex_header", "表格表头层级过多，无法可靠识别")

    for row_index in range(data_start, len(matrix)):
        for column_index in range(len(matrix[row_index])):
            if (row_index, column_index) not in origin_positions:
                continue
            placed = matrix[row_index][column_index]
            if placed.row_span <= 1 and placed.col_span <= 1:
                continue
            if column_index == 0 and placed.col_span == 1:
                _append_warning(warnings, "首列存在跨行单元格，已按行标签展开")
                continue
            raise TableNotComputableError(
                "complex_merge",
                "表格数据区存在无法可靠还原的合并单元格",
            )

    columns, data_rows, row_labels = _resolve_columns_from_grid(
        matrix,
        header_rows,
        data_start,
        unit_hint,
        warnings,
    )
    return ParsedTableData(
        columns=columns,
        rows=data_rows,
        row_labels=row_labels,
        warnings=warnings[:MAX_WARNINGS],
        caption=caption,
        unit_hint=unit_hint,
    )


def parse_html_table(
    content: str,
    *,
    caption: str | None = None,
    unit_hint: str | None = None,
) -> ParsedTableData:
    """Parse the first ``<table>`` of a MinerU HTML table representation."""
    if len(content) > MAX_TABLE_HTML_CHARS:
        raise TableNotComputableError("table_too_large", "表格 HTML 内容超过可分析上限")
    collector = _HTMLTableCollector()
    try:
        collector.feed(content)
        collector.close()
    except AssertionError as error:  # pragma: no cover - defensive HTMLParser guard
        raise TableNotComputableError("invalid_html", "表格 HTML 无法解析") from error
    if not collector.tables:
        raise TableNotComputableError("unsupported_table_format", "未在 HTML 中找到表格结构")
    warnings: list[str] = []
    if len(collector.tables) > 1:
        _append_warning(warnings, "HTML 中包含多个表格，仅解析第一个表格")
    return assemble_structured_table(
        collector.tables[0],
        source_kind="html",
        caption=caption,
        unit_hint=unit_hint,
        initial_warnings=warnings,
    )


def parse_markdown_table(
    content: str,
    *,
    caption: str | None = None,
    unit_hint: str | None = None,
) -> ParsedTableData:
    """Parse a Markdown pipe table representation."""
    if len(content) > MAX_MARKDOWN_TABLE_CHARS:
        raise TableNotComputableError("table_too_large", "表格 Markdown 内容超过可分析上限")
    rows = _collect_markdown_rows(content)
    return assemble_structured_table(
        rows,
        source_kind="markdown",
        caption=caption,
        unit_hint=unit_hint,
    )


def parse_table_content(
    content: str,
    representation_type: Literal["html", "markdown"],
    *,
    caption: str | None = None,
    unit_hint: str | None = None,
) -> ParsedTableData:
    """Dispatch to the parser matching the canonical table representation type."""
    if representation_type == "html":
        return parse_html_table(content, caption=caption, unit_hint=unit_hint)
    return parse_markdown_table(content, caption=caption, unit_hint=unit_hint)


def detect_unit_hint(caption: str | None, context: str | None = None) -> str | None:
    """Extract a table-level unit hint such as ``单位：亿美元`` from nearby text."""
    return _detect_unit_hint(caption, context)


__all__ = [
    "ParsedTableData",
    "assemble_structured_table",
    "detect_unit_hint",
    "parse_html_table",
    "parse_markdown_table",
    "parse_table_content",
]
