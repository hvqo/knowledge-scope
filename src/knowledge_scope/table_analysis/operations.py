"""Deterministic table operations implemented with :class:`decimal.Decimal`.

No dynamic code, SQL or free-form expression is evaluated here.  Every
operation either returns explicit values with the formula and inputs that
produced them, or a controlled ``not_computable`` / ``error`` outcome.
"""
# ruff: noqa: RUF001

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from typing import Literal

from knowledge_scope.table_analysis.models import (
    MAX_RESULT_ROWS,
    MAX_WARNINGS,
    AggregateParameters,
    ColumnKind,
    ColumnRef,
    CompareParameters,
    FilterParameters,
    GrowthParameters,
    ResultColumn,
    ResultRow,
    RowRef,
    SelectParameters,
    SortParameters,
    StructuredTable,
    TableCell,
    TableColumn,
    TableNotComputableError,
    TableOperationError,
)

PERCENT_QUANTUM: Decimal = Decimal("0.0001")
MAX_FORMULA_CHARS = 2_000


@dataclass(frozen=True, slots=True)
class OperationOutcome:
    """Result of one deterministic operation before provenance is attached."""

    status: Literal["ok", "not_computable", "error"]
    error_code: str | None = None
    message: str | None = None
    columns: list[ResultColumn] = field(default_factory=list)
    rows: list[ResultRow] = field(default_factory=list)
    formula: str | None = None
    inputs: dict[str, str] = field(default_factory=dict)
    units: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _normalize_label(value: str) -> str:
    return " ".join(value.replace("\u00a0", " ").split()).strip().casefold()


def _append_warning(warnings: list[str], message: str) -> None:
    if message not in warnings and len(warnings) < MAX_WARNINGS:
        warnings.append(message)


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    return str(value)


def _percent_text(value: Decimal) -> str:
    return str(value.quantize(PERCENT_QUANTUM, rounding=ROUND_HALF_UP))


def _cell_decimal(cell: TableCell) -> Decimal | None:
    if cell.value is None:
        return None
    try:
        return Decimal(cell.value)
    except InvalidOperation:
        return None


def _cell_display(cell: TableCell) -> str | None:
    if cell.value is not None:
        return cell.value
    text = cell.text.strip()
    return text or None


def _result_column(column: TableColumn) -> ResultColumn:
    return ResultColumn(name=column.header, kind=column.kind, unit=column.unit)


def _resolve_column(table: StructuredTable, reference: ColumnRef) -> TableColumn:
    if isinstance(reference, int):
        if reference < 0 or reference >= len(table.columns):
            raise TableOperationError(
                "unknown_column",
                f"列序号 {reference} 超出范围（共 {len(table.columns)} 列）",
            )
        return table.columns[reference]
    target = _normalize_label(reference)
    if not target:
        raise TableOperationError("invalid_parameter", "列名不能为空")
    matches = [
        column
        for column in table.columns
        if _normalize_label(column.header) == target
        or any(_normalize_label(raw) == target for raw in column.raw_headers)
    ]
    if not matches:
        raise TableOperationError("unknown_column", f"未找到列：{reference}")
    if len(matches) > 1:
        raise TableOperationError(
            "ambiguous_column",
            f"列名“{reference}”对应多列，请改用列序号",
        )
    return matches[0]


def _resolve_columns(
    table: StructuredTable,
    references: list[ColumnRef],
) -> list[TableColumn]:
    if not references:
        return list(table.columns)
    resolved: list[TableColumn] = []
    seen: set[int] = set()
    for reference in references:
        column = _resolve_column(table, reference)
        if column.index not in seen:
            resolved.append(column)
            seen.add(column.index)
    return resolved


def _resolve_row(table: StructuredTable, reference: RowRef) -> int:
    row_count = len(table.rows)
    if isinstance(reference, int):
        if reference < 0 or reference >= row_count:
            raise TableOperationError(
                "unknown_row",
                f"行序号 {reference} 超出范围（共 {row_count} 行数据）",
            )
        return reference
    target = _normalize_label(reference)
    matches = [
        index
        for index, label in enumerate(table.row_labels)
        if label is not None and _normalize_label(label) == target
    ]
    if not matches:
        raise TableOperationError("unknown_row", f"未找到行：{reference}")
    if len(matches) > 1:
        raise TableOperationError(
            "ambiguous_row",
            f"行标签“{reference}”重复，无法唯一确定，请改用行序号",
        )
    return matches[0]


def _require_numeric_column(column: TableColumn) -> None:
    if column.kind not in {"number", "percent"}:
        raise TableOperationError(
            "column_not_numeric",
            f"列“{column.header}”不是数值列（类型：{column.kind}）",
        )


def _numeric_values(table: StructuredTable, column: TableColumn) -> list[Decimal | None]:
    return [_cell_decimal(row[column.index]) for row in table.rows]


def _parse_filter_number(value: str) -> Decimal:
    text = value.strip().replace("\u00a0", " ").replace("，", ",").replace(" ", "")
    percent = text.endswith(("%", "％"))
    if percent:
        text = text[:-1]
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = text.replace(",", "").replace("−", "-")
    if not text:
        raise TableOperationError("invalid_parameter", "筛选值不能为空")
    try:
        parsed = Decimal(text)
    except InvalidOperation:
        raise TableOperationError("invalid_parameter", f"筛选值不是数值：{value}") from None
    if negative:
        parsed = -parsed
    return parsed


def _row_cells_display(row: list[TableCell], columns: list[TableColumn]) -> list[str | None]:
    return [_cell_display(row[column.index]) for column in columns]


def _truncate_rows(
    rows: list[ResultRow],
    warnings: list[str],
) -> list[ResultRow]:
    if len(rows) <= MAX_RESULT_ROWS:
        return rows
    _append_warning(warnings, f"结果行数超过上限，仅返回前 {MAX_RESULT_ROWS} 行")
    return rows[:MAX_RESULT_ROWS]


def run_select(table: StructuredTable, parameters: SelectParameters) -> OperationOutcome:
    """Read explicit rows/columns as parsed values."""
    columns = _resolve_columns(table, parameters.columns)
    warnings: list[str] = []
    if parameters.rows is None:
        row_indexes = list(range(len(table.rows)))
    else:
        row_indexes = [_resolve_row(table, reference) for reference in parameters.rows]
    rows = [
        ResultRow(cells=_row_cells_display(table.rows[index], columns)) for index in row_indexes
    ]
    rows = _truncate_rows(rows, warnings)
    return OperationOutcome(
        status="ok",
        columns=[_result_column(column) for column in columns],
        rows=rows,
        inputs={"row_count": str(len(table.rows)), "selected_rows": str(len(row_indexes))},
        units=[column.unit for column in columns if column.unit],
        warnings=warnings,
    )


def run_filter(table: StructuredTable, parameters: FilterParameters) -> OperationOutcome:
    """Keep rows whose value satisfies one explicit comparison."""
    column = _resolve_column(table, parameters.column)
    columns = _resolve_columns(table, parameters.columns)
    warnings: list[str] = []
    if column.kind in {"number", "percent"}:
        if parameters.operator == "contains":
            raise TableOperationError("invalid_parameter", "数值列不支持 contains 运算符")
        target = _parse_filter_number(parameters.value)
        compared: list[tuple[int, Decimal]] = []
        for index, cell in enumerate(table.rows):
            value = _cell_decimal(cell[column.index])
            if value is None:
                continue
            matched = {
                "eq": value == target,
                "ne": value != target,
                "gt": value > target,
                "gte": value >= target,
                "lt": value < target,
                "lte": value <= target,
            }[parameters.operator]
            if matched:
                compared.append((index, value))
        if compared and len(compared) < len(table.rows):
            _append_warning(warnings, "部分缺失值未参与比较")
        selected = [index for index, _ in compared]
        expression = f"{column.header} {parameters.operator} {parameters.value}"
        inputs = {
            "column": column.header,
            "operator": parameters.operator,
            "value": parameters.value,
            "matched_rows": str(len(selected)),
        }
    else:
        if parameters.operator in {"gt", "gte", "lt", "lte"}:
            raise TableOperationError("invalid_parameter", "文本列只支持 eq / ne / contains")
        target_text = _normalize_label(parameters.value)
        selected = []
        for index, cell in enumerate(table.rows):
            text = _normalize_label(cell[column.index].text)
            matched = {
                "eq": text == target_text,
                "ne": text != target_text,
                "contains": target_text in text,
            }[parameters.operator]
            if matched:
                selected.append(index)
        expression = f"{column.header} {parameters.operator} {parameters.value}"
        inputs = {
            "column": column.header,
            "operator": parameters.operator,
            "value": parameters.value,
            "matched_rows": str(len(selected)),
        }
    rows = [ResultRow(cells=_row_cells_display(table.rows[index], columns)) for index in selected]
    rows = _truncate_rows(rows, warnings)
    return OperationOutcome(
        status="ok",
        columns=[_result_column(item) for item in columns],
        rows=rows,
        formula=f"filter({expression})",
        inputs=inputs,
        units=[item.unit for item in columns if item.unit],
        warnings=warnings,
    )


def run_sort(table: StructuredTable, parameters: SortParameters) -> OperationOutcome:
    """Sort data rows by one numeric or text column."""
    column = _resolve_column(table, parameters.column)
    columns = _resolve_columns(table, parameters.columns)
    warnings: list[str] = []
    descending = parameters.direction == "desc"

    if column.kind in {"number", "percent"}:
        keyed = [(index, _cell_decimal(row[column.index])) for index, row in enumerate(table.rows)]
        numeric = [(index, value) for index, value in keyed if value is not None]
        if len(numeric) < len(keyed):
            _append_warning(warnings, "部分缺失值未参与排序，已排在末尾")
        numeric.sort(key=lambda item: item[1], reverse=descending)
        ordered = [index for index, _ in numeric] + [
            index for index, value in keyed if value is None
        ]
    else:
        keyed_text = [
            (index, _normalize_label(row[column.index].text))
            for index, row in enumerate(table.rows)
        ]
        keyed_text.sort(key=lambda item: (item[1], item[0]), reverse=descending)
        ordered = [index for index, _ in keyed_text]

    truncated = len(ordered) > parameters.limit
    if truncated:
        _append_warning(warnings, f"仅返回排序后的前 {parameters.limit} 行")
    ordered = ordered[: parameters.limit]
    rows = [ResultRow(cells=_row_cells_display(table.rows[index], columns)) for index in ordered]
    return OperationOutcome(
        status="ok",
        columns=[_result_column(item) for item in columns],
        rows=rows,
        formula=f"sort({column.header}, {parameters.direction}, limit={parameters.limit})",
        inputs={"column": column.header, "direction": parameters.direction},
        units=[item.unit for item in columns if item.unit],
        warnings=warnings,
    )


def _aggregate_value(
    function: str,
    values: list[Decimal],
) -> Decimal | None:
    if not values:
        return None
    if function == "sum":
        with localcontext() as context:
            context.prec = 28
            return sum(values, Decimal(0))
    if function == "avg":
        with localcontext() as context:
            context.prec = 28
            return sum(values, Decimal(0)) / Decimal(len(values))
    if function == "min":
        return min(values)
    return max(values)


def run_aggregate(
    table: StructuredTable,
    parameters: AggregateParameters,
) -> OperationOutcome:
    """Aggregate one numeric column, optionally grouped by another column."""
    column = _resolve_column(table, parameters.column)
    _require_numeric_column(column)
    warnings: list[str] = []
    functions: list[str] = []
    for function in parameters.functions:
        if function not in functions:
            functions.append(function)
    if len(functions) != len(parameters.functions):
        _append_warning(warnings, "重复的聚合函数已去重")

    group_column: TableColumn | None = None
    if parameters.group_by is not None:
        group_column = _resolve_column(table, parameters.group_by)

    grouped: dict[str, list[Decimal]] = {}
    group_order: list[str] = []
    for row in table.rows:
        group_key = row[group_column.index].text.strip() if group_column is not None else ""
        if group_key not in grouped:
            grouped[group_key] = []
            group_order.append(group_key)
        value = _cell_decimal(row[column.index])
        if value is not None:
            grouped[group_key].append(value)

    if not group_order:
        raise TableNotComputableError("no_numeric_data", "表格中没有可聚合的数据行")

    result_columns: list[ResultColumn] = []
    if group_column is not None:
        result_columns.append(_result_column(group_column))
    for function in functions:
        result_columns.append(
            ResultColumn(
                name=f"{function}({column.header})",
                kind=column.kind,
                unit=column.unit,
            )
        )

    rows: list[ResultRow] = []
    empty_groups = 0
    for group_key in group_order:
        values = grouped[group_key]
        if not values:
            empty_groups += 1
        cells: list[str | None] = []
        if group_column is not None:
            cells.append(group_key or None)
        for function in functions:
            value = _aggregate_value(function, values)
            cells.append(_decimal_text(value) if value is not None else None)
        rows.append(ResultRow(cells=cells))
    if empty_groups:
        _append_warning(warnings, f"{empty_groups} 个分组没有数值数据")
    if len(group_order) > MAX_RESULT_ROWS:
        raise TableOperationError(
            "result_too_large",
            f"分组数量 {len(group_order)} 超过结果上限 {MAX_RESULT_ROWS}",
        )

    formula_parts: list[str] = []
    overall_values = [value for value in _numeric_values(table, column) if value is not None]
    for function in functions:
        aggregate = _aggregate_value(function, overall_values)
        if aggregate is not None:
            formula_parts.append(f"{function}({column.header}) = {_decimal_text(aggregate)}")
    formula = "; ".join(formula_parts) if formula_parts else None
    if column.kind == "percent" and "sum" in functions:
        _append_warning(warnings, "百分数列求和通常没有业务意义，请谨慎使用")
    inputs = {
        "column": column.header,
        "functions": ",".join(functions),
        "groups": str(len(group_order)),
        "value_count": str(len(overall_values)),
    }
    if group_column is not None:
        inputs["group_by"] = group_column.header
    units = [column.unit] if column.unit else []
    if group_column is not None and group_column.unit:
        units.insert(0, group_column.unit)
    return OperationOutcome(
        status="ok",
        columns=result_columns,
        rows=rows,
        formula=formula,
        inputs=inputs,
        units=units,
        warnings=warnings,
    )


def _growth_cells(
    metric_name: str,
    base_value: Decimal,
    target_value: Decimal,
    *,
    unit: str | None,
    value_kind: str,
) -> tuple[list[ResultColumn], ResultRow, str, dict[str, str], list[str]]:
    warnings: list[str] = []
    absolute_delta = target_value - base_value
    change_kind = "percentage_point" if value_kind == "percent" else "absolute"
    if base_value == 0:
        raise TableOperationError(
            "division_by_zero",
            f"指标“{metric_name}”的基期值为 0，无法计算相对变化",
        )
    with localcontext() as context:
        context.prec = 28
        relative = (absolute_delta / base_value) * Decimal(100)
    relative_text = _percent_text(relative)
    if value_kind == "percent":
        _append_warning(warnings, "百分数指标的绝对变化以百分点表示")
    absolute_unit = unit
    value_column_kind: ColumnKind = "mixed" if value_kind == "percent" else "number"
    columns = [
        ResultColumn(name="指标", kind="text"),
        ResultColumn(name="基期值", kind=value_column_kind, unit=unit),
        ResultColumn(name="报告期值", kind=value_column_kind, unit=unit),
        ResultColumn(
            name="绝对变化",
            kind="percent" if change_kind == "percentage_point" else "number",
            unit=absolute_unit,
        ),
        ResultColumn(name="相对变化", kind="percent", unit="%"),
    ]
    row = ResultRow(
        cells=[
            metric_name,
            _decimal_text(base_value),
            _decimal_text(target_value),
            _decimal_text(absolute_delta),
            relative_text,
        ]
    )
    formula = (
        f"(({_decimal_text(target_value)} - {_decimal_text(base_value)}) / "
        f"{_decimal_text(base_value)}) × 100% = {relative_text}%"
    )
    inputs = {
        "base_value": _decimal_text(base_value),
        "target_value": _decimal_text(target_value),
        "absolute_delta": _decimal_text(absolute_delta),
        "relative_change_percent": relative_text,
        "change_kind": change_kind,
    }
    return columns, row, formula, inputs, warnings


def run_growth(table: StructuredTable, parameters: GrowthParameters) -> OperationOutcome:
    """Compute absolute delta and relative growth between two values of one metric."""
    if parameters.axis == "rows":
        assert parameters.column is not None
        assert parameters.base_row is not None
        assert parameters.target_row is not None
        column = _resolve_column(table, parameters.column)
        _require_numeric_column(column)
        base_index = _resolve_row(table, parameters.base_row)
        target_index = _resolve_row(table, parameters.target_row)
        base_cell = table.rows[base_index][column.index]
        target_cell = table.rows[target_index][column.index]
        base_value = _cell_decimal(base_cell)
        target_value = _cell_decimal(target_cell)
        base_label = table.row_labels[base_index] or f"第 {base_index} 行"
        target_label = table.row_labels[target_index] or f"第 {target_index} 行"
        metric_name = column.header
        inputs_extra = {"base_row": base_label, "target_row": target_label}
    else:
        assert parameters.row is not None
        assert parameters.base_column is not None
        assert parameters.target_column is not None
        row_index = _resolve_row(table, parameters.row)
        base_column = _resolve_column(table, parameters.base_column)
        target_column = _resolve_column(table, parameters.target_column)
        _require_numeric_column(base_column)
        _require_numeric_column(target_column)
        if base_column.kind != target_column.kind:
            raise TableNotComputableError(
                "incomparable_column",
                "基期列与报告期列的数值类型不一致，无法比较",
            )
        base_value = _cell_decimal(table.rows[row_index][base_column.index])
        target_value = _cell_decimal(table.rows[row_index][target_column.index])
        base_label = base_column.header
        target_label = target_column.header
        metric_name = table.row_labels[row_index] or base_column.header
        column = target_column
        inputs_extra = {"base_column": base_label, "target_column": target_label}

    if base_value is None:
        raise TableOperationError("missing_numeric_value", f"基期“{base_label}”缺少数值")
    if target_value is None:
        raise TableOperationError("missing_numeric_value", f"报告期“{target_label}”缺少数值")
    columns, row, formula, inputs, warnings = _growth_cells(
        metric_name,
        base_value,
        target_value,
        unit=column.unit,
        value_kind=column.kind,
    )
    inputs.update(inputs_extra)
    return OperationOutcome(
        status="ok",
        columns=columns,
        rows=[row],
        formula=formula,
        inputs=inputs,
        units=[column.unit] if column.unit else [],
        warnings=warnings,
    )


def _keyed_rows(
    table: StructuredTable,
    key_column: TableColumn,
    value_column: TableColumn,
) -> tuple[dict[str, Decimal], dict[str, str]]:
    values: dict[str, Decimal] = {}
    labels: dict[str, str] = {}
    duplicates: list[str] = []
    for row in table.rows:
        key = row[key_column.index].text.strip()
        if not key:
            continue
        normalized = _normalize_label(key)
        number = _cell_decimal(row[value_column.index])
        if normalized in values:
            duplicates.append(key)
            continue
        if number is None:
            continue
        values[normalized] = number
        labels[normalized] = key
    if duplicates:
        raise TableNotComputableError(
            "duplicate_key",
            f"键列“{key_column.header}”存在重复值，无法唯一对齐：{', '.join(duplicates[:3])}",
        )
    return values, labels


def run_compare(
    table: StructuredTable,
    compare_table: StructuredTable,
    parameters: CompareParameters,
) -> OperationOutcome:
    """Compare one metric of two tables after aligning their key rows."""
    key_column_a = _resolve_column(table, parameters.key_column)
    value_column_a = _resolve_column(table, parameters.value_column)
    key_column_b = _resolve_column(compare_table, parameters.key_column)
    value_column_b = _resolve_column(compare_table, parameters.value_column)
    _require_numeric_column(value_column_a)
    _require_numeric_column(value_column_b)
    if value_column_a.kind not in {"number", "percent"} or value_column_b.kind not in {
        "number",
        "percent",
    }:
        raise TableNotComputableError("incomparable_column", "参与对比的指标列不是数值列")

    unit_a = value_column_a.unit
    unit_b = value_column_b.unit
    if unit_a is None or unit_b is None:
        raise TableNotComputableError(
            "unit_missing",
            "参与对比的指标缺少单位，无法确认统计口径",
        )
    if unit_a != unit_b:
        raise TableNotComputableError(
            "unit_mismatch",
            f"参与对比的指标单位不一致（{unit_a} 与 {unit_b}），不允许直接计算",
        )

    values_a, labels_a = _keyed_rows(table, key_column_a, value_column_a)
    values_b, _labels_b = _keyed_rows(compare_table, key_column_b, value_column_b)
    if parameters.keys is not None:
        requested = [_normalize_label(key) for key in parameters.keys]
        missing = [key for key in requested if key not in values_a or key not in values_b]
        if missing:
            raise TableOperationError(
                "unknown_key",
                f"以下键在两张表中不都存在：{', '.join(missing[:3])}",
            )
        ordered_keys = requested
    else:
        ordered_keys = [key for key in values_a if key in values_b]
    if not ordered_keys:
        raise TableNotComputableError("no_matching_keys", "两张表格没有可对齐的共同键值")

    warnings: list[str] = []
    zero_base_count = 0
    rows: list[ResultRow] = []
    for key in ordered_keys:
        value_a = values_a[key]
        value_b = values_b[key]
        difference = value_a - value_b
        if value_b == 0:
            zero_base_count += 1
            relative_text = None
        else:
            with localcontext() as context:
                context.prec = 28
                relative = ((value_a - value_b) / value_b) * Decimal(100)
            relative_text = _percent_text(relative)
        rows.append(
            ResultRow(
                cells=[
                    labels_a[key],
                    _decimal_text(value_a),
                    _decimal_text(value_b),
                    _decimal_text(difference),
                    relative_text,
                ]
            )
        )
    if zero_base_count:
        _append_warning(
            warnings,
            f"{zero_base_count} 个键的对比表数值为 0，相对变化不可计算（已留空）",
        )
    function_name = (
        "percentage_point_difference" if value_column_a.kind == "percent" else "difference"
    )
    formula = (
        f"{function_name}({value_column_a.header}) = "
        "A - B；relative_change_percent = (A - B) / B × 100%"
    )
    columns = [
        ResultColumn(name=key_column_a.header, kind="text"),
        ResultColumn(name=f"A·{value_column_a.header}", kind=value_column_a.kind, unit=unit_a),
        ResultColumn(name=f"B·{value_column_b.header}", kind=value_column_b.kind, unit=unit_b),
        ResultColumn(
            name="绝对差异",
            kind="number",
            unit=unit_a,
        ),
        ResultColumn(name="相对差异", kind="percent", unit="%"),
    ]
    inputs = {
        "key_column": key_column_a.header,
        "value_column": value_column_a.header,
        "matched_keys": str(len(rows)),
        "unit": unit_a,
    }
    return OperationOutcome(
        status="ok",
        columns=columns,
        rows=_truncate_rows(rows, warnings),
        formula=formula,
        inputs=inputs,
        units=[unit_a],
        warnings=warnings,
    )


__all__ = [
    "OperationOutcome",
    "run_aggregate",
    "run_compare",
    "run_filter",
    "run_growth",
    "run_select",
    "run_sort",
]
