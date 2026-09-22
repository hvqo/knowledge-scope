import type { ChatBIAnalysisResult, ChatBIResultScalar } from "../../api/types";

export type AnalysisChartKind = "bar" | "line" | "pie";

export interface AnalysisChartPoint {
  label: string;
  value: number;
}

export interface AnalysisChartSpec {
  kind: AnalysisChartKind;
  categoryColumn: string;
  valueColumn: string;
  points: AnalysisChartPoint[];
}

const NUMERIC_TYPES = /int|numeric|decimal|number|float|double|real|money|serial/i;
const TEMPORAL_TYPES = /date|time|timestamp/i;

function isNumber(value: ChatBIResultScalar): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function displayLabel(value: ChatBIResultScalar): string | null {
  if (typeof value === "string" || typeof value === "number") {
    return String(value);
  }
  return null;
}

function looksTemporal(columnName: string, dataType: string): boolean {
  return TEMPORAL_TYPES.test(columnName) || TEMPORAL_TYPES.test(dataType) || /日期|时间/.test(columnName);
}

/** Select a deliberately conservative chart only when two-dimensional data supports it. */
export function inferAnalysisChart(result: ChatBIAnalysisResult): AnalysisChartSpec | null {
  if (result.rows.length === 0 || result.rows.length > 24 || result.columns.length < 2) {
    return null;
  }
  const categoryIndex = result.columns.findIndex((column) => !NUMERIC_TYPES.test(column.data_type));
  if (categoryIndex < 0) {
    return null;
  }
  const valueIndex = result.columns.findIndex(
    (column, index) => index !== categoryIndex && NUMERIC_TYPES.test(column.data_type),
  );
  if (valueIndex < 0) {
    return null;
  }
  const points: AnalysisChartPoint[] = [];
  for (const row of result.rows) {
    const label = displayLabel(row[categoryIndex]);
    const value = row[valueIndex];
    if (label === null || !isNumber(value)) {
      return null;
    }
    points.push({ label, value });
  }
  if (points.length === 0 || new Set(points.map((point) => point.label)).size !== points.length) {
    return null;
  }
  const category = result.columns[categoryIndex];
  const value = result.columns[valueIndex];
  if (looksTemporal(category.name, category.data_type)) {
    return {
      kind: "line",
      categoryColumn: category.name,
      valueColumn: value.name,
      points,
    };
  }
  if (points.length <= 6 && points.every((point) => point.value >= 0)) {
    return {
      kind: "pie",
      categoryColumn: category.name,
      valueColumn: value.name,
      points,
    };
  }
  return {
    kind: "bar",
    categoryColumn: category.name,
    valueColumn: value.name,
    points,
  };
}
