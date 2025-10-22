import { describe, expect, it } from "vitest";

import type { ChatBIAnalysisResult } from "../../api/types";
import { inferAnalysisChart } from "./chartInference";

function result(
  columns: ChatBIAnalysisResult["columns"],
  rows: ChatBIAnalysisResult["rows"],
): ChatBIAnalysisResult {
  return {
    query_id: "query-1",
    datasource_id: "datasource-1",
    execution_status: "succeeded",
    eligibility: null,
    answer: "完成",
    columns,
    rows,
    row_count: rows.length,
    truncated: false,
    truncation_reason: null,
    redacted_sql: null,
    warnings: [],
    error_message: null,
    elapsed_ms: 2,
  };
}

describe("inferAnalysisChart", () => {
  it("chooses a category bar chart for a small grouped result", () => {
    const spec = inferAnalysisChart(
      result(
        [
          { name: "region", data_type: "text", nullable: false, ordinal: 0 },
          { name: "count", data_type: "bigint", nullable: false, ordinal: 1 },
        ],
        [["华东", 12], ["华南", 8], ["华北", 5], ["西南", 3], ["东北", 2], ["西北", 1], ["海外", 1]],
      ),
    );
    expect(spec?.kind).toBe("bar");
    expect(spec?.categoryColumn).toBe("region");
    expect(spec?.valueColumn).toBe("count");
  });

  it("chooses a line chart for temporal data and rejects unsupported shapes", () => {
    const spec = inferAnalysisChart(
      result(
        [
          { name: "sold_on", data_type: "date", nullable: false, ordinal: 0 },
          { name: "amount", data_type: "numeric", nullable: false, ordinal: 1 },
        ],
        [["2026-01-01", 10], ["2026-01-02", 12]],
      ),
    );
    expect(spec?.kind).toBe("line");
    expect(
      inferAnalysisChart(
        result(
          [{ name: "amount", data_type: "numeric", nullable: false, ordinal: 0 }],
          [[1], [2]],
        ),
      ),
    ).toBeNull();
  });

  it("uses a pie chart only for a small non-negative distribution", () => {
    const spec = inferAnalysisChart(
      result(
        [
          { name: "category", data_type: "text", nullable: false, ordinal: 0 },
          { name: "total", data_type: "integer", nullable: false, ordinal: 1 },
        ],
        [["A", 3], ["B", 2], ["C", 1]],
      ),
    );
    expect(spec?.kind).toBe("pie");
  });
});
