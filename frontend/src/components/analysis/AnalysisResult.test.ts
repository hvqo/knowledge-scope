import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";

import type { ChatBIAnalysisResult } from "../../api/types";
import AnalysisResult from "./AnalysisResult.vue";

function decisionResult(
  executionStatus: "clarify" | "refuse" | "eligibility_unavailable",
): ChatBIAnalysisResult {
  const isUnavailable = executionStatus === "eligibility_unavailable";
  return {
    query_id: "query-1",
    datasource_id: "datasource-1",
    execution_status: executionStatus,
    eligibility: {
      decision: isUnavailable ? "unavailable" : executionStatus,
      reason_code: isUnavailable
        ? "eligibility_check_unavailable"
        : executionStatus === "clarify"
          ? "ambiguous_intent"
          : "unsupported_capability",
      user_message: isUnavailable ? null : "请补充分析范围。",
      clarification_question: isUnavailable ? null : "你想看哪个时间段？",
    },
    answer: isUnavailable ? null : "请补充分析范围。",
    columns: [],
    rows: [],
    row_count: 0,
    truncated: false,
    truncation_reason: null,
    redacted_sql: null,
    warnings: [],
    error_message: isUnavailable ? "暂时无法判断该问题。" : null,
    elapsed_ms: 3,
  };
}

describe("AnalysisResult eligibility states", () => {
  it("keeps clarify, refuse and unavailable terminal states free of SQL/results", () => {
    for (const state of ["clarify", "refuse", "eligibility_unavailable"] as const) {
      const wrapper = mount(AnalysisResult, {
        props: { result: decisionResult(state), datasourceName: "演示业务库" },
      });
      expect(wrapper.find("table").exists()).toBe(false);
      expect(wrapper.find("details").exists()).toBe(false);
      expect(wrapper.text()).not.toContain("SQL");
    }
  });
});
