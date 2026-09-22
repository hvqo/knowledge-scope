import { afterEach, describe, expect, it, vi } from "vitest";
import { VueQueryPlugin, QueryClient } from "@tanstack/vue-query";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia } from "pinia";

import AnalysisView from "./AnalysisView.vue";

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

const datasource = {
  id: "datasource-1",
  display_name: "演示业务库",
  dialect: "postgresql",
  enabled: true,
  default_database: "business",
  default_schema: "public",
  connection_configured: true,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

function successPayload(): Record<string, unknown> {
  return {
    query_id: "query-1",
    datasource_id: "datasource-1",
    execution_status: "succeeded",
    eligibility: {
      decision: "eligible",
      reason_code: "eligible_analytical",
      user_message: null,
      clarification_question: null,
    },
    answer: "华东客户数量最多。",
    columns: [
      { name: "region", data_type: "text", nullable: false, ordinal: 0 },
      { name: "count", data_type: "bigint", nullable: false, ordinal: 1 },
    ],
    rows: [["华东", 12], ["华南", 8]],
    row_count: 2,
    truncated: false,
    truncation_reason: null,
    redacted_sql: "SELECT region, COUNT(*) FROM public.sales GROUP BY region",
    warnings: [],
    error_message: null,
    elapsed_ms: 21,
  };
}

describe("AnalysisView", () => {
  afterEach(() => vi.restoreAllMocks());

  it("loads a datasource, sends a question, renders table/chart and keeps history local", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.includes("/chatbi/data-sources?") && init?.method === undefined) {
        return jsonResponse({ items: [datasource], total: 1, limit: 100, offset: 0 });
      }
      expect(url).toContain("/chatbi/data-sources/datasource-1/ask");
      expect(init?.method).toBe("POST");
      expect(JSON.parse(String(init?.body))).toEqual({ question: "统计各地区客户数" });
      return jsonResponse(successPayload());
    });

    const wrapper = mount(AnalysisView, {
      global: {
        plugins: [
          createPinia(),
          [VueQueryPlugin, { queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }) }],
        ],
      },
    });
    await vi.waitFor(() => expect(wrapper.get("select").element.value).toBe("datasource-1"));

    const textarea = wrapper.get("textarea");
    await textarea.setValue("统计各地区客户数");
    await textarea.trigger("keydown", { key: "Enter", shiftKey: false });
    await flushPromises();
    await vi.waitFor(() => expect(wrapper.text()).toContain("华东客户数量最多"));

    expect(wrapper.text()).toContain("演示业务库");
    expect(wrapper.text()).toContain("region");
    expect(wrapper.text()).toContain("12");
    expect(wrapper.text()).toContain("查看已执行的安全 SQL");
    expect(wrapper.text()).toContain("统计各地区客户数");
    await wrapper.get("button[role='tab']:last-child").trigger("click");
    expect(wrapper.find(".result-chart").exists()).toBe(true);
    expect(fetchMock.mock.calls.filter(([input]) => String(input).includes("/ask"))).toHaveLength(1);
  });

  it("renders a clarification without creating a result table", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      if (String(input).includes("/chatbi/data-sources?") && init?.method === undefined) {
        return jsonResponse({ items: [datasource], total: 1, limit: 100, offset: 0 });
      }
      return jsonResponse({
        ...successPayload(),
        execution_status: "clarify",
        eligibility: {
          decision: "clarify",
          reason_code: "ambiguous_intent",
          user_message: "需要补充分析范围。",
          clarification_question: "你想按哪个时间范围查看？",
        },
        answer: "需要补充分析范围。",
        columns: [],
        rows: [],
        row_count: 0,
        redacted_sql: null,
      });
    });
    const wrapper = mount(AnalysisView, {
      global: {
        plugins: [
          createPinia(),
          [VueQueryPlugin, { queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }) }],
        ],
      },
    });
    await vi.waitFor(() => expect(wrapper.get("select").element.value).toBe("datasource-1"));
    await wrapper.get("textarea").setValue("帮我看看");
    await wrapper.get("textarea").trigger("keydown", { key: "Enter", shiftKey: false });
    await flushPromises();
    await vi.waitFor(() => expect(wrapper.text()).toContain("需要补充分析范围"));
    expect(wrapper.text()).toContain("你想按哪个时间范围查看？");
    expect(wrapper.find("table").exists()).toBe(false);
  });

  it("shows a safe request error and does not submit twice while loading", async () => {
    let resolveAsk: ((response: Response) => void) | undefined;
    const pendingAsk = new Promise<Response>((resolve) => {
      resolveAsk = resolve;
    });
    let askCalls = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      if (String(input).includes("/chatbi/data-sources?") && init?.method === undefined) {
        return jsonResponse({ items: [datasource], total: 1, limit: 100, offset: 0 });
      }
      askCalls += 1;
      return pendingAsk;
    });
    const wrapper = mount(AnalysisView, {
      global: {
        plugins: [
          createPinia(),
          [VueQueryPlugin, { queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }) }],
        ],
      },
    });
    await vi.waitFor(() => expect(wrapper.get("select").element.value).toBe("datasource-1"));
    const textarea = wrapper.get("textarea");
    await textarea.setValue("查找销售数量");
    await textarea.trigger("keydown", { key: "Enter", shiftKey: false });
    await vi.waitFor(() => expect(askCalls).toBe(1));
    await textarea.setValue("再次提交");
    await textarea.trigger("keydown", { key: "Enter", shiftKey: false });
    expect(askCalls).toBe(1);

    resolveAsk?.(new Response("{}", { status: 503, statusText: "Unavailable" }));
    await vi.waitFor(() => expect(wrapper.text()).toContain("服务暂时不可用"));
  });

  it("keeps an empty successful result explicit", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      if (String(input).includes("/chatbi/data-sources?") && init?.method === undefined) {
        return jsonResponse({ items: [datasource], total: 1, limit: 100, offset: 0 });
      }
      return jsonResponse({
        ...successPayload(),
        answer: "没有匹配数据。",
        columns: [{ name: "count", data_type: "bigint", nullable: true, ordinal: 0 }],
        rows: [],
        row_count: 0,
      });
    });
    const wrapper = mount(AnalysisView, {
      global: {
        plugins: [
          createPinia(),
          [VueQueryPlugin, { queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }) }],
        ],
      },
    });
    await vi.waitFor(() => expect(wrapper.get("select").element.value).toBe("datasource-1"));
    await wrapper.get("textarea").setValue("统计一个不存在的区域");
    await wrapper.get("textarea").trigger("keydown", { key: "Enter", shiftKey: false });
    await vi.waitFor(() => expect(wrapper.text()).toContain("没有匹配数据"));
    expect(wrapper.text()).toContain("查询成功，但没有返回匹配数据。");
  });
});
