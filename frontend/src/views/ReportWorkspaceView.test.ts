import { VueQueryPlugin, QueryClient } from "@tanstack/vue-query";
import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";

import ReportAIPanel from "../components/report/ReportAIPanel.vue";
import ReportWorkspaceView from "./ReportWorkspaceView.vue";

const { routerPush, routeState } = vi.hoisted(() => ({
  routerPush: vi.fn(),
  routeState: {
    current: undefined as { id: string } | undefined,
    updateGuard: undefined as ((to: unknown, from: unknown) => unknown) | undefined,
    leaveGuard: undefined as (() => unknown) | undefined,
  },
}));

vi.mock("vue-router", async () => {
  const { reactive } = await import("vue");
  const params = reactive({ id: "report-1" });
  routeState.current = params;
  return {
    useRoute: () => ({ params }),
    useRouter: () => ({ push: routerPush }),
    onBeforeRouteUpdate: (guard: (to: unknown, from: unknown) => unknown) => {
      routeState.updateGuard = guard;
    },
    onBeforeRouteLeave: (guard: () => unknown) => {
      routeState.leaveGuard = guard;
    },
  };
});

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const baseReport = {
  id: "report-1",
  title: "季度经营报告",
  knowledge_base_id: "kb-1",
  datasource_id: "ds-1",
  sections: [{
    id: "section-1",
    report_id: "report-1",
    title: "概览",
    content: "初始内容",
    position: 0,
    sources: [{
      id: "source-1",
      section_id: "section-1",
      kind: "rag_evidence",
      title: "质量控制",
      knowledge_base_id: "kb-1",
      datasource_id: null,
      document_id: "document-1",
      document_title: "质量手册.pdf",
      evidence_id: "evidence-1",
      chunk_id: null,
      page_start: 2,
      page_end: 2,
      evidence_type: "text",
      snippet: "质量控制证据",
      section_path: ["质量控制"],
      source_block_ids: ["block-1"],
      question: null,
      answer: null,
      columns: [],
      rows: [],
      row_count: null,
      truncated: null,
      truncation_reason: null,
      redacted_sql: null,
      chart_spec: null,
      created_at: "2026-01-01T00:00:00Z",
    }],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  }, {
    id: "section-2",
    report_id: "report-1",
    title: "结论",
    content: "第二章节内容",
    position: 1,
    sources: [],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  }],
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

describe("ReportWorkspaceView", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    routerPush.mockReset();
    if (routeState.current) {
      routeState.current.id = "report-1";
    }
    routeState.updateGuard = undefined;
    routeState.leaveGuard = undefined;
  });

  it("loads a persisted report, saves the active section, and previews sources", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.includes("/reports/report-1") && init?.method === undefined) {
        return jsonResponse(baseReport);
      }
      if (url.includes("/knowledge-bases?") && init?.method === undefined) {
        return jsonResponse({ items: [{
          id: "kb-1",
          name: "质量资料",
          description: null,
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
        }], total: 1, limit: 100, offset: 0 });
      }
      if (url.includes("/chatbi/data-sources?") && init?.method === undefined) {
        return jsonResponse({ items: [{
          id: "ds-1",
          display_name: "演示业务库",
          dialect: "postgresql",
          enabled: true,
          default_database: "business",
          default_schema: "public",
          connection_configured: true,
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
        }], total: 1, limit: 100, offset: 0 });
      }
      if (url.includes("/documents?") && init?.method === undefined) {
        return jsonResponse({ items: [{
          id: "document-1",
          knowledge_base_id: "kb-1",
          original_filename: "质量手册.pdf",
          media_type: "application/pdf",
          size_bytes: 100,
          sha256: "hash",
          status: "uploaded",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
        }], total: 1, limit: 100, offset: 0 });
      }
      expect(init?.method).toBe("PATCH");
      if (url.endsWith("/sections/section-1")) {
        const body = JSON.parse(String(init?.body));
        return jsonResponse({ ...baseReport.sections[0], ...body });
      }
      expect(url).toBe("/api/v1/reports/report-1");
      return jsonResponse({ ...baseReport, title: "已保存报告" });
    });

    const wrapper = mount(ReportWorkspaceView, {
      global: {
        plugins: [[VueQueryPlugin, {
          queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }),
        }]],
      },
    });

    await vi.waitFor(() =>
      expect((wrapper.get(".report-editor__textarea").element as HTMLTextAreaElement).value).toBe("初始内容"),
    );
    await wrapper.get(".report-editor__textarea").setValue("已补充结论");
    await wrapper.get(".report-editor__save").trigger("click");
    await vi.waitFor(() => expect(wrapper.text()).toContain("已保存"));

    const patchCalls = fetchMock.mock.calls.filter(([, init]) => init?.method === "PATCH");
    expect(patchCalls).toHaveLength(2);
    expect(JSON.parse(String(patchCalls[1]?.[1]?.body))).toMatchObject({
      title: "概览",
      content: "已补充结论",
    });

    await wrapper.get(".report-editor__preview").trigger("click");
    await flushPromises();
    expect(wrapper.find(".report-preview").exists()).toBe(true);
    expect(wrapper.text()).toContain("质量手册.pdf");
  });

  it("saves dirty edits before switching sections", async () => {
    const persistedReport = JSON.parse(JSON.stringify(baseReport)) as typeof baseReport;
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.includes("/reports/report-1") && init?.method === undefined) {
        return jsonResponse(persistedReport);
      }
      if (url.includes("/knowledge-bases?") && init?.method === undefined) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      if (url.includes("/chatbi/data-sources?") && init?.method === undefined) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      if (url.includes("/documents?") && init?.method === undefined) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      expect(init?.method).toBe("PATCH");
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      if (url.endsWith("/sections/section-1")) {
        Object.assign(persistedReport.sections[0], body);
        return jsonResponse(persistedReport.sections[0]);
      }
      expect(url).toBe("/api/v1/reports/report-1");
      Object.assign(persistedReport, body);
      return jsonResponse(persistedReport);
    });

    const wrapper = mount(ReportWorkspaceView, {
      global: {
        plugins: [[VueQueryPlugin, {
          queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }),
        }]],
      },
    });

    await vi.waitFor(() =>
      expect((wrapper.get(".report-editor__textarea").element as HTMLTextAreaElement).value).toBe("初始内容"),
    );
    await wrapper.get(".report-editor__textarea").setValue("切换前已保存");
    await wrapper.findAll(".report-outline__section")[1]?.trigger("click");

    await vi.waitFor(() =>
      expect((wrapper.get(".report-editor__textarea").element as HTMLTextAreaElement).value).toBe("第二章节内容"),
    );
    const patchCalls = fetchMock.mock.calls.filter(([, init]) => init?.method === "PATCH");
    expect(patchCalls).toHaveLength(2);
    expect(JSON.parse(String(patchCalls[1]?.[1]?.body))).toMatchObject({
      title: "概览",
      content: "切换前已保存",
    });
  });

  it("saves a dirty draft before changing reports", async () => {
    const persistedReport = JSON.parse(JSON.stringify(baseReport)) as typeof baseReport;
    const reportB = JSON.parse(JSON.stringify(baseReport)) as typeof baseReport;
    reportB.id = "report-2";
    reportB.sections[0].report_id = "report-2";
    reportB.sections[0].content = "报告 B 内容";
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (init?.method === undefined && url.includes("/reports/report-1")) {
        return jsonResponse(persistedReport);
      }
      if (init?.method === undefined && url.includes("/reports/report-2")) {
        return jsonResponse(reportB);
      }
      if (url.includes("/knowledge-bases?") || url.includes("/chatbi/data-sources?") || url.includes("/documents?")) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      expect(init?.method).toBe("PATCH");
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      if (url.endsWith("/sections/section-1")) {
        Object.assign(persistedReport.sections[0], body);
        return jsonResponse(persistedReport.sections[0]);
      }
      expect(url).toBe("/api/v1/reports/report-1");
      Object.assign(persistedReport, body);
      return jsonResponse(persistedReport);
    });

    const wrapper = mount(ReportWorkspaceView, {
      global: {
        plugins: [[VueQueryPlugin, {
          queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }),
        }]],
      },
    });

    await vi.waitFor(() =>
      expect((wrapper.get(".report-editor__textarea").element as HTMLTextAreaElement).value).toBe("初始内容"),
    );
    await wrapper.get(".report-editor__textarea").setValue("切换报告前已保存");

    const guardResult = await routeState.updateGuard?.(
      { params: { id: "report-2" } },
      { params: { id: "report-1" } },
    );
    expect(guardResult).toBe(true);
    const patchCalls = fetchMock.mock.calls.filter(([, init]) => init?.method === "PATCH");
    expect(JSON.parse(String(patchCalls[1]?.[1]?.body))).toMatchObject({
      title: "概览",
      content: "切换报告前已保存",
    });

    routeState.current!.id = "report-2";
    await vi.waitFor(() =>
      expect((wrapper.get(".report-editor__textarea").element as HTMLTextAreaElement).value).toBe("报告 B 内容"),
    );
  });

  it("blocks a report change when saving the dirty draft fails", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (init?.method === undefined && url.includes("/reports/report-1")) {
        return jsonResponse(baseReport);
      }
      if (url.includes("/knowledge-bases?") || url.includes("/chatbi/data-sources?") || url.includes("/documents?")) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      expect(url).toBe("/api/v1/reports/report-1");
      expect(init?.method).toBe("PATCH");
      return jsonResponse({ detail: "保存失败" }, 500);
    });

    const wrapper = mount(ReportWorkspaceView, {
      global: {
        plugins: [[VueQueryPlugin, {
          queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }),
        }]],
      },
    });

    await vi.waitFor(() =>
      expect((wrapper.get(".report-editor__textarea").element as HTMLTextAreaElement).value).toBe("初始内容"),
    );
    await wrapper.get(".report-editor__textarea").setValue("不能丢失的草稿");

    const guardResult = await routeState.updateGuard?.(
      { params: { id: "report-2" } },
      { params: { id: "report-1" } },
    );
    expect(guardResult).toBe(false);
    expect(routeState.current!.id).toBe("report-1");
    expect((wrapper.get(".report-editor__textarea").element as HTMLTextAreaElement).value).toBe("不能丢失的草稿");
    expect(wrapper.text()).toContain("当前章节保存失败");
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes("/reports/report-2"))).toBe(false);
  });

  it("clears source results when the workspace route changes", async () => {
    const reportB = JSON.parse(JSON.stringify(baseReport)) as typeof baseReport;
    reportB.id = "report-2";
    let resolveLateResult: ((response: Response) => void) | null = null;
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.includes("/reports/report-1") && init?.method === undefined) {
        return jsonResponse(baseReport);
      }
      if (url.includes("/reports/report-2") && init?.method === undefined) {
        return jsonResponse(reportB);
      }
      if (url.includes("/knowledge-bases?") && init?.method === undefined) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      if (url.includes("/chatbi/data-sources?") && init?.method === undefined) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      if (url.includes("/documents?") && init?.method === undefined) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      expect(url).toContain("/chatbi/data-sources/ds-1/ask");
      const request = JSON.parse(String(init?.body)) as { question: string };
      if (request.question === "延迟问题") {
        return new Promise<Response>((resolve) => {
          resolveLateResult = resolve;
        });
      }
      return jsonResponse({
        query_id: "query-1",
        datasource_id: "ds-1",
        execution_status: "succeeded",
        eligibility: null,
        answer: "报告 A 的分析结果",
        columns: [{ name: "数量", data_type: "bigint", nullable: false, ordinal: 0 }],
        rows: [[3]],
        row_count: 1,
        truncated: false,
        truncation_reason: null,
        redacted_sql: null,
        warnings: [],
        error_message: null,
        elapsed_ms: 4,
      });
    });

    const wrapper = mount(ReportWorkspaceView, {
      global: {
        plugins: [[VueQueryPlugin, {
          queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }),
        }]],
      },
    });
    await vi.waitFor(() => expect(wrapper.get(".report-editor__textarea")).toBeTruthy());
    await wrapper.findAll("[role='tab']")[1]?.trigger("click");
    await wrapper.get(".report-source-panel textarea").setValue("统计数量");
    await wrapper.get(".report-source-panel form").trigger("submit");
    await vi.waitFor(() => expect(wrapper.text()).toContain("报告 A 的分析结果"));

    expect(routeState.current).toBeDefined();
    routeState.current!.id = "report-2";
    await vi.waitFor(() => expect(wrapper.text()).not.toContain("报告 A 的分析结果"));
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes("/reports/report-2"))).toBe(true);

    routeState.current!.id = "report-1";
    await vi.waitFor(() => expect(wrapper.text()).toContain("概览"));
    await wrapper.findAll("[role='tab']")[1]?.trigger("click");
    await wrapper.get(".report-source-panel textarea").setValue("延迟问题");
    await wrapper.get(".report-source-panel form").trigger("submit");
    await vi.waitFor(() => expect(resolveLateResult).not.toBeNull());

    routeState.current!.id = "report-2";
    resolveLateResult!(jsonResponse({
      query_id: "late-query",
      datasource_id: "ds-1",
      execution_status: "succeeded",
      eligibility: null,
      answer: "过期结果不应显示",
      columns: [],
      rows: [],
      row_count: 0,
      truncated: false,
      truncation_reason: null,
      redacted_sql: null,
      warnings: [],
      error_message: null,
      elapsed_ms: 4,
    }));
    await vi.waitFor(() => expect(wrapper.text()).not.toContain("过期结果不应显示"));
  });

  it("previews an AI section draft and keeps it local until the user saves", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (init?.method === undefined && url.includes("/reports/report-1")) {
        return jsonResponse(baseReport);
      }
      if (
        url.includes("/knowledge-bases?") ||
        url.includes("/chatbi/data-sources?") ||
        url.includes("/documents?")
      ) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      if (url.endsWith("/sections/section-1/ai/generate")) {
        return jsonResponse({
          operation: "generate",
          content: "AI 草稿 [S-A1B2C3D4]",
          citations: [{
            marker: "S-A1B2C3D4",
            source_id: "source-1",
            title: "质量控制",
            page_start: 2,
            page_end: 2,
          }],
        });
      }
      throw new Error(`unexpected request: ${url}`);
    });

    const wrapper = mount(ReportWorkspaceView, {
      global: {
        plugins: [[VueQueryPlugin, {
          queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }),
        }]],
      },
    });

    await vi.waitFor(() => expect(wrapper.get(".report-editor__textarea")).toBeTruthy());
    await wrapper.findAll(".report-ai-panel__actions button")[1]!.trigger("click");
    await vi.waitFor(() => expect(wrapper.text()).toContain("章节预览"));
    await wrapper.get(".report-ai-panel__accept").trigger("click");
    expect((wrapper.get(".report-editor__textarea").element as HTMLTextAreaElement).value).toBe(
      "AI 草稿 [S-A1B2C3D4]",
    );
    expect(fetchMock.mock.calls.some(([, request]) => request?.method === "PATCH")).toBe(false);
  });

  it("ignores a late AI response after switching reports", async () => {
    const reportB = JSON.parse(JSON.stringify(baseReport)) as typeof baseReport;
    reportB.id = "report-2";
    reportB.sections[0].report_id = "report-2";
    reportB.sections[0].content = "报告 B 内容";
    let resolveAI: ((response: Response) => void) | null = null;
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (init?.method === undefined && url.includes("/reports/report-1")) {
        return jsonResponse(baseReport);
      }
      if (init?.method === undefined && url.includes("/reports/report-2")) {
        return jsonResponse(reportB);
      }
      if (url.includes("/knowledge-bases?") || url.includes("/chatbi/data-sources?") || url.includes("/documents?")) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      if (url.endsWith("/sections/section-1/ai/generate")) {
        return new Promise<Response>((resolve) => {
          resolveAI = resolve;
        });
      }
      throw new Error(`unexpected request: ${url}`);
    });

    const wrapper = mount(ReportWorkspaceView, {
      global: {
        plugins: [[VueQueryPlugin, {
          queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }),
        }]],
      },
    });

    await vi.waitFor(() => expect(wrapper.get(".report-editor__textarea")).toBeTruthy());
    await wrapper.findAll(".report-ai-panel__actions button")[1]!.trigger("click");
    await vi.waitFor(() => expect(resolveAI).not.toBeNull());

    routeState.current!.id = "report-2";
    await vi.waitFor(() =>
      expect((wrapper.get(".report-editor__textarea").element as HTMLTextAreaElement).value).toBe("报告 B 内容"),
    );
    resolveAI!(jsonResponse({
      operation: "generate",
      content: "报告 A 的过期草稿",
      citations: [],
    }));
    await flushPromises();

    expect(wrapper.find(".report-ai-panel__preview").exists()).toBe(false);
    expect(wrapper.text()).not.toContain("报告 A 的过期草稿");
    expect(fetchMock.mock.calls.some(([, request]) => request?.method === "PATCH")).toBe(false);
  });

  it("keeps the newest AI request when an older response arrives late", async () => {
    let aiCallCount = 0;
    let resolveFirst: ((response: Response) => void) | null = null;
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (init?.method === undefined && url.includes("/reports/report-1")) {
        return jsonResponse(baseReport);
      }
      if (url.includes("/knowledge-bases?") || url.includes("/chatbi/data-sources?") || url.includes("/documents?")) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      if (url.endsWith("/sections/section-1/ai/generate")) {
        aiCallCount += 1;
        if (aiCallCount === 1) {
          return new Promise<Response>((resolve) => {
            resolveFirst = resolve;
          });
        }
        return jsonResponse({ operation: "generate", content: "第二次结果", citations: [] });
      }
      throw new Error(`unexpected request: ${url}`);
    });

    const wrapper = mount(ReportWorkspaceView, {
      global: {
        plugins: [[VueQueryPlugin, {
          queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }),
        }]],
      },
    });

    await vi.waitFor(() => expect(wrapper.get(".report-editor__textarea")).toBeTruthy());
    const generateButton = wrapper.findAll(".report-ai-panel__actions button")[1]!;
    await generateButton.trigger("click");
    await vi.waitFor(() => expect(aiCallCount).toBe(1));
    await wrapper.findComponent(ReportAIPanel).vm.$emit("generateSection", "第二次");
    await vi.waitFor(() => expect(aiCallCount).toBe(2));
    await vi.waitFor(() =>
      expect((wrapper.get(".report-ai-panel__draft").element as HTMLTextAreaElement).value).toBe("第二次结果"),
    );

    resolveFirst!(jsonResponse({ operation: "generate", content: "第一次结果", citations: [] }));
    await flushPromises();
    expect((wrapper.get(".report-ai-panel__draft").element as HTMLTextAreaElement).value).toBe("第二次结果");
    expect((wrapper.get(".report-ai-panel__draft").element as HTMLTextAreaElement).value).not.toBe("第一次结果");
    expect(fetchMock.mock.calls.filter(([input]) => String(input).endsWith("/ai/generate"))).toHaveLength(2);
  });

  it("previews and accepts or discards each section edit operation", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (init?.method === undefined && url.includes("/reports/report-1")) {
        return jsonResponse(baseReport);
      }
      if (url.includes("/knowledge-bases?") || url.includes("/chatbi/data-sources?") || url.includes("/documents?")) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      if (init?.method === "PATCH") {
        return jsonResponse(baseReport);
      }
      if (url.endsWith("/sections/section-1/ai/edit")) {
        const body = JSON.parse(String(init?.body)) as { operation: string };
        return jsonResponse({
          operation: body.operation,
          content: `${body.operation} 预览`,
          citations: [],
        });
      }
      throw new Error(`unexpected request: ${url}`);
    });

    const wrapper = mount(ReportWorkspaceView, {
      global: {
        plugins: [[VueQueryPlugin, {
          queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }),
        }]],
      },
    });

    await vi.waitFor(() => expect(wrapper.get(".report-editor__textarea")).toBeTruthy());
    for (const [index, operation] of [[2, "rewrite"], [3, "expand"], [4, "summarize"]] as const) {
      await wrapper.findAll(".report-ai-panel__actions button")[index]!.trigger("click");
      await vi.waitFor(() =>
        expect((wrapper.get(".report-ai-panel__draft").element as HTMLTextAreaElement).value).toBe(
          `${operation} 预览`,
        ),
      );
      if (operation === "expand") {
        await wrapper.get(".report-ai-panel__accept").trigger("click");
        expect((wrapper.get(".report-editor__textarea").element as HTMLTextAreaElement).value).toBe("expand 预览");
      } else {
        await wrapper.get(".report-ai-panel__discard").trigger("click");
      }
      expect(wrapper.find(".report-ai-panel__preview").exists()).toBe(false);
    }
    expect(fetchMock.mock.calls.filter(([, request]) => request?.method === "PATCH")).toHaveLength(2);
  });

  it("triggers DOCX/PDF export and surfaces export failures", async () => {
    const createObjectURL = vi.fn(() => "blob:report");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (init?.method === undefined && url.includes("/reports/report-1") && !url.includes("/export/")) {
        return jsonResponse(baseReport);
      }
      if (url.includes("/knowledge-bases?") || url.includes("/chatbi/data-sources?") || url.includes("/documents?")) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      if (url.endsWith("/export/docx")) {
        return new Response(new Uint8Array([1, 2, 3]), {
          status: 200,
          headers: { "Content-Disposition": "attachment; filename*=UTF-8''%E5%AD%A3%E5%BA%A6%E6%8A%A5%E5%91%8A.docx" },
        });
      }
      if (url.endsWith("/export/pdf")) {
        return jsonResponse({ detail: "导出失败" }, 503);
      }
      throw new Error(`unexpected request: ${url}`);
    });

    const wrapper = mount(ReportWorkspaceView, {
      global: {
        plugins: [[VueQueryPlugin, {
          queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }),
        }]],
      },
    });

    await vi.waitFor(() => expect(wrapper.get(".report-editor__textarea")).toBeTruthy());
    await wrapper.findAll(".report-workspace-page__export")[0]!.trigger("click");
    await vi.waitFor(() => expect(createObjectURL).toHaveBeenCalled());
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:report");
    await vi.waitFor(() =>
      expect(wrapper.findAll(".report-workspace-page__export")[0]!.attributes("disabled")).toBeUndefined(),
    );
    await wrapper.findAll(".report-workspace-page__export")[1]!.trigger("click");
    await vi.waitFor(() => expect(wrapper.text()).toContain("服务暂时不可用"));
    expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith("/export/pdf"))).toBe(true);
  });
});
