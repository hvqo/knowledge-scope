import { VueQueryPlugin, QueryClient } from "@tanstack/vue-query";
import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";

import ReportListView from "./ReportListView.vue";

const { routerPush } = vi.hoisted(() => ({ routerPush: vi.fn() }));

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: routerPush }),
}));

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("ReportListView", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    routerPush.mockReset();
  });

  it("loads reports and creates a report with selected sources", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.includes("/reports?") && init?.method === undefined) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      if (url.includes("/knowledge-bases?") && init?.method === undefined) {
        return jsonResponse({
          items: [{
            id: "kb-1",
            name: "质量资料",
            description: null,
            created_at: "2026-01-01T00:00:00Z",
            updated_at: "2026-01-01T00:00:00Z",
          }],
          total: 1,
          limit: 100,
          offset: 0,
        });
      }
      if (url.includes("/chatbi/data-sources?") && init?.method === undefined) {
        return jsonResponse({
          items: [{
            id: "ds-1",
            display_name: "演示业务库",
            dialect: "postgresql",
            enabled: true,
            default_database: "business",
            default_schema: "public",
            connection_configured: true,
            created_at: "2026-01-01T00:00:00Z",
            updated_at: "2026-01-01T00:00:00Z",
          }],
          total: 1,
          limit: 100,
          offset: 0,
        });
      }
      expect(url).toContain("/reports");
      expect(init?.method).toBe("POST");
      expect(JSON.parse(String(init?.body))).toEqual({
        title: "季度经营报告",
        knowledge_base_id: "kb-1",
        datasource_id: "ds-1",
      });
      return jsonResponse({
        id: "report-1",
        title: "季度经营报告",
        knowledge_base_id: "kb-1",
        datasource_id: "ds-1",
        sections: [],
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      }, 201);
    });

    const wrapper = mount(ReportListView, {
      global: {
        plugins: [[VueQueryPlugin, {
          queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }),
        }]],
      },
    });

    await vi.waitFor(() => expect(wrapper.find("select").exists()).toBe(true));
    const selects = wrapper.findAll("select");
    await wrapper.get("input").setValue("季度经营报告");
    await selects[0].setValue("kb-1");
    await selects[1].setValue("ds-1");
    await wrapper.get("form").trigger("submit");
    await flushPromises();

    expect(fetchMock.mock.calls.some(([input, init]) =>
      String(input).endsWith("/reports") && init?.method === "POST",
    )).toBe(true);
  });

  it("shows an empty state without inventing report data", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes("/reports?")) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      if (url.includes("/knowledge-bases?")) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
    });

    const wrapper = mount(ReportListView, {
      global: {
        plugins: [[VueQueryPlugin, {
          queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }),
        }]],
      },
    });

    await vi.waitFor(() => expect(wrapper.text()).toContain("还没有报告"));
    expect(wrapper.text()).not.toContain("示例报告");
  });
});
