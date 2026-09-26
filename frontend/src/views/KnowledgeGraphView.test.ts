import { VueQueryPlugin, QueryClient } from "@tanstack/vue-query";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia } from "pinia";
import { createMemoryHistory, createRouter } from "vue-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { GraphOverview } from "../api/types";
import KnowledgeGraphView from "./KnowledgeGraphView.vue";

const { routerReplace } = vi.hoisted(() => ({ routerReplace: vi.fn() }));

vi.mock("vue-router", async () => {
  const actual = await vi.importActual<Record<string, unknown>>("vue-router");
  return {
    ...actual,
    useRouter: () => ({ replace: routerReplace }),
  };
});

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function knowledgeBasesPayload() {
  return {
    items: [
      {
        id: "kb-1",
        name: "质量资料",
        description: null,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      },
    ],
    total: 1,
    limit: 100,
    offset: 0,
  };
}

function overviewPayload(): GraphOverview {
  return {
    knowledge_base_id: "kb-1",
    entity_count: 2,
    relation_count: 1,
    entity_types: [{ entity_type: "概念", entity_count: 2 }],
    truncated: false,
    nodes: [
      {
        entity_id: "entity-a",
        canonical_name: "出水水质",
        entity_type: "概念",
        aliases: ["出水品质"],
        document_id: "doc-1",
        canonical_entity_ids: [],
        evidence_count: 2,
        degree: 1,
      },
      {
        entity_id: "entity-b",
        canonical_name: "浊度",
        entity_type: "概念",
        aliases: [],
        document_id: "doc-1",
        canonical_entity_ids: [],
        evidence_count: 1,
        degree: 1,
      },
    ],
    edges: [
      {
        relation_id: "relation-a",
        source_entity_id: "entity-a",
        target_entity_id: "entity-b",
        relation_type: "包含",
        kind: "relation",
        document_id: "doc-1",
      },
    ],
  };
}

function detailPayload() {
  return {
    entity: {
      entity_id: "entity-a",
      canonical_name: "出水水质",
      entity_type: "概念",
      aliases: ["出水品质"],
      document_id: "doc-1",
      canonical_entity_ids: ["canonical-1"],
      evidence_count: 2,
      degree: 1,
    },
    canonical_entities: [
      { canonical_entity_id: "canonical-1", canonical_name: "出水水质", entity_type: "概念" },
    ],
    evidence: [
      {
        chunk_id: "chunk-1",
        document_id: "doc-1",
        page_start: 3,
        page_end: 4,
        section_path: ["第三章"],
      },
    ],
    neighbors: [
      {
        entity_id: "entity-b",
        canonical_name: "浊度",
        entity_type: "概念",
        document_id: "doc-1",
        direction: "forward",
        relation_id: "relation-a",
        relation_type: "包含",
        page_start: 3,
        page_end: 4,
        shared_canonical_name: null,
      },
    ],
  };
}

function documentsPayload() {
  return {
    items: [
      {
        id: "doc-1",
        knowledge_base_id: "kb-1",
        original_filename: "运行手册.pdf",
        media_type: "application/pdf",
        size_bytes: 1024,
        sha256: "a".repeat(64),
        status: "uploaded",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      },
    ],
    total: 1,
    limit: 100,
    offset: 0,
  };
}

async function mountView() {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", name: "knowledge-graph", component: { template: "<div />" } },
      { path: "/knowledge-bases", name: "knowledge-bases", component: { template: "<div />" } },
    ],
  });
  await router.push("/");
  await router.isReady();
  return mount(KnowledgeGraphView, {
    global: {
      plugins: [
        createPinia(),
        [
          VueQueryPlugin,
          { queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }) },
        ],
        router,
      ],
    },
  });
}

describe("KnowledgeGraphView", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    routerReplace.mockReset();
  });

  it("renders the graph and resolves one entity's evidence after a node click", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes("/v1/knowledge-bases?")) {
        return jsonResponse(knowledgeBasesPayload());
      }
      if (url.includes("/v1/knowledge-bases/kb-1/documents?")) {
        return jsonResponse(documentsPayload());
      }
      if (url.includes("/graph/entities/entity-a")) {
        return jsonResponse(detailPayload());
      }
      if (url.includes("/v1/knowledge-bases/kb-1/graph?")) {
        return jsonResponse(overviewPayload());
      }
      return jsonResponse({ detail: "unexpected request" }, 500);
    });

    const wrapper = await mountView();
    await vi.waitFor(() => expect(wrapper.findAll(".graph-node")).toHaveLength(2));
    await flushPromises();

    expect(wrapper.text()).toContain("2 个实体 · 1 条关系");
    expect(wrapper.text()).toContain("选择一个节点");
    expect(wrapper.findAll(".edge-label")).toHaveLength(0);

    await wrapper.findAll(".graph-node")[0].trigger("click");
    await vi.waitFor(() => expect(wrapper.text()).toContain("出水水质"));
    await flushPromises();

    const detailCall = fetchMock.mock.calls.find(([input]) =>
      String(input).includes("/graph/entities/entity-a"),
    );
    expect(detailCall).toBeDefined();
    expect(routerReplace).toHaveBeenCalled();
    expect(JSON.stringify(routerReplace.mock.calls[0]?.[0])).toContain("kb-1");
    expect(wrapper.text()).toContain("运行手册.pdf");
    expect(wrapper.text()).toContain("第 3-4 页");
    expect(wrapper.text()).toContain("浊度");
    expect(wrapper.text()).toContain("包含");
  });

  it("scopes the request to the first document and draws merges as dashed edges", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes("/v1/knowledge-bases?")) {
        return jsonResponse(knowledgeBasesPayload());
      }
      if (url.includes("/v1/knowledge-bases/kb-1/documents?")) {
        return jsonResponse(documentsPayload());
      }
      const payload: GraphOverview = overviewPayload();
      payload.edges = [
        ...payload.edges,
        {
          relation_id: "canonical_bridge:canonical-1:entity-a:entity-b",
          source_entity_id: "entity-a",
          target_entity_id: "entity-b",
          relation_type: "出水水质",
          kind: "canonical_bridge",
          document_id: null,
        },
      ];
      return jsonResponse(payload);
    });

    const wrapper = await mountView();

    await vi.waitFor(() => expect(wrapper.findAll(".graph-edge")).toHaveLength(2));
    const graphCalls = fetchMock.mock.calls
      .map(([input]) => String(input))
      .filter((url) => url.includes("/v1/knowledge-bases/kb-1/graph?"));
    expect(graphCalls.some((url) => url.includes("document_id=doc-1"))).toBe(true);
    expect(wrapper.find(".graph-edge.is-merge").exists()).toBe(true);
  });

  it("keeps an empty graph explicit", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes("/v1/knowledge-bases?")) {
        return jsonResponse(knowledgeBasesPayload());
      }
      if (url.includes("/v1/knowledge-bases/kb-1/documents?")) {
        return jsonResponse(documentsPayload());
      }
      return jsonResponse({ ...overviewPayload(), nodes: [], edges: [] });
    });

    const wrapper = await mountView();

    await vi.waitFor(() => expect(wrapper.text()).toContain("这个知识库还没有图谱内容"));
    expect(wrapper.find(".graph-canvas").exists()).toBe(false);
  });

  it("shows a safe error when the graph service is unavailable", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes("/v1/knowledge-bases?")) {
        return jsonResponse(knowledgeBasesPayload());
      }
      if (url.includes("/v1/knowledge-bases/kb-1/documents?")) {
        return jsonResponse(documentsPayload());
      }
      return jsonResponse({ detail: "Neo4j unavailable" }, 503);
    });

    const wrapper = await mountView();

    await vi.waitFor(() => expect(wrapper.text()).toContain("图谱暂时无法加载"));
    expect(wrapper.text()).toContain("服务暂时不可用，请稍后重试。");
    expect(wrapper.find(".graph-canvas").exists()).toBe(false);
  });
});
