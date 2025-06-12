import { VueQueryPlugin, QueryClient } from "@tanstack/vue-query";
import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Document } from "../../api/types";
import DocumentPreviewDialog from "./DocumentPreviewDialog.vue";

const document: Document = {
  id: "doc-1",
  knowledge_base_id: "kb-1",
  original_filename: "质量手册.pdf",
  media_type: "application/pdf",
  size_bytes: 2048,
  sha256: "a".repeat(64),
  status: "uploaded",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function pdfResponse(): Response {
  return new Response(new Uint8Array([0x25, 0x50, 0x44, 0x46]), {
    status: 206,
    headers: { "Content-Type": "application/pdf" },
  });
}

function installFetchRoutes(routes: { chunks: () => Response; file: () => Response }) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    if (url.includes("/chunks")) {
      return routes.chunks();
    }
    if (url.includes("/file")) {
      return routes.file();
    }
    throw new Error("unexpected request: " + url);
  });
}

function mountDialog() {
  return mount(DocumentPreviewDialog, {
    props: { modelValue: true, knowledgeBaseId: "kb-1", document },
    global: {
      plugins: [
        [
          VueQueryPlugin,
          { queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }) },
        ],
      ],
    },
  });
}

describe("DocumentPreviewDialog", () => {
  afterEach(() => vi.restoreAllMocks());

  it("lists persisted chunks and moves the embedded PDF to the selected page", async () => {
    const fetchMock = installFetchRoutes({
      chunks: () =>
        jsonResponse({
          document_id: "doc-1",
          page_count: 12,
          items: [
            {
              chunk_id: "chunk-1",
              ordinal: 0,
              page_start: 1,
              page_end: 2,
              section_path: ["质量控制"],
              content_types: ["text"],
              asset_refs: [],
              text: "第一章 质量控制要求。",
            },
            {
              chunk_id: "chunk-2",
              ordinal: 1,
              page_start: 5,
              page_end: 5,
              section_path: ["质量控制", "检验"],
              content_types: ["table"],
              asset_refs: [],
              text: "检验记录表",
            },
          ],
        }),
      file: pdfResponse,
    });

    const wrapper = mountDialog();
    await vi.waitFor(() => expect(wrapper.find("iframe").exists()).toBe(true));

    expect(fetchMock.mock.calls.some(([input]) => String(input).includes("/file"))).toBe(true);
    expect(
      fetchMock.mock.calls.some(([input]) => String(input).includes("/documents/doc-1/chunks")),
    ).toBe(true);
    await vi.waitFor(() =>
      expect(wrapper.findAll(".document-preview__chunk")).toHaveLength(2),
    );
    expect(wrapper.text()).toContain("共 12 页");
    expect(wrapper.text()).toContain("第一章 质量控制要求。");
    expect(wrapper.get("iframe").attributes("src")).toContain("/documents/doc-1/file#page=1");

    await wrapper.findAll(".document-preview__chunk")[1].trigger("click");
    await flushPromises();

    expect(wrapper.get(".document-preview__viewer-page").text()).toContain("第 5 页");
    expect(wrapper.get("iframe").attributes("src")).toContain("/documents/doc-1/file#page=5");
  });

  it("keeps the source PDF previewable when no chunks exist yet", async () => {
    installFetchRoutes({
      chunks: () => jsonResponse({ document_id: "doc-1", page_count: null, items: [] }),
      file: pdfResponse,
    });

    const wrapper = mountDialog();
    await vi.waitFor(() => expect(wrapper.find("iframe").exists()).toBe(true));

    expect(wrapper.text()).toContain("该文档还没有切片，仍可预览原始 PDF。");
    expect(wrapper.get("iframe").attributes("src")).toContain("/documents/doc-1/file#page=1");
  });

  it("explains why a source PDF cannot be previewed instead of rendering the error body", async () => {
    installFetchRoutes({
      chunks: () => jsonResponse({ document_id: "doc-1", page_count: null, items: [] }),
      file: () => jsonResponse({ detail: "该文档来自外部语料, 未配置原始文件目录" }, 404),
    });

    const wrapper = mountDialog();
    await vi.waitFor(() =>
      expect(wrapper.text()).toContain("该文档来自外部语料, 未配置原始文件目录"),
    );

    expect(wrapper.find("iframe").exists()).toBe(false);
  });
});
