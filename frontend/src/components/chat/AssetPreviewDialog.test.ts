import { VueQueryPlugin, QueryClient } from "@tanstack/vue-query";
import { mount } from "@vue/test-utils";
import { defineComponent, h } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { EvidenceDetail, RAGCitation } from "../../api/types";

const mocks = vi.hoisted(() => ({
  fetchEvidenceDetail: vi.fn(),
  evidenceAssetUrl: vi.fn(),
  documentFileUrl: vi.fn(),
}));

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/client")>();
  return { ...actual, ...mocks };
});

import { ApiError } from "../../api/client";
import AssetPreviewDialog from "./AssetPreviewDialog.vue";

const ElDialogStub = defineComponent({
  name: "ElDialog",
  props: { modelValue: { type: Boolean, default: false }, title: { type: String, default: "" } },
  emits: ["update:modelValue"],
  setup(_props, { slots }) {
    return () => h("div", { class: "el-dialog-stub" }, slots.default?.());
  },
});

const EVIDENCE_ID = "evidence_v1_" + "a".repeat(64);

function makeCitation(overrides: Partial<RAGCitation> = {}): RAGCitation {
  return {
    marker: "C1",
    document_id: "doc-1",
    knowledge_base_id: "kb-1",
    candidate_kind: "evidence",
    chunk_id: null,
    evidence_id: EVIDENCE_ID,
    modality: "image",
    representation_ids: [],
    asset_refs: ["sample/auto/images/figure.jpg"],
    page_start: 2,
    page_end: 2,
    source_block_ids: ["p2-b1"],
    section_path: ["第一章", "光合作用"],
    section_title: "光合作用",
    snippet: "图 5-14 光合作用过程的示意图",
    snippet_kind: "representation",
    final_rank: 1,
    final_reranker_score: 0.9,
    branch_provenance: [],
    ...overrides,
  };
}

function makeDetail(overrides: Partial<EvidenceDetail> = {}): EvidenceDetail {
  return {
    document_id: "doc-1",
    knowledge_base_id: "kb-1",
    evidence_id: EVIDENCE_ID,
    modality: "image",
    page_start: 2,
    page_end: 2,
    source_block_ids: ["p2-b1"],
    section_path: ["第一章", "光合作用"],
    source_fingerprint: "c".repeat(64),
    representations: [
      {
        representation_id: "representation_v1_" + "b".repeat(64),
        representation_type: "caption",
        content: "图 5-14 光合作用过程的示意图",
        searchable: true,
      },
      {
        representation_id: "representation_v1_" + "d".repeat(64),
        representation_type: "asset_ref",
        content: null,
        searchable: false,
      },
    ],
    assets: [{ index: 0, media_type: "image/png", size_bytes: 100, available: true }],
    ...overrides,
  };
}

function mountDialog(citation: RAGCitation = makeCitation()) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return mount(AssetPreviewDialog, {
    props: {
      modelValue: true,
      knowledgeBaseId: "kb-1",
      citation,
      documentTitle: "生物教材",
    },
    global: {
      plugins: [[VueQueryPlugin, { queryClient }]],
      stubs: { ElDialog: ElDialogStub },
    },
  });
}

describe("AssetPreviewDialog", () => {
  beforeEach(() => {
    mocks.fetchEvidenceDetail.mockReset();
    mocks.evidenceAssetUrl.mockReset();
    mocks.documentFileUrl.mockReset();
    mocks.evidenceAssetUrl.mockImplementation(
      (knowledgeBaseId: string, documentId: string, evidenceId: string, index: number) =>
        `/assets/${knowledgeBaseId}/${documentId}/${evidenceId}/${index}`,
    );
    mocks.documentFileUrl.mockImplementation(
      (knowledgeBaseId: string, documentId: string, page?: number) =>
        `/file/${knowledgeBaseId}/${documentId}#page=${page}`,
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows the original image for image evidence", async () => {
    mocks.fetchEvidenceDetail.mockResolvedValue(makeDetail());
    const wrapper = mountDialog();

    await vi.waitFor(() => {
      expect(wrapper.find(".asset-preview__image").exists()).toBe(true);
    });
    expect(mocks.fetchEvidenceDetail).toHaveBeenCalledWith("kb-1", "doc-1", EVIDENCE_ID);
    expect(wrapper.get(".asset-preview__image").attributes("src")).toBe(
      `/assets/kb-1/doc-1/${EVIDENCE_ID}/0`,
    );
    expect(wrapper.text()).toContain("图 5-14 光合作用过程的示意图");
    expect(wrapper.text()).toContain("仅用于核验来源");
  });

  it("renders table evidence as safe structured rows without executing HTML", async () => {
    mocks.fetchEvidenceDetail.mockResolvedValue(
      makeDetail({
        modality: "table",
        representations: [
          {
            representation_id: "representation_v1_" + "b".repeat(64),
            representation_type: "html",
            content:
              '<table><tr><th>阶段</th><th>场所</th></tr><tr><td>光反应</td><td>类囊体薄膜</td></tr><script>window.__assetXss = true</script></table>',
            searchable: true,
          },
        ],
        assets: [],
      }),
    );
    const wrapper = mountDialog(makeCitation({ modality: "table" }));

    await vi.waitFor(() => {
      expect(wrapper.find(".asset-preview__table").exists()).toBe(true);
    });
    expect(wrapper.get(".asset-preview__table").text()).toContain("光反应");
    expect(wrapper.get(".asset-preview__table").text()).toContain("类囊体薄膜");
    expect(wrapper.find("script").exists()).toBe(false);
    expect((window as unknown as { __assetXss?: boolean }).__assetXss).toBeUndefined();
  });

  it("renders formula evidence with the shared LaTeX renderer", async () => {
    mocks.fetchEvidenceDetail.mockResolvedValue(
      makeDetail({
        modality: "formula",
        representations: [
          {
            representation_id: "representation_v1_" + "b".repeat(64),
            representation_type: "latex",
            content: "6CO_2+6H_2O \\rightarrow C_6H_{12}O_6",
            searchable: true,
          },
        ],
        assets: [],
      }),
    );
    const wrapper = mountDialog(makeCitation({ modality: "formula" }));

    await vi.waitFor(() => {
      expect(wrapper.find(".asset-preview__formula .katex").exists()).toBe(true);
    });
  });

  it("degrades to a controlled message when the asset was not retained", async () => {
    mocks.fetchEvidenceDetail.mockResolvedValue(
      makeDetail({
        assets: [{ index: 0, media_type: null, size_bytes: null, available: false }],
      }),
    );
    const wrapper = mountDialog();

    await vi.waitFor(() => {
      expect(wrapper.text()).toContain("该来源的原始图片未能保留");
    });
    expect(wrapper.find(".asset-preview__image").exists()).toBe(false);
  });

  it("surfaces the server controlled error without internal paths", async () => {
    mocks.fetchEvidenceDetail.mockRejectedValue(
      new ApiError(404, "Not Found", "证据产物不存在、已过期或不可用"),
    );
    const wrapper = mountDialog();

    await vi.waitFor(() => {
      expect(wrapper.text()).toContain("证据产物不存在、已过期或不可用");
    });
    expect(wrapper.text()).not.toContain("/home/");
    expect(wrapper.html()).not.toContain("mineru");
  });

  it("opens the original PDF page from the footer", async () => {
    mocks.fetchEvidenceDetail.mockResolvedValue(makeDetail());
    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null);
    const wrapper = mountDialog();

    await vi.waitFor(() => {
      expect(wrapper.find(".asset-preview__image").exists()).toBe(true);
    });
    await wrapper
      .findAll(".asset-preview__button")
      .find((button) => button.text().includes("打开原文 PDF"))!
      .trigger("click");

    expect(mocks.documentFileUrl).toHaveBeenCalledWith("kb-1", "doc-1", 2);
    expect(openSpy).toHaveBeenCalledWith(
      "/file/kb-1/doc-1#page=2",
      "_blank",
      "noopener,noreferrer",
    );
  });

  it("emits close updates", async () => {
    mocks.fetchEvidenceDetail.mockResolvedValue(makeDetail());
    const wrapper = mountDialog();
    await vi.waitFor(() => {
      expect(wrapper.find(".asset-preview__image").exists()).toBe(true);
    });

    await wrapper.get(".asset-preview__button--ghost").trigger("click");

    expect(wrapper.emitted("update:modelValue")).toEqual([[false]]);
  });
});
