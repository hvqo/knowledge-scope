import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";

import VisualSourcesCard from "./VisualSourcesCard.vue";
import type { RAGVisualSource } from "../../api/types";

vi.mock("../../api/client", () => ({
  evidenceAssetUrl: (kb: string, doc: string, evidence: string, index: number) =>
    `/api/v1/test/${kb}/${doc}/${evidence}/${index}`,
  documentFileUrl: (kb: string, doc: string, page?: number) =>
    `/api/v1/test/${kb}/${doc}/file#page=${page ?? 1}`,
}));

const source: RAGVisualSource = {
  knowledge_base_id: "kb-1",
  document_id: "doc-1",
  evidence_id: "evidence_v1_" + "a".repeat(64),
  page_start: 2,
  page_end: 2,
  source_block_ids: ["p2-b1"],
  asset_index: 0,
  label: "图 1 示例图片",
  fact_status: "visual_estimate",
};

describe("VisualSourcesCard", () => {
  it("renders the used image with a review notice", () => {
    const wrapper = mount(VisualSourcesCard, {
      props: {
        sources: [source],
        notice: "视觉数值未经核验，已省略精确数字并标注为视觉估计",
        gate: "visual_numeric_withheld",
        factStatus: "visual_estimate",
      },
    });

    expect(wrapper.text()).toContain("本次分析图片");
    expect(wrapper.text()).toContain("视觉分析结果需核对");
    expect(wrapper.text()).toContain("未经核验");
    const image = wrapper.get(".visual-sources__thumb img");
    expect(image.attributes("src")).toContain("evidence_v1_");
  });

  it("emits open when the thumbnail is clicked", async () => {
    const wrapper = mount(VisualSourcesCard, {
      props: {
        sources: [source],
        notice: null,
        gate: "none",
        factStatus: "visual_estimate",
      },
    });

    await wrapper.get(".visual-sources__thumb").trigger("click");
    expect(wrapper.emitted("open")?.[0]).toEqual([source]);
  });

  it("shows a missing-asset notice instead of a broken image", async () => {
    const wrapper = mount(VisualSourcesCard, {
      props: {
        sources: [source],
        notice: null,
        gate: "none",
        factStatus: "not_available",
      },
    });

    await wrapper.get(".visual-sources__thumb img").trigger("error");
    expect(wrapper.text()).toContain("原图不可用，无法核验");
    expect(wrapper.find(".visual-sources__thumb").exists()).toBe(false);
  });
});
