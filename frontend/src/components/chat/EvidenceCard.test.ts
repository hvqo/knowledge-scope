import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import EvidenceCard from "./EvidenceCard.vue";
import type { RAGCitation } from "../../api/types";

function makeCitation(overrides: Partial<RAGCitation> = {}): RAGCitation {
  return {
    marker: "C1",
    document_id: "doc-1",
    knowledge_base_id: "kb-1",
    candidate_kind: "evidence",
    chunk_id: null,
    evidence_id: "evidence_v1_" + "a".repeat(64),
    modality: "image",
    representation_ids: ["representation_v1_" + "b".repeat(64)],
    asset_refs: ["sample/auto/images/figure.jpg"],
    page_start: 2,
    page_end: 2,
    source_block_ids: ["p2-b1"],
    section_path: ["第一章"],
    section_title: "光合作用",
    snippet: "图 5-14 光合作用过程的示意图",
    snippet_kind: "representation",
    final_rank: 1,
    final_reranker_score: 0.9,
    branch_provenance: [],
    ...overrides,
  };
}

describe("EvidenceCard", () => {
  it("selects the card from the hit area", async () => {
    const wrapper = mount(EvidenceCard, {
      props: { citation: makeCitation(), selected: false, documentTitle: "生物教材" },
    });

    await wrapper.get(".evidence-card__hit").trigger("click");

    expect(wrapper.emitted("select")).toHaveLength(1);
    expect(wrapper.emitted("preview")).toBeUndefined();
  });

  it("offers original asset preview for evidence citations", async () => {
    const wrapper = mount(EvidenceCard, {
      props: { citation: makeCitation(), selected: false, documentTitle: null },
    });

    const action = wrapper.get(".evidence-card__action");
    expect(action.text()).toContain("查看来源素材");
    await action.trigger("click");

    expect(wrapper.emitted("preview")).toHaveLength(1);
    expect(wrapper.emitted("select")).toBeUndefined();
  });

  it("hides preview for chunk citations", () => {
    const wrapper = mount(EvidenceCard, {
      props: {
        citation: makeCitation({
          candidate_kind: "chunk",
          chunk_id: "chunk-1",
          evidence_id: null,
          modality: null,
          asset_refs: [],
        }),
        selected: true,
        documentTitle: null,
      },
    });

    expect(wrapper.find(".evidence-card__action").exists()).toBe(false);
    expect(wrapper.get(".evidence-card").classes()).toContain("is-selected");
  });
});
