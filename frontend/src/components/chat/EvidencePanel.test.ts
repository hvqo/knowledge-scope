import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import EvidencePanel from "./EvidencePanel.vue";
import type { RAGCitation } from "../../api/types";

function makeCitation(marker: string): RAGCitation {
  return {
    marker,
    document_id: "doc-1",
    knowledge_base_id: "kb-1",
    candidate_kind: "evidence",
    chunk_id: null,
    evidence_id: "evidence_v1_" + marker.toLowerCase().padEnd(64, "0"),
    modality: "table",
    representation_ids: [],
    asset_refs: ["sample/auto/images/table.jpg"],
    page_start: 3,
    page_end: 3,
    source_block_ids: ["p3-b2"],
    section_path: [],
    section_title: "表格",
    snippet: null,
    snippet_kind: "representation",
    final_rank: 2,
    final_reranker_score: 0.8,
    branch_provenance: [],
  };
}

describe("EvidencePanel", () => {
  it("forwards select and preview events with the clicked marker", async () => {
    const wrapper = mount(EvidencePanel, {
      props: {
        citations: [makeCitation("C1"), makeCitation("C2")],
        selectedMarker: null,
        documentTitles: { "doc-1": "教材" },
      },
    });

    await wrapper.findAll(".evidence-card__action")[1].trigger("click");
    expect(wrapper.emitted("previewCitation")).toEqual([["C2"]]);

    await wrapper.findAll(".evidence-card__hit")[0].trigger("click");
    expect(wrapper.emitted("selectCitation")).toEqual([["C1"]]);
  });
});
