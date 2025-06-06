import { flushPromises, mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import type { ChatBIAnalysisResult, RAGCitation } from "../../api/types";
import ReportSourcePanel from "./ReportSourcePanel.vue";

const citation: RAGCitation = {
  marker: "C1",
  document_id: "document-1",
  knowledge_base_id: "kb-1",
  candidate_kind: "chunk",
  chunk_id: "chunk-1",
  evidence_id: null,
  modality: null,
  representation_ids: [],
  asset_refs: [],
  page_start: 2,
  page_end: 3,
  source_block_ids: ["block-1"],
  section_path: ["质量控制"],
  section_title: "质量控制",
  snippet: "原始证据片段",
  snippet_kind: "source",
  final_rank: null,
  final_reranker_score: null,
  branch_provenance: [],
};

const chatBIResult: ChatBIAnalysisResult = {
  query_id: "query-1",
  datasource_id: "ds-1",
  execution_status: "succeeded",
  eligibility: null,
  answer: "华东最多。",
  columns: [
    { name: "地区", data_type: "text", nullable: false, ordinal: 0 },
    { name: "数量", data_type: "bigint", nullable: false, ordinal: 1 },
  ],
  rows: [["华东", 3], ["华南", 2]],
  row_count: 2,
  truncated: false,
  truncation_reason: null,
  redacted_sql: "SELECT region, COUNT(*) FROM public.sales GROUP BY region",
  warnings: [],
  error_message: null,
  elapsed_ms: 10,
};

function mountPanel() {
  return mount(ReportSourcePanel, {
    props: {
      knowledgeBaseId: "kb-1",
      datasourceId: "ds-1",
      documentTitles: { "document-1": "质量手册.pdf" },
      references: [],
      ragAnswer: "已找到证据。",
      ragCitations: [citation],
      ragLoading: false,
      ragError: null,
      chatBIResult,
      chatBILoading: false,
      chatBIError: null,
      disabled: false,
    },
  });
}

describe("ReportSourcePanel", () => {
  it("renders grounded RAG evidence and emits insertion for the active section", async () => {
    const wrapper = mountPanel();
    expect(wrapper.text()).toContain("质量手册.pdf");
    expect(wrapper.text()).toContain("原始证据片段");
    await wrapper.get(".report-source-panel__insert").trigger("click");
    expect(wrapper.emitted("insertRag")?.[0]).toEqual([citation]);
  });

  it("reuses the ChatBI result table/chart and emits the question plus chart contract", async () => {
    const wrapper = mountPanel();
    await wrapper.findAll("[role='tab']")[1].trigger("click");
    await flushPromises();

    expect(wrapper.find(".result-table").exists()).toBe(true);
    expect(wrapper.find(".result-chart").exists()).toBe(true);
    const question = wrapper.get("textarea");
    await question.setValue("统计各地区数量");
    await wrapper.get("form").trigger("submit");
    await flushPromises();
    await wrapper.setProps({ chatBIResult: null });
    await wrapper.setProps({ chatBIResult: { ...chatBIResult, answer: "新结果" } });
    await wrapper.get(".report-source-panel__insert").trigger("click");
    const emitted = wrapper.emitted("insertChatBI");
    expect(emitted).toHaveLength(1);
    expect(emitted?.[0]?.[0]).toBe("统计各地区数量");
    expect(emitted?.[0]?.[1]).toMatchObject({
      kind: "pie",
      category_column: "地区",
      value_column: "数量",
    });
  });
});
