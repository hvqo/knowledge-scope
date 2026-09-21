import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

import type { KnowledgeBase, RAGCitation } from "../../api/types";
import AssistantMessage from "./AssistantMessage.vue";
import ChatComposer from "./ChatComposer.vue";
import ConversationSidebar from "./ConversationSidebar.vue";
import EvidenceCard from "./EvidenceCard.vue";
import { useChatStore } from "../../stores/chat";

const knowledgeBases: KnowledgeBase[] = [
  {
    id: "kb-1",
    name: "产品资料",
    description: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  },
];

const citation: RAGCitation = {
  marker: "C1",
  document_id: "doc-1",
  knowledge_base_id: "kb-1",
  candidate_kind: "evidence",
  chunk_id: null,
  evidence_id: "evidence-1",
  modality: "table",
  representation_ids: ["representation-1"],
  asset_refs: ["assets/table-1"],
  page_start: 2,
  page_end: 2,
  source_block_ids: ["block-1"],
  section_path: ["章节", "表格"],
  section_title: "表格",
  snippet: "表格检索表示",
  snippet_kind: "representation",
  final_rank: 1,
  final_reranker_score: 0.8,
  branch_provenance: [{
    branch: "multimodal",
    rank: 1,
    score: 0.8,
    graph_seed_entity_id: null,
    graph_retrieval_reason: null,
  }],
};

describe("chat workspace components", () => {
  it("selects a knowledge base and creates session conversations", async () => {
    setActivePinia(createPinia());
    const store = useChatStore();
    const wrapper = mount(ConversationSidebar, {
      props: {
        conversations: store.conversations,
        activeConversationId: null,
        knowledgeBases,
        selectedKnowledgeBaseId: null,
        knowledgeBasesLoading: false,
        knowledgeBasesError: false,
        knowledgeBasesErrorMessage: "",
      },
    });

    await wrapper.get("select").setValue("kb-1");
    expect(wrapper.emitted("selectKnowledgeBase")?.[0]).toEqual(["kb-1"]);
    await wrapper.get(".new-conversation-button").trigger("click");
    expect(wrapper.emitted("newConversation")).toHaveLength(1);

    store.setSelectedKnowledgeBase("kb-1");
    const id = store.newConversation();
    const messageId = store.addMessage(id, "assistant", "回答", "complete");
    store.updateMessage(id, messageId, (message) => {
      message.citations = [citation];
    });
    expect(store.activeConversation?.knowledgeBaseId).toBe("kb-1");
    expect(store.activeConversation?.messages[0].citations[0].evidence_id).toBe("evidence-1");
  });

  it("sends from the multiline composer and supports stopping", async () => {
    const wrapper = mount(ChatComposer, { props: { disabled: false, canSend: true } });
    await wrapper.get("textarea").setValue("查找资料");
    await wrapper.get("textarea").trigger("keydown", { key: "Enter", shiftKey: false });
    expect(wrapper.emitted("send")?.[0]).toEqual(["查找资料"]);

    await wrapper.setProps({ disabled: true });
    await wrapper.get(".composer__button--stop").trigger("click");
    expect(wrapper.emitted("stop")).toHaveLength(1);
  });

  it("makes citations selectable and renders multimodal metadata", async () => {
    const messageWrapper = mount(AssistantMessage, {
      props: {
        message: {
          id: "message-1",
          role: "assistant",
          content: "答案 [C1]",
          status: "complete",
          citations: [citation],
        },
      },
    });
    await messageWrapper.get(".citation-marker").trigger("click");
    expect(messageWrapper.emitted("selectCitation")?.[0]).toEqual(["C1"]);

    const cardWrapper = mount(EvidenceCard, {
      props: { citation, selected: true, documentTitle: "资料.pdf" },
    });
    expect(cardWrapper.text()).toContain("资料.pdf");
    expect(cardWrapper.text()).toContain("表格证据");
    expect(cardWrapper.text()).toContain("检索表示不替代原始来源");
  });
});
