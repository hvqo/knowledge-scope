import { mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ChatConversation, ChatMessageView, KnowledgeBase, RAGCitation } from "../../api/types";
import AssistantMessage from "./AssistantMessage.vue";
import ChatComposer from "./ChatComposer.vue";
import ChatMessageList from "./ChatMessageList.vue";
import ChatSidebar from "./ChatSidebar.vue";
import EvidenceCard from "./EvidenceCard.vue";

const { confirmMock } = vi.hoisted(() => ({ confirmMock: vi.fn() }));

vi.mock("element-plus", async (importOriginal) => {
  const actual = await importOriginal<typeof import("element-plus")>();
  return {
    ...actual,
    ElMessage: { success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() },
    ElMessageBox: { confirm: confirmMock },
  };
});

const knowledgeBases: KnowledgeBase[] = [
  {
    id: "kb-1",
    name: "产品资料",
    description: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  },
];

const conversations: ChatConversation[] = [
  {
    id: "conversation-1",
    title: "产品有哪些质量要求？",
    knowledge_base_id: "kb-1",
    project_id: null,
    message_count: 2,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: new Date().toISOString(),
  },
  {
    id: "conversation-2",
    title: "另一个问题",
    knowledge_base_id: null,
    project_id: null,
    message_count: 0,
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
  branch_provenance: [
    {
      branch: "multimodal",
      rank: 1,
      score: 0.8,
      graph_seed_entity_id: null,
      graph_retrieval_reason: null,
    },
  ],
};

function sidebarProps() {
  return {
    conversations,
    activeConversationId: "conversation-1",
    conversationsLoading: false,
    conversationsError: false,
    conversationsErrorMessage: "",
    knowledgeBases,
    selectedKnowledgeBaseId: "kb-1",
    knowledgeBasesLoading: false,
    knowledgeBasesError: false,
    knowledgeBasesErrorMessage: "",
    projects: [],
    memoryCount: 0,
    isBusy: false,
    isOpen: true,
  };
}

afterEach(() => {
  confirmMock.mockReset();
});

describe("chat sidebar", () => {
  it("creates, selects, and rebinds conversations", async () => {
    const wrapper = mount(ChatSidebar, { props: sidebarProps() });

    await wrapper.get(".new-conversation").trigger("click");
    expect(wrapper.emitted("newConversation")).toHaveLength(1);

    await wrapper.findAll(".conversation-main")[1].trigger("click");
    expect(wrapper.emitted("selectConversation")?.[0]).toEqual(["conversation-2"]);

    await wrapper.get("select").setValue("");
    expect(wrapper.emitted("selectKnowledgeBase")?.[0]).toEqual([null]);

    expect(wrapper.text()).toContain("产品有哪些质量要求？");
    expect(wrapper.text()).toContain("刚刚");
    expect(wrapper.text()).toContain("未绑定知识库");
  });

  it("renames a conversation inline", async () => {
    const wrapper = mount(ChatSidebar, { props: sidebarProps() });

    await wrapper.get('[title="重命名"]').trigger("click");
    const input = wrapper.get(".conversation-rename");
    expect((input.element as HTMLInputElement).value).toBe("产品有哪些质量要求？");

    await input.setValue("质量要求清单");
    await input.trigger("keydown", { key: "Enter" });

    expect(wrapper.emitted("renameConversation")?.[0]).toEqual(["conversation-1", "质量要求清单"]);
  });

  it("only deletes after the confirmation dialog is accepted", async () => {
    confirmMock.mockRejectedValueOnce(new Error("cancelled"));
    const wrapper = mount(ChatSidebar, { props: sidebarProps() });

    const deleteButton = wrapper.findAll(".conversation-action--danger")[0];
    await deleteButton.trigger("click");
    await vi.waitFor(() => expect(confirmMock).toHaveBeenCalledTimes(1));
    expect(wrapper.emitted("deleteConversation")).toBeUndefined();

    confirmMock.mockResolvedValueOnce("confirm");
    await deleteButton.trigger("click");
    await vi.waitFor(() =>
      expect(wrapper.emitted("deleteConversation")?.[0]).toEqual(["conversation-1"]),
    );
  });
});

describe("chat composer", () => {
  it("sends with Enter and stops while streaming", async () => {
    const wrapper = mount(ChatComposer, { props: { disabled: false, canSend: true, retrievalMode: "unified" } });

    await wrapper.get("textarea").setValue("查找资料");
    await wrapper.get("textarea").trigger("keydown", { key: "Enter", shiftKey: false });
    expect(wrapper.emitted("send")?.[0]).toEqual(["查找资料"]);
    expect((wrapper.get("textarea").element as HTMLTextAreaElement).value).toBe("");

    await wrapper.setProps({ disabled: true });
    await wrapper.get(".composer__button--stop").trigger("click");
    expect(wrapper.emitted("stop")).toHaveLength(1);
  });

  it("switches the retrieval scope", async () => {
    const wrapper = mount(ChatComposer, {
      props: { disabled: false, canSend: true, retrievalMode: "auto" },
    });

    expect(wrapper.get(".composer__mode.is-active").text()).toBe("智能");

    await wrapper.findAll(".composer__mode")[2].trigger("click");
    expect(wrapper.emitted("update:retrievalMode")?.[0]).toEqual(["dense"]);

    await wrapper.findAll(".composer__mode")[1].trigger("click");
    expect(wrapper.emitted("update:retrievalMode")?.[1]).toEqual(["unified"]);
  });

  it("keeps Shift + Enter for new lines and blocks empty sends", async () => {
    const wrapper = mount(ChatComposer, { props: { disabled: false, canSend: true, retrievalMode: "unified" } });

    await wrapper.get("textarea").setValue("草稿");
    await wrapper.get("textarea").trigger("keydown", { key: "Enter", shiftKey: true });
    expect(wrapper.emitted("send")).toBeUndefined();

    await wrapper.get("textarea").setValue("   ");
    await wrapper.get(".composer__button").trigger("click");
    expect(wrapper.emitted("send")).toBeUndefined();
  });
});

describe("chat messages", () => {
  const messages: ChatMessageView[] = [
    {
      id: "message-1",
      role: "user",
      status: "complete",
      content: "产品有哪些质量要求？",
      citations: [],
      position: 0,
      created_at: "2026-01-01T00:00:00Z",
    },
    {
      id: "message-2",
      role: "assistant",
      status: "complete",
      content: "需要满足三点要求 [C1]",
      citations: [citation],
      position: 1,
      created_at: "2026-01-01T00:00:00Z",
    },
  ];

  it("renders both roles and opens sources from a message", async () => {
    const wrapper = mount(ChatMessageList, {
      props: { messages, isStreaming: false, isReady: true },
    });

    expect(wrapper.text()).toContain("产品有哪些质量要求？");
    expect(wrapper.text()).toContain("需要满足三点要求");
    expect(wrapper.findAll(".message-action")).toHaveLength(2);

    await wrapper.findAll(".message-action")[1].trigger("click");
    expect(wrapper.emitted("openSources")?.[0]).toEqual([[citation]]);

    await wrapper.get(".citation-marker").trigger("click");
    expect(wrapper.emitted("selectCitation")?.[0]).toEqual(["C1", [citation]]);
  });

  it("lists only the sources the answer actually references", async () => {
    const wrapper = mount(ChatMessageList, {
      props: {
        messages: [
          {
            ...messages[1],
            id: "message-3",
            content: "我是基于当前知识库的资料问答助手，资料中没有相关说明。",
            citations: [citation],
          },
        ],
        isStreaming: false,
        isReady: true,
      },
    });

    const labels = wrapper.findAll(".message-action").map((action) => action.text());
    expect(labels).toEqual(["复制"]);
    expect(wrapper.text()).not.toContain("来源");
  });

  it("offers example prompts only when a knowledge base is selected", async () => {
    const wrapper = mount(ChatMessageList, {
      props: { messages: [], isStreaming: false, isReady: true },
    });

    const prompts = wrapper.findAll(".prompt-card");
    expect(prompts.length).toBeGreaterThan(0);
    await prompts[0].trigger("click");
    expect(wrapper.emitted("selectPrompt")).toHaveLength(1);

    const disabledWrapper = mount(ChatMessageList, {
      props: { messages: [], isStreaming: false, isReady: false },
    });
    expect(disabledWrapper.get(".prompt-card").attributes("disabled")).toBeDefined();
    expect(disabledWrapper.text()).toContain("请先在左侧选择一个知识库");
  });

  it("shows a reading hint while the answer is still streaming", () => {
    const wrapper = mount(ChatMessageList, {
      props: { messages: [messages[0]], isStreaming: true, isReady: true },
    });

    expect(wrapper.text()).toContain("正在阅读资料…");
  });
});

describe("assistant message", () => {
  it("makes citations selectable and renders multimodal metadata", async () => {
    const messageWrapper = mount(AssistantMessage, {
      props: {
        message: { content: "答案 [C1]", status: "complete", citations: [citation] },
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
