import { describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";

import ChatSidebar from "./ChatSidebar.vue";
import type { ChatConversation, ChatProject } from "../../api/types";

vi.mock("element-plus", () => ({
  ElMessageBox: {
    confirm: vi.fn().mockResolvedValue("confirm"),
    prompt: vi.fn().mockResolvedValue({ value: "新项目" }),
  },
}));

const projects: ChatProject[] = [
  {
    id: "project-1",
    title: "中考化学复习",
    description: null,
    conversation_count: 1,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  },
];

const conversations: ChatConversation[] = [
  {
    id: "conversation-1",
    title: "项目里的对话",
    knowledge_base_id: "kb-1",
    project_id: "project-1",
    message_count: 2,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  },
  {
    id: "conversation-2",
    title: "未分组的对话",
    knowledge_base_id: "kb-1",
    project_id: null,
    message_count: 1,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  },
];

function sidebarProps() {
  return {
    conversations,
    projects,
    activeConversationId: "conversation-1",
    conversationsLoading: false,
    conversationsError: false,
    conversationsErrorMessage: "",
    knowledgeBases: [
      {
        id: "kb-1",
        name: "产品资料",
        description: null,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      },
    ],
    selectedKnowledgeBaseId: "kb-1",
    knowledgeBasesLoading: false,
    knowledgeBasesError: false,
    knowledgeBasesErrorMessage: "",
    memoryCount: 2,
    isBusy: false,
    isOpen: true,
  };
}

describe("ChatSidebar projects", () => {
  it("shows a projects section and a recent section", () => {
    const wrapper = mount(ChatSidebar, { props: sidebarProps() });

    expect(wrapper.get(".projects-head__toggle").text()).toContain("项目");
    expect(wrapper.get(".project-row__name").text()).toBe("中考化学复习");
    // Project conversations live under their row; recent holds the ungrouped.
    expect(wrapper.findAll(".project-row__conversations .conversation-item")).toHaveLength(1);
    const recent = wrapper.findAll(".conversation-block .conversation-item");
    expect(recent).toHaveLength(1);
    expect(recent[0]?.text()).toContain("未分组的对话");
  });

  it("collapses a project so its conversations hide", async () => {
    const wrapper = mount(ChatSidebar, { props: sidebarProps() });

    await wrapper.get(".project-row__main").trigger("click");
    expect(wrapper.findAll(".project-row__conversations .conversation-item")).toHaveLength(0);
    await wrapper.get(".project-row__main").trigger("click");
    expect(wrapper.findAll(".project-row__conversations .conversation-item")).toHaveLength(1);
  });

  it("moves a conversation between a project and the ungrouped bucket", async () => {
    const wrapper = mount(ChatSidebar, { props: sidebarProps() });

    await wrapper
      .findAll(".conversation-item")
      .find((item) => item.text().includes("未分组的对话"))
      ?.get('[title="移动到项目"]')
      .trigger("click");
    expect(wrapper.find(".move-menu").exists()).toBe(true);

    await wrapper.get(".move-menu__option").trigger("click");
    expect(wrapper.emitted("moveConversation")?.[0]).toEqual(["conversation-2", "project-1"]);
  });

  it("moves a conversation out of its project", async () => {
    const wrapper = mount(ChatSidebar, { props: sidebarProps() });

    await wrapper
      .findAll(".conversation-item")
      .find((item) => item.text().includes("项目里的对话"))
      ?.get('[title="移动到项目"]')
      .trigger("click");

    const options = wrapper.findAll(".move-menu__option");
    await options.at(-1)!.trigger("click");
    expect(wrapper.emitted("moveConversation")?.[0]).toEqual(["conversation-1", null]);
  });

  it("creates a project from the section header", async () => {
    const wrapper = mount(ChatSidebar, { props: sidebarProps() });

    await wrapper.get('[title="新建项目"]').trigger("click");

    expect(wrapper.emitted("createProject")?.[0]).toEqual(["新项目"]);
  });

  it("deletes a project without deleting its conversations", async () => {
    const wrapper = mount(ChatSidebar, { props: sidebarProps() });

    await wrapper.get('[title="删除项目"]').trigger("click");
    await vi.waitFor(() => expect(wrapper.emitted("deleteProject")).toBeDefined());
    expect(wrapper.emitted("deleteProject")?.[0]).toEqual(["project-1"]);
  });

  it("renames a project inline", async () => {
    const wrapper = mount(ChatSidebar, { props: sidebarProps() });

    await wrapper.get('[title="重命名项目"]').trigger("click");
    const input = wrapper.get(".conversation-rename");
    await input.setValue("化学冲刺");
    await input.trigger("keydown", { key: "Enter" });

    expect(wrapper.emitted("renameProject")?.[0]).toEqual(["project-1", "化学冲刺"]);
  });

  it("surfaces the memory entry with its count", async () => {
    const wrapper = mount(ChatSidebar, { props: sidebarProps() });

    expect(wrapper.get(".memory-entry").text()).toContain("2");
    await wrapper.get(".memory-entry").trigger("click");
    expect(wrapper.emitted("openMemories")).toHaveLength(1);
  });
});
