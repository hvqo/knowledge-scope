import { computed, ref } from "vue";
import { defineStore } from "pinia";

import type { RAGCitation } from "../api/types";

export type ChatMessageRole = "user" | "assistant";
export type ChatMessageStatus = "complete" | "streaming" | "error";

export interface ChatMessage {
  id: string;
  role: ChatMessageRole;
  content: string;
  status: ChatMessageStatus;
  citations: RAGCitation[];
}

export interface ChatConversation {
  id: string;
  title: string;
  knowledgeBaseId: string | null;
  messages: ChatMessage[];
  updatedAt: number;
}

function createId(prefix: string): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function createConversation(knowledgeBaseId: string | null): ChatConversation {
  return {
    id: createId("conversation"),
    title: "新对话",
    knowledgeBaseId,
    messages: [],
    updatedAt: Date.now(),
  };
}

export const useChatStore = defineStore("chat", () => {
  const conversations = ref<ChatConversation[]>([]);
  const activeConversationId = ref<string | null>(null);
  const selectedKnowledgeBaseId = ref<string | null>(null);
  const activeConversation = computed(() =>
    conversations.value.find((conversation) => conversation.id === activeConversationId.value) ??
    null,
  );

  function newConversation(knowledgeBaseId = selectedKnowledgeBaseId.value): string {
    const conversation = createConversation(knowledgeBaseId);
    conversations.value.unshift(conversation);
    activeConversationId.value = conversation.id;
    return conversation.id;
  }

  function ensureActiveConversation(): ChatConversation {
    if (activeConversation.value !== null) {
      return activeConversation.value;
    }
    const id = newConversation();
    const conversation = conversations.value.find((item) => item.id === id);
    if (conversation === undefined) {
      throw new Error("conversation could not be created");
    }
    return conversation;
  }

  function selectConversation(id: string): void {
    const conversation = conversations.value.find((item) => item.id === id);
    if (conversation === undefined) {
      return;
    }
    activeConversationId.value = id;
    if (conversation.knowledgeBaseId !== null) {
      selectedKnowledgeBaseId.value = conversation.knowledgeBaseId;
    }
  }

  function setSelectedKnowledgeBase(id: string | null): void {
    selectedKnowledgeBaseId.value = id;
    const conversation = activeConversation.value;
    if (conversation !== null && conversation.messages.length === 0) {
      conversation.knowledgeBaseId = id;
      conversation.updatedAt = Date.now();
    }
  }

  function addMessage(
    conversationId: string,
    role: ChatMessageRole,
    content: string,
    status: ChatMessageStatus,
  ): string {
    const conversation = conversations.value.find((item) => item.id === conversationId);
    if (conversation === undefined) {
      throw new Error("conversation does not exist");
    }
    const message: ChatMessage = {
      id: createId("message"),
      role,
      content,
      status,
      citations: [],
    };
    conversation.messages.push(message);
    conversation.updatedAt = Date.now();
    if (role === "user" && conversation.title === "新对话") {
      conversation.title = content.trim().slice(0, 30) || "新对话";
    }
    return message.id;
  }

  function updateMessage(
    conversationId: string,
    messageId: string,
    update: (message: ChatMessage) => void,
  ): void {
    const conversation = conversations.value.find((item) => item.id === conversationId);
    const message = conversation?.messages.find((item) => item.id === messageId);
    if (conversation === undefined || message === undefined) {
      return;
    }
    update(message);
    conversation.updatedAt = Date.now();
  }

  return {
    conversations,
    activeConversationId,
    activeConversation,
    selectedKnowledgeBaseId,
    newConversation,
    ensureActiveConversation,
    selectConversation,
    setSelectedKnowledgeBase,
    addMessage,
    updateMessage,
  };
});
