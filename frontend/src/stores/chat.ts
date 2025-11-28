import { defineStore } from "pinia";
import { ref } from "vue";

import type { RAGAnswerMode, RAGRetrievalMode } from "../api/types";

/**
 * Chat workspace UI state.
 *
 * Conversation history lives on the server and is read through Vue Query; this
 * store only keeps the selection the user made so navigating away and back
 * restores the same workspace.
 */
export const useChatStore = defineStore("chat", () => {
  const selectedKnowledgeBaseId = ref<string | null>(null);
  const activeConversationId = ref<string | null>(null);
  const isSidebarOpen = ref(false);
  const retrievalMode = ref<RAGRetrievalMode | "auto">("auto");
  const answerMode = ref<RAGAnswerMode>("standard");

  function setSelectedKnowledgeBase(id: string | null): void {
    selectedKnowledgeBaseId.value = id;
  }

  function setRetrievalMode(mode: RAGRetrievalMode | "auto"): void {
    retrievalMode.value = mode;
  }

  function setAnswerMode(mode: RAGAnswerMode): void {
    answerMode.value = mode;
  }

  function setActiveConversation(id: string | null): void {
    activeConversationId.value = id;
  }

  function toggleSidebar(): void {
    isSidebarOpen.value = !isSidebarOpen.value;
  }

  function closeSidebar(): void {
    isSidebarOpen.value = false;
  }

  return {
    selectedKnowledgeBaseId,
    activeConversationId,
    isSidebarOpen,
    retrievalMode,
    answerMode,
    setSelectedKnowledgeBase,
    setRetrievalMode,
    setAnswerMode,
    setActiveConversation,
    toggleSidebar,
    closeSidebar,
  };
});
