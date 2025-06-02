<script setup lang="ts">
import { useQuery } from "@tanstack/vue-query";
import { computed, nextTick, onBeforeUnmount, ref, watch } from "vue";

import {
  fetchDocuments,
  fetchKnowledgeBases,
  getUserFacingError,
  streamRagQuery,
} from "../api/client";
import type { Document, RAGCitation, RAGStreamEvent } from "../api/types";
import ChatComposer from "../components/chat/ChatComposer.vue";
import ChatMessageList from "../components/chat/ChatMessageList.vue";
import ConversationSidebar from "../components/chat/ConversationSidebar.vue";
import EvidencePanel from "../components/chat/EvidencePanel.vue";
import { useChatStore } from "../stores/chat";

const chatStore = useChatStore();
const messageListRef = ref<{ scrollToBottom: () => void } | null>(null);
const selectedMarker = ref<string | null>(null);
const isStreaming = ref(false);
const activeController = ref<AbortController | null>(null);
let activeRun = 0;
let isUnmounting = false;

const knowledgeBasesQuery = useQuery({
  queryKey: ["chat-knowledge-bases"],
  queryFn: () => fetchKnowledgeBases({ limit: 100 }),
  staleTime: 30_000,
});

const documentsQuery = useQuery({
  queryKey: computed(() => ["chat-documents", chatStore.selectedKnowledgeBaseId]),
  queryFn: () => fetchDocuments(chatStore.selectedKnowledgeBaseId ?? "", { limit: 100 }),
  enabled: computed(() => chatStore.selectedKnowledgeBaseId !== null),
  staleTime: 30_000,
});

const knowledgeBases = computed(() => knowledgeBasesQuery.data.value?.items ?? []);
const knowledgeBasesLoading = computed(() => knowledgeBasesQuery.isPending.value);
const knowledgeBasesError = computed(() => knowledgeBasesQuery.isError.value);
const knowledgeBasesErrorMessage = computed(() =>
  getUserFacingError(knowledgeBasesQuery.error.value, "知识库暂时无法加载，请稍后重试。"),
);
const activeConversation = computed(() => chatStore.activeConversation);
const messages = computed(() => activeConversation.value?.messages ?? []);
const selectedKnowledgeBase = computed(() =>
  knowledgeBases.value.find((item) => item.id === chatStore.selectedKnowledgeBaseId) ?? null,
);
const documentTitles = computed<Record<string, string>>(() => {
  const items: Document[] = documentsQuery.data.value?.items ?? [];
  return Object.fromEntries(items.map((document) => [document.id, document.original_filename]));
});
const visibleCitations = computed<RAGCitation[]>(() => {
  const assistantMessages = messages.value.filter((message) => message.role === "assistant");
  return assistantMessages.at(-1)?.citations ?? [];
});
const canSend = computed(
  () => chatStore.selectedKnowledgeBaseId !== null && !knowledgeBasesLoading.value,
);
const messageSignature = computed(() =>
  messages.value.map((message) => `${message.id}:${message.content.length}`).join("|")
);

watch(
  knowledgeBases,
  (items) => {
    const selectedId = chatStore.selectedKnowledgeBaseId;
    if (items.length === 0) {
      chatStore.setSelectedKnowledgeBase(null);
      return;
    }
    if (selectedId === null || !items.some((item) => item.id === selectedId)) {
      chatStore.setSelectedKnowledgeBase(items[0].id);
    }
  },
  { immediate: true },
);

watch(messageSignature, () => {
  void nextTick(() => messageListRef.value?.scrollToBottom());
});

function selectKnowledgeBase(id: string | null): void {
  if (isStreaming.value) {
    return;
  }
  const hadMessages = (activeConversation.value?.messages.length ?? 0) > 0;
  const changed = chatStore.selectedKnowledgeBaseId !== id;
  chatStore.setSelectedKnowledgeBase(id);
  if (hadMessages && changed) {
    chatStore.newConversation(id);
  }
  selectedMarker.value = null;
}

function newConversation(): void {
  if (isStreaming.value) {
    return;
  }
  chatStore.newConversation();
  selectedMarker.value = null;
}

function selectConversation(id: string): void {
  if (isStreaming.value) {
    return;
  }
  chatStore.selectConversation(id);
  selectedMarker.value = null;
}

function setAssistantStatus(
  conversationId: string,
  assistantId: string,
  status: "complete" | "streaming" | "error",
): void {
  chatStore.updateMessage(conversationId, assistantId, (message) => {
    message.status = status;
  });
}

function updateAssistantContent(
  conversationId: string,
  assistantId: string,
  content: string,
): void {
  chatStore.updateMessage(conversationId, assistantId, (message) => {
    message.content += content;
  });
}

function updateAssistantCitations(
  conversationId: string,
  assistantId: string,
  citations: RAGCitation[],
): void {
  chatStore.updateMessage(conversationId, assistantId, (message) => {
    message.citations = citations;
  });
  selectedMarker.value = citations[0]?.marker ?? null;
}

function streamFailureMessage(category: string): string {
  if (category === "retrieval") {
    return "暂时无法读取当前知识库，请稍后重试。";
  }
  if (category === "request") {
    return "问题格式不正确，请调整后重试。";
  }
  if (category === "timeout") {
    return "回答生成超时，请稍后重试。";
  }
  return "回答生成未完成，请稍后重试。";
}

function handleStreamEvent(
  event: RAGStreamEvent,
  conversationId: string,
  assistantId: string,
): void {
  if (event.event === "answer_delta") {
    updateAssistantContent(conversationId, assistantId, event.data.text);
  } else if (event.event === "citations") {
    updateAssistantCitations(conversationId, assistantId, event.data.items);
  } else if (event.event === "error") {
    setAssistantStatus(conversationId, assistantId, "error");
    chatStore.updateMessage(conversationId, assistantId, (message) => {
      if (!message.content.trim()) {
        message.content = streamFailureMessage(event.data.category);
      }
    });
  } else {
    const complete = event.data;
    setAssistantStatus(
      conversationId,
      assistantId,
      complete.status === "error" ? "error" : "complete",
    );
    if (complete.status === "error") {
      chatStore.updateMessage(conversationId, assistantId, (message) => {
        if (!message.content.trim()) {
          message.content = "回答生成未完成，请稍后重试。";
        }
      });
    }
  }
}

async function sendQuestion(query: string): Promise<void> {
  const knowledgeBaseId = chatStore.selectedKnowledgeBaseId;
  if (!query.trim() || knowledgeBaseId === null || isStreaming.value) {
    return;
  }

  const conversation = chatStore.ensureActiveConversation();
  conversation.knowledgeBaseId = knowledgeBaseId;
  const conversationId = conversation.id;
  chatStore.addMessage(conversationId, "user", query, "complete");
  const assistantId = chatStore.addMessage(conversationId, "assistant", "", "streaming");
  selectedMarker.value = null;
  isStreaming.value = true;
  const controller = new AbortController();
  activeController.value = controller;
  const run = ++activeRun;
  let receivedComplete = false;

  try {
    const stream = streamRagQuery(
      { query, knowledge_base_id: knowledgeBaseId, retrieval_mode: "dense" },
      controller.signal,
    );
    for await (const event of stream) {
      if (run !== activeRun || isUnmounting) {
        return;
      }
      if (event.event === "complete") {
        receivedComplete = true;
      }
      handleStreamEvent(event, conversationId, assistantId);
    }
    if (run === activeRun) {
      chatStore.updateMessage(conversationId, assistantId, (message) => {
        if (!receivedComplete) {
          message.status = "error";
          if (!message.content.trim()) {
            message.content = "连接已中断，回答没有完成，请稍后重试。";
          }
        } else if (message.status === "streaming") {
          message.status = "complete";
        }
        if (!message.content.trim()) {
          message.content = "没有收到可展示的回答。";
        }
      });
    }
  } catch (error) {
    if (run !== activeRun || isUnmounting || controller.signal.aborted) {
      return;
    }
    setAssistantStatus(conversationId, assistantId, "error");
    chatStore.updateMessage(conversationId, assistantId, (message) => {
      if (!message.content.trim()) {
        message.content = getUserFacingError(error, "回答暂时无法生成，请稍后重试。");
      }
    });
  } finally {
    if (run === activeRun) {
      isStreaming.value = false;
      activeController.value = null;
    }
  }
}

function stopGeneration(): void {
  const controller = activeController.value;
  if (controller === null) {
    return;
  }
  activeRun += 1;
  controller.abort();
  const conversation = activeConversation.value;
  const assistant = conversation?.messages.at(-1);
  if (conversation !== null && assistant?.role === "assistant") {
    setAssistantStatus(conversation.id, assistant.id, "error");
    chatStore.updateMessage(conversation.id, assistant.id, (message) => {
      if (!message.content.trim()) {
        message.content = "已停止生成。";
      }
    });
  }
  activeController.value = null;
  isStreaming.value = false;
}

function selectCitation(marker: string): void {
  if (visibleCitations.value.some((citation) => citation.marker === marker)) {
    selectedMarker.value = marker;
  }
}

function retryKnowledgeBases(): void {
  void knowledgeBasesQuery.refetch();
}

onBeforeUnmount(() => {
  isUnmounting = true;
  activeRun += 1;
  activeController.value?.abort();
});
</script>

<template>
  <section class="chat-page">
    <div class="chat-page__heading">
      <div>
        <span class="eyebrow">知识工作区</span>
        <h1>AI 问答</h1>
      </div>
      <div class="chat-page__context">
        <span
          class="context-dot"
          aria-hidden="true"
        />
        <span v-if="selectedKnowledgeBase">正在使用 {{ selectedKnowledgeBase.name }}</span>
        <span v-else>请选择知识库开始提问</span>
      </div>
    </div>

    <div
      v-if="knowledgeBasesError"
      class="chat-page__notice chat-page__notice--error"
      role="alert"
    >
      {{ knowledgeBasesErrorMessage }}
      <button
        type="button"
        @click="retryKnowledgeBases"
      >
        重试
      </button>
    </div>

    <div class="chat-workspace">
      <ConversationSidebar
        :conversations="chatStore.conversations"
        :active-conversation-id="chatStore.activeConversationId"
        :knowledge-bases="knowledgeBases"
        :selected-knowledge-base-id="chatStore.selectedKnowledgeBaseId"
        :knowledge-bases-loading="knowledgeBasesLoading"
        :knowledge-bases-error="knowledgeBasesError"
        :knowledge-bases-error-message="knowledgeBasesErrorMessage"
        @new-conversation="newConversation"
        @select-conversation="selectConversation"
        @select-knowledge-base="selectKnowledgeBase"
        @retry-knowledge-bases="retryKnowledgeBases"
      />

      <main class="chat-main">
        <header class="chat-main__header">
          <div>
            <h2>{{ activeConversation?.title || "新对话" }}</h2>
            <span>基于当前知识库的资料回答</span>
          </div>
          <span class="retrieval-badge">文本检索</span>
        </header>
        <ChatMessageList
          ref="messageListRef"
          :messages="messages"
          :is-streaming="isStreaming"
          @select-citation="selectCitation"
        />
        <div
          v-if="!canSend && !knowledgeBasesLoading"
          class="chat-main__hint"
          role="status"
        >
          <span aria-hidden="true">↳</span>
          请选择一个知识库，回答将只依据其中的资料生成。
        </div>
        <ChatComposer
          :disabled="isStreaming"
          :can-send="canSend"
          @send="sendQuestion"
          @stop="stopGeneration"
        />
      </main>

      <EvidencePanel
        :citations="visibleCitations"
        :selected-marker="selectedMarker"
        :document-titles="documentTitles"
        @select-citation="selectCitation"
      />
    </div>
  </section>
</template>

<style scoped>
.chat-page {
  display: flex;
  min-height: calc(100vh - 104px);
  flex-direction: column;
  gap: 18px;
}

.chat-page__heading {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  min-height: 48px;
  gap: 20px;
}

.eyebrow {
  color: var(--ks-accent);
  font-size: 11px;
  font-weight: 720;
  letter-spacing: 0.08em;
}

h1 {
  margin: 4px 0 0;
  color: var(--ks-ink);
  font-size: clamp(25px, 3vw, 32px);
  letter-spacing: -0.06em;
}

.chat-page__context {
  display: flex;
  align-items: center;
  gap: 7px;
  padding-bottom: 4px;
  color: var(--ks-muted);
  font-size: 12px;
}

.context-dot {
  width: 7px;
  height: 7px;
  background: var(--ks-success);
  border-radius: 50%;
  box-shadow: 0 0 0 4px rgb(63 128 99 / 10%);
}

.chat-page__notice {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 14px;
  color: var(--ks-danger);
  font-size: 12px;
  background: #fff8f7;
  border: 1px solid #f1d8d5;
  border-radius: var(--ks-radius-sm);
}

.chat-page__notice button {
  padding: 0;
  color: var(--ks-accent-strong);
  font-size: 12px;
  font-weight: 680;
  background: transparent;
  border: 0;
  cursor: pointer;
}

.chat-workspace {
  display: flex;
  min-height: 0;
  height: min(720px, calc(100vh - 190px));
  overflow: hidden;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-lg);
  box-shadow: var(--ks-shadow-sm);
}

.chat-main {
  display: flex;
  min-width: 0;
  flex: 1;
  flex-direction: column;
  background: var(--ks-surface);
}

.chat-main__header {
  display: flex;
  min-height: 68px;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 0 28px;
  border-bottom: 1px solid var(--ks-border);
}

.chat-main__header h2 {
  margin: 0 0 3px;
  overflow: hidden;
  color: var(--ks-ink);
  font-size: 14px;
  font-weight: 700;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chat-main__header span {
  color: var(--ks-muted);
  font-size: 11px;
}

.retrieval-badge {
  padding: 5px 8px;
  color: var(--ks-accent-strong) !important;
  font-size: 10px !important;
  font-weight: 680;
  background: var(--ks-accent-soft);
  border-radius: 999px;
}

.chat-main__hint {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 0 24px 10px;
  color: var(--ks-muted);
  font-size: 11px;
}

.chat-main__hint span {
  color: var(--ks-accent);
  font-size: 15px;
}

@media (max-width: 940px) {
  .chat-page__heading {
    align-items: flex-start;
    flex-direction: column;
    gap: 8px;
  }

  .chat-workspace {
    height: calc(100vh - 230px);
  }

  .evidence-panel {
    display: none;
  }
}
</style>
