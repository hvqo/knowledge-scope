<script setup lang="ts">
import { useMutation, useQuery, useQueryClient } from "@tanstack/vue-query";
import { ElMessage } from "element-plus";
import { computed, onBeforeUnmount, ref, watch } from "vue";

import {
  appendChatMessage,
  createChatConversation,
  createChatMemory,
  createChatProject,
  deleteChatConversation,
  deleteChatMemory,
  deleteChatProject,
  extractChatMemories,
  fetchChatConversation,
  fetchChatConversations,
  fetchChatMemories,
  fetchChatProjects,
  fetchDocuments,
  fetchKnowledgeBases,
  getUserFacingError,
  streamRagQuery,
  updateChatConversation,
  updateChatProject,
} from "../api/client";
import type { RAGHistoryTurn } from "../api/client";
import type { ChatMessageView, RAGCitation, RAGStreamEvent } from "../api/types";
import MemoryDialog from "../components/chat/MemoryDialog.vue";
import ChatComposer from "../components/chat/ChatComposer.vue";
import ChatMessageList from "../components/chat/ChatMessageList.vue";
import ChatSidebar from "../components/chat/ChatSidebar.vue";
import { referencedCitations } from "../components/chat/citations";
import EvidencePanel from "../components/chat/EvidencePanel.vue";
import { useChatStore } from "../stores/chat";

const chatStore = useChatStore();
const queryClient = useQueryClient();
const pendingMessage = ref<ChatMessageView | null>(null);
const selectedMarker = ref<string | null>(null);
const isSourceOpen = ref(false);
const drawerCitations = ref<RAGCitation[]>([]);
const rewrittenQuery = ref<string | null>(null);
const retrievalEscalated = ref(false);
const isStreaming = ref(false);
const activeController = ref<AbortController | null>(null);
let activeRun = 0;
let isUnmounting = false;
let hasRestoredConversation = false;

// Answers — especially cache hits — can arrive in one burst.  Playing them
// back at a steady typewriter pace reads much better than a wall of text.
// Users who ask the OS for reduced motion get the text immediately.
const PACE_TICK_MS = 16;
const PACE_MIN_CHUNK = 3;
const PACE_CHUNK_DIVISOR = 30;
let paceTimer: number | null = null;
let paceBuffer = "";

function prefersReducedMotion(): boolean {
  return (
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

function drainPaceChunk(): void {
  if (paceBuffer === "") {
    stopPacing();
    return;
  }
  const chunkSize = Math.max(PACE_MIN_CHUNK, Math.ceil(paceBuffer.length / PACE_CHUNK_DIVISOR));
  const chunk = paceBuffer.slice(0, chunkSize);
  paceBuffer = paceBuffer.slice(chunkSize);
  const current = pendingMessage.value;
  if (current !== null && current.status === "streaming") {
    pendingMessage.value = { ...current, content: current.content + chunk };
  }
  if (paceBuffer === "") {
    stopPacing();
  }
}

function stopPacing(): void {
  if (paceTimer !== null) {
    window.clearInterval(paceTimer);
    paceTimer = null;
  }
}

function queueAnswerDelta(text: string): void {
  if (prefersReducedMotion()) {
    const current = pendingMessage.value;
    if (current !== null && current.status === "streaming") {
      pendingMessage.value = { ...current, content: current.content + text };
    }
    return;
  }
  paceBuffer += text;
  if (paceTimer === null) {
    paceTimer = window.setInterval(drainPaceChunk, PACE_TICK_MS);
  }
}

function flushPacedText(): void {
  stopPacing();
  if (paceBuffer === "") {
    return;
  }
  const rest = paceBuffer;
  paceBuffer = "";
  const current = pendingMessage.value;
  if (current !== null && current.status === "streaming") {
    pendingMessage.value = { ...current, content: current.content + rest };
  }
}

async function waitForPacedText(): Promise<void> {
  while (paceBuffer !== "" && paceTimer !== null) {
    await new Promise((resolve) => window.setTimeout(resolve, PACE_TICK_MS));
  }
  flushPacedText();
}

const knowledgeBasesQuery = useQuery({
  queryKey: ["chat-knowledge-bases"],
  queryFn: () => fetchKnowledgeBases({ limit: 100 }),
  staleTime: 60_000,
});

const conversationsQuery = useQuery({
  queryKey: ["chat-conversations"],
  queryFn: () => fetchChatConversations({ limit: 100 }),
  staleTime: 15_000,
});

const conversationDetailQuery = useQuery({
  queryKey: computed(() => ["chat-conversation", chatStore.activeConversationId]),
  queryFn: () => fetchChatConversation(chatStore.activeConversationId ?? ""),
  enabled: computed(() => chatStore.activeConversationId !== null),
  staleTime: 10_000,
});

const knowledgeBases = computed(() => knowledgeBasesQuery.data.value?.items ?? []);
const knowledgeBasesLoading = computed(() => knowledgeBasesQuery.isPending.value);
const knowledgeBasesError = computed(() => knowledgeBasesQuery.isError.value);
const knowledgeBasesErrorMessage = computed(() =>
  getUserFacingError(knowledgeBasesQuery.error.value, "知识库暂时无法加载，请稍后重试。"),
);

const conversations = computed(() => conversationsQuery.data.value?.items ?? []);

const projectsQuery = useQuery({
  queryKey: ["chat-projects"],
  queryFn: fetchChatProjects,
});
const projects = computed(() => projectsQuery.data.value ?? []);

function invalidateWorkspace(): void {
  void queryClient.invalidateQueries({ queryKey: ["chat-projects"] });
  void queryClient.invalidateQueries({ queryKey: ["chat-conversations"] });
  void queryClient.invalidateQueries({ queryKey: ["chat-memories"] });
}

const createProjectMutation = useMutation({
  mutationFn: (title: string) => createChatProject({ title }),
  onSuccess: () => invalidateWorkspace(),
  onError: (error) => {
    ElMessage.error(getUserFacingError(error, "项目创建失败，请稍后重试。"));
  },
});

const deleteProjectMutation = useMutation({
  mutationFn: (projectId: string) => deleteChatProject(projectId),
  onSuccess: () => invalidateWorkspace(),
  onError: (error) => {
    ElMessage.error(getUserFacingError(error, "项目删除失败，请稍后重试。"));
  },
});

const renameProjectMutation = useMutation({
  mutationFn: ({ id, title }: { id: string; title: string }) =>
    updateChatProject(id, { title }),
  onSuccess: () => invalidateWorkspace(),
  onError: (error) => {
    ElMessage.error(getUserFacingError(error, "项目重命名失败，请稍后重试。"));
  },
});

const moveConversationMutation = useMutation({
  mutationFn: ({ id, projectId }: { id: string; projectId: string | null }) =>
    updateChatConversation(id, { project_id: projectId }),
  onSuccess: () => invalidateWorkspace(),
  onError: (error) => {
    ElMessage.error(getUserFacingError(error, "移动失败，请稍后重试。"));
  },
});

const rememberMutation = useMutation({
  mutationFn: (content: string) => {
    const knowledgeBaseId = selectedKnowledgeBaseId.value ?? "";
    return createChatMemory({ knowledge_base_id: knowledgeBaseId, content });
  },
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: ["chat-memories"] });
  },
  onError: (error) => {
    ElMessage.error(getUserFacingError(error, "记忆保存失败，请稍后重试。"));
  },
});

const forgetMutation = useMutation({
  mutationFn: (memoryId: string) => deleteChatMemory(memoryId),
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: ["chat-memories"] });
  },
  onError: (error) => {
    ElMessage.error(getUserFacingError(error, "记忆删除失败，请稍后重试。"));
  },
});

function extractMemoriesQuietly(conversationId: string): void {
  // Fire and forget: distilling memories must never disturb the conversation.
  void extractChatMemories(conversationId)
    .then(() => {
      if (selectedKnowledgeBaseId.value !== null) {
        void queryClient.invalidateQueries({ queryKey: ["chat-memories"] });
      }
    })
    .catch(() => {
      // The backend already tolerates extraction failures.
    });
}
const conversationsLoading = computed(() => conversationsQuery.isPending.value);
const conversationsError = computed(() => conversationsQuery.isError.value);
const conversationsErrorMessage = computed(() =>
  getUserFacingError(conversationsQuery.error.value, "历史对话暂时无法加载，请稍后重试。"),
);

const activeConversationSummary = computed(() => {
  const id = chatStore.activeConversationId;
  if (id === null) {
    return null;
  }
  return (
    conversationDetailQuery.data.value?.conversation ??
    conversations.value.find((item) => item.id === id) ??
    null
  );
});

const selectedKnowledgeBaseId = computed(
  () => activeConversationSummary.value?.knowledge_base_id ?? chatStore.selectedKnowledgeBaseId,
);
const canSend = computed(() => selectedKnowledgeBaseId.value !== null);
const selectedKnowledgeBaseName = computed(
  () => knowledgeBases.value.find((item) => item.id === selectedKnowledgeBaseId.value)?.name ?? null,
);

const memoriesQuery = useQuery({
  queryKey: computed(() => ["chat-memories", selectedKnowledgeBaseId.value]),
  queryFn: () => fetchChatMemories(selectedKnowledgeBaseId.value ?? ""),
  enabled: computed(() => selectedKnowledgeBaseId.value !== null),
});
const memories = computed(() => memoriesQuery.data.value ?? []);
const isMemoryOpen = ref(false);


const documentsQuery = useQuery({
  queryKey: computed(() => ["chat-documents", selectedKnowledgeBaseId.value]),
  queryFn: () => fetchDocuments(selectedKnowledgeBaseId.value ?? "", { limit: 100 }),
  enabled: computed(() => selectedKnowledgeBaseId.value !== null),
  staleTime: 120_000,
});

const documentTitles = computed<Record<string, string>>(() =>
  Object.fromEntries(
    (documentsQuery.data.value?.items ?? []).map((document) => [
      document.id,
      document.original_filename,
    ]),
  ),
);

const HISTORY_MAX_MESSAGES = 200;
const HISTORY_MESSAGE_MAX_CHARACTERS = 4_000;
// The server compresses whatever exceeds its recent-turn window, so the client
// only guards the transport budget.
const HISTORY_TOTAL_CHARACTERS = 60_000;

/** Every turn, within the transport budget; the backend compresses the rest. */
function buildHistory(messages: ChatMessageView[]): RAGHistoryTurn[] {
  const history: RAGHistoryTurn[] = [];
  let total = 0;
  for (const message of [...messages].reverse()) {
    if (history.length >= HISTORY_MAX_MESSAGES || total >= HISTORY_TOTAL_CHARACTERS) {
      break;
    }
    const content = message.content.trim().slice(0, HISTORY_MESSAGE_MAX_CHARACTERS);
    if (content === "") {
      continue;
    }
    history.unshift({ role: message.role, content });
    total += content.length;
  }
  return history;
}

const persistedMessages = computed(() => conversationDetailQuery.data.value?.messages ?? []);
const messages = computed<ChatMessageView[]>(() =>
  pendingMessage.value === null
    ? [...persistedMessages.value]
    : [...persistedMessages.value, pendingMessage.value],
);
const lastAssistantCitations = computed<RAGCitation[]>(() => {
  const assistantMessages = messages.value.filter((message) => message.role === "assistant");
  const lastMessage = assistantMessages.at(-1);
  return lastMessage === undefined
    ? []
    : referencedCitations(lastMessage.content, lastMessage.citations);
});
const sourceCount = computed(() => lastAssistantCitations.value.length);

const isBusy = computed(() => isStreaming.value);

const renameMutation = useMutation({
  mutationFn: (variables: { id: string; title: string }) =>
    updateChatConversation(variables.id, { title: variables.title }),
  onSuccess: (_conversation, variables) => {
    void queryClient.invalidateQueries({ queryKey: ["chat-conversations"] });
    void queryClient.invalidateQueries({ queryKey: ["chat-conversation", variables.id] });
  },
});

const rebindMutation = useMutation({
  mutationFn: (variables: { id: string; knowledgeBaseId: string | null }) =>
    updateChatConversation(variables.id, { knowledge_base_id: variables.knowledgeBaseId }),
  onSuccess: (_conversation, variables) => {
    void queryClient.invalidateQueries({ queryKey: ["chat-conversations"] });
    void queryClient.invalidateQueries({ queryKey: ["chat-conversation", variables.id] });
  },
});

const deleteMutation = useMutation({
  mutationFn: (id: string) => deleteChatConversation(id),
  onSuccess: (_result, id) => {
    if (chatStore.activeConversationId === id) {
      chatStore.setActiveConversation(null);
      pendingMessage.value = null;
      closeSources();
    }
    void queryClient.invalidateQueries({ queryKey: ["chat-conversations"] });
  },
});

watch(
  knowledgeBases,
  (items) => {
    if (items.length === 0) {
      chatStore.setSelectedKnowledgeBase(null);
      return;
    }
    const current = chatStore.selectedKnowledgeBaseId;
    if (current === null || !items.some((item) => item.id === current)) {
      chatStore.setSelectedKnowledgeBase(items[0].id);
    }
  },
  { immediate: true },
);

watch(
  conversations,
  (items) => {
    if (hasRestoredConversation || items.length === 0 || chatStore.activeConversationId !== null) {
      return;
    }
    hasRestoredConversation = true;
    chatStore.setActiveConversation(items[0].id);
  },
  { immediate: true },
);

function closeSources(): void {
  isSourceOpen.value = false;
  drawerCitations.value = [];
  selectedMarker.value = null;
}

function startNewConversation(): void {
  if (isBusy.value) {
    return;
  }
  hasRestoredConversation = true;
  chatStore.setActiveConversation(null);
  pendingMessage.value = null;
  closeSources();
  chatStore.closeSidebar();
}

function selectConversation(id: string): void {
  if (isBusy.value) {
    return;
  }
  chatStore.setActiveConversation(id);
  pendingMessage.value = null;
  closeSources();
  chatStore.closeSidebar();
}

function changeKnowledgeBase(id: string | null): void {
  const conversationId = chatStore.activeConversationId;
  if (conversationId === null) {
    chatStore.setSelectedKnowledgeBase(id);
    return;
  }
  rebindMutation.mutate(
    { id: conversationId, knowledgeBaseId: id },
    {
      onError: (error) => {
        ElMessage.error(getUserFacingError(error, "知识库切换失败，请稍后重试。"));
      },
    },
  );
}

function renameConversation(id: string, title: string): void {
  renameMutation.mutate(
    { id, title },
    {
      onError: (error) => {
        ElMessage.error(getUserFacingError(error, "重命名失败，请稍后重试。"));
      },
    },
  );
}

function deleteConversation(id: string): void {
  deleteMutation.mutate(id, {
    onError: (error) => {
      ElMessage.error(getUserFacingError(error, "删除失败，请稍后重试。"));
    },
  });
}

async function ensureConversation(knowledgeBaseId: string): Promise<string> {
  const existing = chatStore.activeConversationId;
  if (existing !== null) {
    return existing;
  }
  const created = await createChatConversation({ knowledge_base_id: knowledgeBaseId });
  chatStore.setActiveConversation(created.id);
  hasRestoredConversation = true;
  await queryClient.invalidateQueries({ queryKey: ["chat-conversations"] });
  return created.id;
}

function streamFailureMessage(category: string): string {
  if (category === "retrieval") {
    return "暂时无法读取当前知识库，请稍后重试。";
  }
  if (category === "empty_output") {
    return "模型未返回回答内容，请稍后重试。";
  }
  if (category === "truncated_output") {
    return "回答生成达到长度限制，请稍后重试。";
  }
  if (category === "request") {
    return "问题格式不正确，请调整后重试。";
  }
  if (category === "timeout") {
    return "回答生成超时，请稍后重试。";
  }
  return "回答生成未完成，请稍后重试。";
}

function applyStreamEvent(event: RAGStreamEvent): void {
  const current = pendingMessage.value;
  if (current === null) {
    return;
  }
  if (event.event === "answer_delta") {
    queueAnswerDelta(event.data.text);
    return;
  }
  if (event.event === "citations") {
    pendingMessage.value = { ...current, citations: event.data.items };
    selectedMarker.value = event.data.items[0]?.marker ?? null;
    return;
  }
  if (event.event === "error") {
    flushPacedText();
    const flushed = pendingMessage.value ?? current;
    pendingMessage.value = {
      ...flushed,
      status: "error",
      content: flushed.content.trim() || streamFailureMessage(event.data.category),
    };
    return;
  }
  if (event.event === "complete") {
    rewrittenQuery.value = event.data.rewritten_query;
    retrievalEscalated.value = event.data.retrieval_escalated === true;
  }
  if (event.data.status === "error") {
    flushPacedText();
    const flushed = pendingMessage.value ?? current;
    pendingMessage.value = {
      ...flushed,
      status: "error",
      content: flushed.content.trim() || "回答生成未完成，请稍后重试。",
    };
  }
}

async function persistAssistantMessage(
  conversationId: string,
  message: ChatMessageView,
): Promise<void> {
  try {
    await appendChatMessage(conversationId, {
      role: "assistant",
      content: message.content,
      status: message.status === "error" ? "error" : "complete",
      citations: message.citations,
    });
  } catch {
    // Keep the answer visible even when it could not be stored.
    ElMessage.warning("回答未能保存到历史记录");
  } finally {
    pendingMessage.value = null;
    await queryClient.invalidateQueries({ queryKey: ["chat-conversation", conversationId] });
    await queryClient.invalidateQueries({ queryKey: ["chat-conversations"] });
  }
}

async function sendQuestion(question: string): Promise<void> {
  const knowledgeBaseId = selectedKnowledgeBaseId.value;
  if (question.trim() === "" || isStreaming.value) {
    return;
  }
  if (knowledgeBaseId === null) {
    ElMessage.warning("请先选择提问依据的知识库");
    return;
  }

  let conversationId: string;
  const history = buildHistory(persistedMessages.value);
  try {
    conversationId = await ensureConversation(knowledgeBaseId);
    await appendChatMessage(conversationId, { role: "user", content: question });
    await queryClient.invalidateQueries({ queryKey: ["chat-conversation", conversationId] });
    await queryClient.invalidateQueries({ queryKey: ["chat-conversations"] });
  } catch (error) {
    ElMessage.error(getUserFacingError(error, "问题没有保存成功，请稍后重试。"));
    return;
  }

  const run = ++activeRun;
  flushPacedText();
  paceBuffer = "";
  isStreaming.value = true;
  selectedMarker.value = null;
  pendingMessage.value = {
    id: `pending-${run}`,
    role: "assistant",
    status: "streaming",
    content: "",
    citations: [],
    position: -1,
    created_at: new Date().toISOString(),
  };
  const controller = new AbortController();
  activeController.value = controller;
  let receivedComplete = false;

  try {
    const stream = streamRagQuery(
      {
        query: question,
        knowledge_base_id: knowledgeBaseId,
        retrieval_mode: chatStore.retrievalMode,
        history,
      },
      controller.signal,
    );
      for await (const event of stream) {
      if (run !== activeRun || isUnmounting) {
        return;
      }
      if (event.event === "complete") {
        receivedComplete = true;
      }
      applyStreamEvent(event);
    }
    if (run !== activeRun) {
      return;
    }
    // The playback pacer may still be draining the received text.
    await waitForPacedText();
    const current = pendingMessage.value;
    if (current !== null) {
      pendingMessage.value = {
        ...current,
        status: current.status === "error" ? "error" : "complete",
        content:
          current.content.trim() ||
          (receivedComplete ? "没有收到可展示的回答。" : "连接已中断，回答没有完成，请稍后重试。"),
      };
      if (!receivedComplete && current.content.trim() === "") {
        pendingMessage.value = { ...pendingMessage.value, status: "error" };
      }
    }
  } catch (error) {
    if (run !== activeRun || isUnmounting || controller.signal.aborted) {
      return;
    }
    const current = pendingMessage.value;
    if (current !== null) {
      pendingMessage.value = {
        ...current,
        status: "error",
        content: current.content.trim() || getUserFacingError(error, "回答暂时无法生成，请稍后重试。"),
      };
    }
  } finally {
    if (run === activeRun) {
      isStreaming.value = false;
      activeController.value = null;
    }
  }

  const finished = pendingMessage.value;
  if (run === activeRun && finished !== null && !isUnmounting) {
    await persistAssistantMessage(conversationId, finished);
    extractMemoriesQuietly(conversationId);
  }
}

function stopGeneration(): void {
  const controller = activeController.value;
  if (controller === null) {
    return;
  }
  activeRun += 1;
  controller.abort();
  activeController.value = null;
  isStreaming.value = false;
  flushPacedText();
  const current = pendingMessage.value;
  if (current === null) {
    return;
  }
  const hasContent = current.content.trim() !== "";
  const stopped: ChatMessageView = {
    ...current,
    status: hasContent ? "complete" : "error",
    content: hasContent ? current.content : "已停止生成。",
  };
  pendingMessage.value = stopped;
  const conversationId = chatStore.activeConversationId;
  if (conversationId !== null) {
    void persistAssistantMessage(conversationId, stopped);
  }
}

function selectCitation(marker: string, citations: RAGCitation[]): void {
  if (!citations.some((citation) => citation.marker === marker)) {
    return;
  }
  drawerCitations.value = citations;
  selectedMarker.value = marker;
  isSourceOpen.value = true;
}

function selectCitationInDrawer(marker: string): void {
  selectedMarker.value = marker;
}

function openSources(citations: RAGCitation[]): void {
  if (citations.length === 0) {
    return;
  }
  drawerCitations.value = citations;
  isSourceOpen.value = true;
}

function openLastSources(): void {
  openSources(lastAssistantCitations.value);
}

function retryConversations(): void {
  void conversationsQuery.refetch();
}

function retryKnowledgeBases(): void {
  void knowledgeBasesQuery.refetch();
}

function applyPrompt(prompt: string): void {
  void sendQuestion(prompt);
}

onBeforeUnmount(() => {
  isUnmounting = true;
  activeRun += 1;
  activeController.value?.abort();
  stopPacing();
});
</script>

<template>
  <section class="chat-page">
    <header class="chat-header">
      <button
        class="sidebar-toggle"
        type="button"
        aria-label="展开或收起对话列表"
        @click="chatStore.toggleSidebar()"
      >
        ☰
      </button>
      <div class="chat-header__title">
        <h1>{{ activeConversationSummary?.title ?? "新对话" }}</h1>
      </div>
      <div class="chat-header__actions">
        <button
          v-if="sourceCount > 0"
          type="button"
          class="sources-toggle"
          :aria-expanded="isSourceOpen"
          @click="isSourceOpen ? (isSourceOpen = false) : openLastSources()"
        >
          来源 {{ sourceCount }}
        </button>
        <span
          v-if="isStreaming"
          class="streaming-badge"
          role="status"
        >
          正在生成
        </span>
      </div>
    </header>

    <div class="chat-body">
      <ChatSidebar
        :conversations="conversations"
        :projects="projects"
        :memory-count="memories.length"
        :active-conversation-id="chatStore.activeConversationId"
        :conversations-loading="conversationsLoading"
        :conversations-error="conversationsError"
        :conversations-error-message="conversationsErrorMessage"
        :knowledge-bases="knowledgeBases"
        :selected-knowledge-base-id="selectedKnowledgeBaseId"
        :knowledge-bases-loading="knowledgeBasesLoading"
        :knowledge-bases-error="knowledgeBasesError"
        :knowledge-bases-error-message="knowledgeBasesErrorMessage"
        :is-busy="isBusy"
        :is-open="chatStore.isSidebarOpen"
        @new-conversation="startNewConversation"
        @select-conversation="selectConversation"
        @select-knowledge-base="changeKnowledgeBase"
        @rename-conversation="renameConversation"
        @delete-conversation="deleteConversation"
        @move-conversation="(id, projectId) => moveConversationMutation.mutate({ id, projectId })"
        @create-project="(title) => createProjectMutation.mutate(title)"
        @delete-project="(id) => deleteProjectMutation.mutate(id)"
        @rename-project="(id, title) => renameProjectMutation.mutate({ id, title })"
        @open-memories="isMemoryOpen = true"
        @retry-conversations="retryConversations"
        @retry-knowledge-bases="retryKnowledgeBases"
        @close="chatStore.closeSidebar()"
      />

      <MemoryDialog
        :open="isMemoryOpen"
        :memories="memories"
        :knowledge-base-name="selectedKnowledgeBaseName"
        :loading="memoriesQuery.isPending.value"
        @close="isMemoryOpen = false"
        @remember="(content) => rememberMutation.mutate(content)"
        @forget="(id) => forgetMutation.mutate(id)"
      />

      <main class="chat-main">
        <ChatMessageList
          :messages="messages"
          :is-streaming="isStreaming"
          :is-ready="canSend"
          @select-citation="selectCitation"
          @select-prompt="applyPrompt"
          @open-sources="openSources"
        />

        <div
          v-if="!canSend && !knowledgeBasesLoading"
          class="chat-hint"
          role="status"
        >
          <span aria-hidden="true">↳</span>
          先在左侧选择一个知识库，回答只会依据其中的资料生成。
        </div>

        <ChatComposer
          :disabled="isStreaming"
          :can-send="canSend"
          :retrieval-mode="chatStore.retrievalMode"
          @send="sendQuestion"
          @stop="stopGeneration"
          @update:retrieval-mode="chatStore.setRetrievalMode"
        />
      </main>

      <aside
        v-if="isSourceOpen"
        class="source-drawer"
        aria-label="回答来源"
      >
        <button
          type="button"
          class="source-drawer__close"
          aria-label="关闭来源面板"
          @click="closeSources"
        >
          ✕
        </button>
        <EvidencePanel
          :citations="drawerCitations"
          :selected-marker="selectedMarker"
          :document-titles="documentTitles"
          :rewritten-query="rewrittenQuery"
          :escalated="retrievalEscalated"
          @select-citation="selectCitationInDrawer"
        />
      </aside>
    </div>
  </section>
</template>

<style scoped>
.chat-page {
  display: flex;
  height: calc(100vh - 104px);
  flex-direction: column;
  overflow: hidden;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-lg);
}

.chat-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 20px;
  border-bottom: 1px solid var(--ks-border);
}

.sidebar-toggle {
  display: none;
  width: 34px;
  height: 34px;
  place-items: center;
  color: var(--ks-muted);
  font-size: 15px;
  background: transparent;
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-sm);
  cursor: pointer;
}

.chat-header__title {
  min-width: 0;
  flex: 1;
}

.chat-header__title h1 {
  margin: 0;
  overflow: hidden;
  color: var(--ks-ink);
  font-size: 16px;
  font-weight: 700;
  text-overflow: ellipsis;
  white-space: nowrap;
}


.chat-header__actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.sources-toggle {
  padding: 6px 12px;
  color: var(--ks-accent-strong);
  font-size: 12px;
  font-weight: 650;
  background: var(--ks-accent-soft);
  border: 1px solid transparent;
  border-radius: 999px;
  cursor: pointer;
}

.sources-toggle:hover {
  border-color: var(--ks-accent);
}

.streaming-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--ks-muted);
  font-size: 12px;
}

.streaming-badge::before {
  width: 6px;
  height: 6px;
  background: var(--ks-accent);
  border-radius: 50%;
  content: "";
  animation: badge-pulse 1.2s ease-in-out infinite;
}

.chat-body {
  position: relative;
  display: flex;
  min-height: 0;
  flex: 1;
}

.chat-main {
  display: flex;
  min-width: 0;
  flex: 1;
  flex-direction: column;
}

.chat-hint {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 24px 10px;
  color: var(--ks-muted);
  font-size: 12px;
}

.source-drawer {
  position: relative;
  display: flex;
  min-width: 320px;
  border-left: 1px solid var(--ks-border);
}

.source-drawer__close {
  position: absolute;
  top: 14px;
  right: 14px;
  z-index: 2;
  display: grid;
  width: 26px;
  height: 26px;
  place-items: center;
  color: var(--ks-muted);
  font-size: 12px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-sm);
  cursor: pointer;
}

.source-drawer__close:hover {
  color: var(--ks-ink);
}

:deep(.evidence-panel) {
  width: 100%;
  border-left: 0;
}

@keyframes badge-pulse {
  0%,
  100% {
    opacity: 0.4;
  }

  50% {
    opacity: 1;
  }
}

@media (max-width: 960px) {
  .sidebar-toggle {
    display: grid;
  }

  .source-drawer {
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    z-index: 20;
    min-width: 0;
    width: min(340px, 88%);
    background: var(--ks-surface-subtle);
    box-shadow: -12px 0 32px rgb(32 38 34 / 12%);
  }
}

@media (max-width: 720px) {
  .chat-page {
    height: calc(100vh - 72px);
  }
}
</style>
