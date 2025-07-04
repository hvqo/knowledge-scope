<script setup lang="ts">
import brandIcon from "../../../imgs/icon.jpg";
import { ElMessage } from "element-plus";
import { computed, nextTick, ref, watch } from "vue";

import type { ChatMessageView, RAGCitation } from "../../api/types";
import AssistantMessage from "./AssistantMessage.vue";
import { referencedCitations } from "./citations";

const props = defineProps<{
  messages: ChatMessageView[];
  isStreaming: boolean;
  isReady: boolean;
}>();

const emit = defineEmits<{
  selectCitation: [marker: string, citations: RAGCitation[]];
  selectPrompt: [text: string];
  openSources: [citations: RAGCitation[]];
}>();

const EXAMPLE_PROMPTS = [
  "这份资料的核心结论是什么？",
  "整理一下关键流程和步骤",
  "有哪些数据和指标被提到？",
] as const;

const sourcesByMessage = computed<Record<string, RAGCitation[]>>(() =>
  Object.fromEntries(
    props.messages.map((message) => [
      message.id,
      referencedCitations(message.content, message.citations),
    ]),
  ),
);

const viewport = ref<HTMLElement | null>(null);
const copiedMessageId = ref<string | null>(null);

function scrollToBottom(): void {
  const element = viewport.value;
  if (element === null) {
    return;
  }
  if (typeof element.scrollTo === "function") {
    element.scrollTo({ top: element.scrollHeight, behavior: "smooth" });
  } else {
    element.scrollTop = element.scrollHeight;
  }
}

watch(
  () => props.messages.map((message) => `${message.id}:${message.content.length}`).join("|"),
  () => {
    void nextTick(scrollToBottom);
  },
  { immediate: true },
);

async function copyMessage(message: ChatMessageView): Promise<void> {
  if (message.content.trim() === "") {
    return;
  }
  const clipboard = navigator.clipboard;
  if (clipboard === undefined || typeof clipboard.writeText !== "function") {
    ElMessage.warning("当前浏览器不支持复制");
    return;
  }
  try {
    await clipboard.writeText(message.content);
    copiedMessageId.value = message.id;
    ElMessage.success("已复制回答");
  } catch {
    ElMessage.error("复制失败，请手动选择文本");
  }
}

defineExpose({ scrollToBottom });
</script>

<template>
  <div
    ref="viewport"
    class="message-viewport"
    aria-live="polite"
    aria-label="对话内容"
  >
    <div
      v-if="messages.length === 0"
      class="message-empty"
    >
      <div
        class="message-empty__mark"
        aria-hidden="true"
      >
        ✦
      </div>
      <h2>今天想了解什么？</h2>
      <p>
        {{ isReady ? "选择一个知识库，提出关于资料的问题，回答会带上来源。" : "请先在左侧选择一个知识库。" }}
      </p>
      <div class="message-empty__prompts">
        <button
          v-for="prompt in EXAMPLE_PROMPTS"
          :key="prompt"
          type="button"
          class="prompt-card"
          :disabled="!isReady"
          @click="emit('selectPrompt', prompt)"
        >
          {{ prompt }}
        </button>
      </div>
    </div>

    <div
      v-else
      class="message-column"
    >
      <article
        v-for="message in messages"
        :key="message.id"
        class="message-row"
        :class="`message-row--${message.role}`"
      >
        <template v-if="message.role === 'user'">
          <div class="user-bubble">
            {{ message.content }}
          </div>
        </template>

        <template v-else>
          <img
            class="assistant-avatar"
            :src="brandIcon"
            alt=""
            aria-hidden="true"
          >
          <div class="assistant-body">
            <AssistantMessage
              :message="message"
              @select-citation="emit('selectCitation', $event, message.citations)"
            />
            <div
              v-if="message.status !== 'streaming' && message.content.trim() !== ''"
              class="message-actions"
            >
              <button
                type="button"
                class="message-action"
                @click="copyMessage(message)"
              >
                {{ copiedMessageId === message.id ? "已复制" : "复制" }}
              </button>
              <button
                v-if="(sourcesByMessage[message.id] ?? []).length > 0"
                type="button"
                class="message-action"
                @click="emit('openSources', sourcesByMessage[message.id] ?? [])"
              >
                来源 {{ (sourcesByMessage[message.id] ?? []).length }}
              </button>
            </div>
          </div>
        </template>
      </article>

      <div
        v-if="isStreaming && messages[messages.length - 1]?.role === 'user'"
        class="reading-hint"
        role="status"
      >
        <span class="reading-hint__dot" />
        <span class="reading-hint__dot" />
        <span class="reading-hint__dot" />
        <span>正在阅读资料…</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.message-viewport {
  display: flex;
  min-height: 0;
  flex: 1;
  flex-direction: column;
  overflow-y: auto;
}

.message-empty {
  display: flex;
  width: min(620px, 100%);
  margin: auto;
  align-items: center;
  padding: 40px 24px;
  text-align: center;
  flex-direction: column;
}

.message-empty__mark {
  display: grid;
  width: 54px;
  height: 54px;
  place-items: center;
  color: var(--ks-accent);
  font-size: 22px;
  background: var(--ks-accent-soft);
  border-radius: 50%;
}

.message-empty h2 {
  margin: 18px 0 8px;
  color: var(--ks-ink);
  font-size: 24px;
  font-weight: 700;
  letter-spacing: -0.02em;
}

.message-empty p {
  max-width: 420px;
  margin: 0;
  color: var(--ks-muted);
  font-size: 14px;
  line-height: 1.75;
}

.message-empty__prompts {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 10px;
  margin-top: 24px;
}

.prompt-card {
  padding: 9px 14px;
  color: var(--ks-text);
  font-size: 13px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: 999px;
  cursor: pointer;
  transition: color var(--ks-duration-fast) var(--ks-ease-out),
    border-color var(--ks-duration-fast) var(--ks-ease-out);
}

.prompt-card:hover:not(:disabled) {
  color: var(--ks-accent-strong);
  border-color: var(--ks-accent);
}

.prompt-card:disabled {
  color: var(--ks-faint);
  cursor: not-allowed;
}

.message-column {
  display: flex;
  width: min(820px, 100%);
  margin: 0 auto;
  flex-direction: column;
  gap: 26px;
  padding: 28px 24px 32px;
}

.message-row {
  display: flex;
  gap: 14px;
}

.message-row--user {
  justify-content: flex-end;
}

.user-bubble {
  max-width: min(560px, 84%);
  padding: 12px 16px;
  color: var(--ks-ink);
  font-size: 15px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
  background: var(--ks-accent-soft);
  border-radius: 18px 18px 6px 18px;
}

.assistant-avatar {
  display: block;
  width: 30px;
  height: 30px;
  flex: 0 0 30px;
  object-fit: cover;
  background: var(--ks-accent);
  border-radius: 9px;
}

.assistant-body {
  min-width: 0;
  flex: 1;
}

.message-actions {
  display: flex;
  gap: 4px;
  margin-top: 10px;
}

.message-action {
  padding: 4px 10px;
  color: var(--ks-muted);
  font-size: 12px;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 6px;
  cursor: pointer;
  transition: color var(--ks-duration-fast) var(--ks-ease-out),
    border-color var(--ks-duration-fast) var(--ks-ease-out);
}

.message-action:hover {
  color: var(--ks-accent-strong);
  border-color: var(--ks-border);
}

.reading-hint {
  display: flex;
  align-items: center;
  gap: 5px;
  padding-left: 44px;
  color: var(--ks-muted);
  font-size: 12px;
}

.reading-hint__dot {
  width: 5px;
  height: 5px;
  background: var(--ks-accent);
  border-radius: 50%;
  animation: loading-bounce 1.2s ease-in-out infinite;
}

.reading-hint__dot:nth-child(2) {
  animation-delay: 120ms;
}

.reading-hint__dot:nth-child(3) {
  animation-delay: 240ms;
}

@keyframes loading-bounce {
  0%,
  60%,
  100% {
    transform: translateY(0);
    opacity: 0.45;
  }

  30% {
    transform: translateY(-4px);
    opacity: 1;
  }
}

@media (max-width: 720px) {
  .message-column {
    padding: 20px 16px 24px;
  }
}
</style>
