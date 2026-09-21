<script setup lang="ts">
import { ref } from "vue";

import type { ChatMessage } from "../../stores/chat";
import AssistantMessage from "./AssistantMessage.vue";

defineProps<{
  messages: ChatMessage[];
  isStreaming: boolean;
}>();

const emit = defineEmits<{ selectCitation: [marker: string] }>();
const listElement = ref<HTMLElement | null>(null);

function scrollToBottom(): void {
  const element = listElement.value;
  if (element === null) {
    return;
  }
  if (typeof element.scrollTo === "function") {
    element.scrollTo({ top: element.scrollHeight, behavior: "smooth" });
  } else {
    element.scrollTop = element.scrollHeight;
  }
}

defineExpose({ scrollToBottom });
</script>

<template>
  <div
    ref="listElement"
    class="chat-message-list"
    aria-live="polite"
    aria-label="对话内容"
  >
    <div
      v-if="messages.length === 0"
      class="chat-empty"
    >
      <div
        class="chat-empty__orb"
        aria-hidden="true"
      >
        ✦
      </div>
      <p class="chat-empty__title">
        从资料中找到答案
      </p>
      <p class="chat-empty__copy">
        选择一个知识库，提出关于文档内容的问题。回答会保留来源，方便回看。
      </p>
      <div
        class="chat-empty__tips"
        aria-label="提问提示"
      >
        <span>查找关键事实</span>
        <span>梳理章节内容</span>
        <span>对比资料观点</span>
      </div>
    </div>

    <template v-else>
      <div
        v-for="message in messages"
        :key="message.id"
        class="chat-message"
        :class="`chat-message--${message.role}`"
      >
        <div
          v-if="message.role === 'user'"
          class="user-message"
        >
          {{ message.content }}
        </div>
        <AssistantMessage
          v-else
          :message="message"
          @select-citation="emit('selectCitation', $event)"
        />
      </div>
      <div
        v-if="isStreaming && messages[messages.length - 1]?.role === 'user'"
        class="chat-loading"
        role="status"
      >
        <span class="chat-loading__dot" />
        <span class="chat-loading__dot" />
        <span class="chat-loading__dot" />
        <span>正在阅读资料…</span>
      </div>
    </template>
  </div>
</template>

<style scoped>
.chat-message-list {
  display: flex;
  min-height: 0;
  flex: 1;
  flex-direction: column;
  gap: 22px;
  overflow-y: auto;
  padding: 34px clamp(24px, 5vw, 72px) 28px;
}

.chat-empty {
  display: flex;
  width: min(540px, 100%);
  margin: auto;
  align-items: center;
  text-align: center;
  flex-direction: column;
}

.chat-empty__orb {
  display: grid;
  width: 58px;
  height: 58px;
  place-items: center;
  color: var(--ks-accent);
  font-size: 24px;
  background: var(--ks-accent-soft);
  border: 8px solid rgb(230 240 236 / 55%);
  border-radius: 50%;
}

.chat-empty__title {
  margin: 20px 0 8px;
  color: var(--ks-ink);
  font-size: 25px;
  font-weight: 720;
  letter-spacing: -0.04em;
}

.chat-empty__copy {
  max-width: 430px;
  margin: 0;
  color: var(--ks-muted);
  font-size: 14px;
  line-height: 1.75;
}

.chat-empty__tips {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
  margin-top: 22px;
}

.chat-empty__tips span {
  padding: 6px 10px;
  color: var(--ks-accent-strong);
  font-size: 11px;
  background: var(--ks-accent-soft);
  border-radius: 999px;
}

.chat-message--user {
  display: flex;
  justify-content: flex-end;
}

.user-message {
  max-width: min(580px, 86%);
  padding: 12px 16px;
  color: var(--ks-ink);
  font-size: 14px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
  background: var(--ks-accent-soft);
  border: 1px solid rgb(55 116 102 / 10%);
  border-radius: 16px 16px 4px 16px;
}

.chat-message--assistant {
  padding-right: 12px;
}

.chat-loading {
  display: flex;
  align-items: center;
  gap: 5px;
  padding-left: 42px;
  color: var(--ks-muted);
  font-size: 12px;
}

.chat-loading__dot {
  width: 5px;
  height: 5px;
  background: var(--ks-accent);
  border-radius: 50%;
  animation: loading-bounce 1.2s ease-in-out infinite;
}

.chat-loading__dot:nth-child(2) {
  animation-delay: 120ms;
}

.chat-loading__dot:nth-child(3) {
  animation-delay: 240ms;
}

.chat-loading__dot + .chat-loading__dot {
  margin-right: 2px;
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
</style>
