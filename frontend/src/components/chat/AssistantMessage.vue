<script setup lang="ts">
import AnswerText from "./AnswerText.vue";
import type { ChatMessageViewStatus, RAGCitation } from "../../api/types";

defineProps<{
  message: {
    content: string;
    status: ChatMessageViewStatus;
    citations: RAGCitation[];
  };
}>();

const emit = defineEmits<{ selectCitation: [marker: string] }>();
</script>

<template>
  <div class="assistant-message">
    <div
      class="assistant-message__content"
      :class="{ 'is-error': message.status === 'error' }"
    >
      <AnswerText
        :content="message.content"
        :citations="message.citations"
        @select-citation="emit('selectCitation', $event)"
      />
      <span
        v-if="message.status === 'streaming'"
        class="streaming-cursor"
        aria-label="正在生成"
      />
    </div>
    <p
      v-if="message.status === 'error'"
      class="assistant-message__hint"
    >
      本次回答没有完成，可以重新提问。
    </p>
  </div>
</template>

<style scoped>
.assistant-message {
  min-width: 0;
}

.assistant-message__content {
  color: var(--ks-text);
  font-size: 15px;
}

.assistant-message__content.is-error :deep(.answer-text) {
  color: var(--ks-danger);
}

.assistant-message__hint {
  margin: 10px 0 0;
  color: var(--ks-muted);
  font-size: 13px;
}

.streaming-cursor {
  display: inline-block;
  width: 7px;
  height: 16px;
  margin-left: 4px;
  vertical-align: -2px;
  background: var(--ks-accent);
  border-radius: 2px;
  animation: cursor-pulse 1s ease-in-out infinite;
}

@keyframes cursor-pulse {
  0%,
  100% {
    opacity: 0.35;
  }

  50% {
    opacity: 1;
  }
}
</style>
