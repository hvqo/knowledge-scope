<script setup lang="ts">
import { computed } from "vue";

import type { ChatMessage } from "../../stores/chat";

const props = defineProps<{ message: ChatMessage }>();
const emit = defineEmits<{ selectCitation: [marker: string] }>();

interface MessagePart {
  value: string;
  marker: string | null;
}

const parts = computed<MessagePart[]>(() => {
  const result: MessagePart[] = [];
  const pattern = /\[C[1-9][0-9]*\]/g;
  let cursor = 0;
  for (const match of props.message.content.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > cursor) {
      result.push({ value: props.message.content.slice(cursor, index), marker: null });
    }
    result.push({ value: match[0], marker: match[0] });
    cursor = index + match[0].length;
  }
  if (cursor < props.message.content.length) {
    result.push({ value: props.message.content.slice(cursor), marker: null });
  }
  return result;
});

function hasCitation(marker: string | null): boolean {
  const citationMarker = marker?.slice(1, -1);
  return citationMarker !== undefined &&
    props.message.citations.some((item) => item.marker === citationMarker);
}

function selectCitation(marker: string | null): void {
  const citationMarker = marker?.slice(1, -1);
  if (hasCitation(marker) && citationMarker !== undefined) {
    emit("selectCitation", citationMarker);
  }
}
</script>

<template>
  <div class="assistant-message">
    <div
      class="assistant-message__avatar"
      aria-hidden="true"
    >
      K
    </div>
    <div class="assistant-message__body">
      <div class="assistant-message__label">
        KnowledgeScope
      </div>
      <div
        class="assistant-message__content"
        :class="{ 'is-error': message.status === 'error' }"
      >
        <template
          v-for="(part, index) in parts"
          :key="`${part.value}-${index}`"
        >
          <button
            v-if="part.marker !== null"
            type="button"
            class="citation-marker"
            :aria-label="`查看来源 ${part.marker}`"
            :disabled="!hasCitation(part.marker)"
            @click="selectCitation(part.marker)"
          >
            {{ part.value }}
          </button>
          <span v-else>{{ part.value }}</span>
        </template>
        <span
          v-if="message.status === 'streaming'"
          class="streaming-cursor"
          aria-label="正在生成"
        />
      </div>
      <div
        v-if="message.status === 'error'"
        class="assistant-message__hint"
      >
        本次回答没有完成，可以稍后重试。
      </div>
    </div>
  </div>
</template>

<style scoped>
.assistant-message {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.assistant-message__avatar {
  display: grid;
  width: 30px;
  height: 30px;
  flex: 0 0 30px;
  place-items: center;
  color: var(--ks-surface);
  font-size: 12px;
  font-weight: 760;
  background: var(--ks-accent);
  border-radius: 9px;
}

.assistant-message__body {
  min-width: 0;
  max-width: min(720px, 100%);
}

.assistant-message__label {
  margin-bottom: 6px;
  color: var(--ks-muted);
  font-size: 11px;
  font-weight: 680;
}

.assistant-message__content {
  color: var(--ks-text);
  font-size: 15px;
  line-height: 1.85;
  white-space: pre-wrap;
  word-break: break-word;
}

.assistant-message__content.is-error {
  color: var(--ks-danger);
}

.citation-marker {
  display: inline;
  padding: 1px 4px;
  color: var(--ks-accent-strong);
  font-size: 0.9em;
  font-weight: 720;
  background: var(--ks-accent-soft);
  border: 0;
  border-radius: 4px;
  cursor: pointer;
}

.citation-marker:hover {
  color: var(--ks-surface);
  background: var(--ks-accent);
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

.assistant-message__hint {
  margin-top: 8px;
  color: var(--ks-muted);
  font-size: 12px;
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
