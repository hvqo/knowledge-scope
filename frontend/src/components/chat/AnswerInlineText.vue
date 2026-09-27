<script setup lang="ts">
import { defineAsyncComponent } from "vue";

import type { AnswerInline } from "./answerFormat";
import type { RAGCitation } from "../../api/types";

// Math is a small part of most answers, so KaTeX loads only when a
// formula actually appears.
const AnswerMath = defineAsyncComponent(() => import("./AnswerMath.vue"));

const props = defineProps<{
  tokens: AnswerInline[];
  citations: RAGCitation[];
}>();

const emit = defineEmits<{ selectCitation: [marker: string] }>();

function citationMarker(token: AnswerInline): string | null {
  return token.kind === "marker" ? token.value.slice(1, -1) : null;
}

function hasCitation(token: AnswerInline): boolean {
  const marker = citationMarker(token);
  return marker !== null && props.citations.some((citation) => citation.marker === marker);
}

function selectCitation(token: AnswerInline): void {
  const marker = citationMarker(token);
  if (marker !== null && hasCitation(token)) {
    emit("selectCitation", marker);
  }
}
</script>

<template>
  <template
    v-for="(token, index) in tokens"
    :key="index"
  >
    <button
      v-if="token.kind === 'marker'"
      type="button"
      class="citation-marker"
      :aria-label="`查看来源 ${token.value}`"
      :disabled="!hasCitation(token)"
      @click="selectCitation(token)"
    >
      {{ token.value }}
    </button>
    <strong
      v-else-if="token.kind === 'bold'"
      class="answer-inline__strong"
    >{{ token.value }}</strong>
    <AnswerMath
      v-else-if="token.kind === 'math'"
      :latex="token.value"
    />
    <code
      v-else-if="token.kind === 'code'"
      class="answer-inline__code"
    >{{ token.value }}</code>
    <span v-else>{{ token.value }}</span>
  </template>
</template>

<style scoped>
.answer-inline__strong {
  color: var(--ks-ink);
  font-weight: 650;
}

.answer-inline__code {
  padding: 1px 5px;
  font-size: 0.92em;
  background: var(--ks-surface-subtle);
  border-radius: 4px;
}

.citation-marker {
  display: inline;
  padding: 1px 5px;
  margin: 0 1px;
  color: var(--ks-accent-strong);
  font-size: 0.88em;
  font-weight: 720;
  background: var(--ks-accent-soft);
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background-color var(--ks-duration-fast) var(--ks-ease-out);
}

.citation-marker:hover:not(:disabled) {
  background: #cfe4dc;
}

.citation-marker:disabled {
  color: var(--ks-faint);
  background: var(--ks-surface-subtle);
  cursor: default;
}
</style>
