<script setup lang="ts">
import { computed, defineAsyncComponent } from "vue";

import AnswerInlineText from "./AnswerInlineText.vue";
import { parseAnswer } from "./answerFormat";
import type { RAGCitation } from "../../api/types";

// Math is a small part of most answers, so KaTeX loads only when a
// formula actually appears.
const AnswerMath = defineAsyncComponent(() => import("./AnswerMath.vue"));

const props = defineProps<{
  content: string;
  citations: RAGCitation[];
}>();

const emit = defineEmits<{ selectCitation: [marker: string] }>();

const blocks = computed(() => parseAnswer(props.content));
</script>

<template>
  <div class="answer-text">
    <template
      v-for="(block, blockIndex) in blocks"
      :key="blockIndex"
    >
      <h3
        v-if="block.kind === 'heading'"
        class="answer-text__heading"
        :class="`is-level-${block.level}`"
      >
        <AnswerInlineText
          :tokens="block.content"
          :citations="citations"
          @select-citation="emit('selectCitation', $event)"
        />
      </h3>

      <ul
        v-else-if="block.kind === 'list' && !block.ordered"
        class="answer-text__list"
      >
        <li
          v-for="(item, itemIndex) in block.items"
          :key="itemIndex"
        >
          <AnswerInlineText
            :tokens="item"
            :citations="citations"
            @select-citation="emit('selectCitation', $event)"
          />
        </li>
      </ul>

      <ol
        v-else-if="block.kind === 'list'"
        class="answer-text__list answer-text__list--ordered"
      >
        <li
          v-for="(item, itemIndex) in block.items"
          :key="itemIndex"
        >
          <AnswerInlineText
            :tokens="item"
            :citations="citations"
            @select-citation="emit('selectCitation', $event)"
          />
        </li>
      </ol>

      <AnswerMath
        v-else-if="block.kind === 'math'"
        :latex="block.value"
        display
      />

      <p
        v-else
        class="answer-text__paragraph"
      >
        <AnswerInlineText
          :tokens="block.content"
          :citations="citations"
          @select-citation="emit('selectCitation', $event)"
        />
      </p>
    </template>
  </div>
</template>

<style scoped>
.answer-text {
  color: var(--ks-text);
  font-size: 15px;
  line-height: 1.78;
  word-break: break-word;
}

.answer-text__paragraph {
  margin: 0 0 12px;
  white-space: pre-wrap;
}

.answer-text > :last-child {
  margin-bottom: 0;
}

.answer-text__heading {
  margin: 2px 0 10px;
  color: var(--ks-ink);
  font-size: 15px;
  font-weight: 700;
  line-height: 1.5;
}

.answer-text__heading.is-level-3 {
  color: var(--ks-muted);
  font-size: 13px;
  font-weight: 650;
  letter-spacing: 0.02em;
}

.answer-text__list {
  margin: 0 0 12px;
  padding-left: 20px;
}

.answer-text__list--ordered {
  padding-left: 22px;
}

.answer-text__list li {
  margin-bottom: 6px;
  padding-left: 2px;
}

.answer-text__list li:last-child {
  margin-bottom: 0;
}

.answer-text__list li::marker {
  color: var(--ks-accent);
  font-weight: 620;
}





</style>
