<script setup lang="ts">
import { computed, nextTick, ref } from "vue";

import type { RAGRetrievalMode } from "../../api/types";

const props = defineProps<{
  disabled: boolean;
  canSend: boolean;
  placeholder?: string;
  retrievalMode: RAGRetrievalMode | "auto";
}>();

const emit = defineEmits<{
  send: [query: string];
  stop: [];
  "update:retrievalMode": [mode: RAGRetrievalMode | "auto"];
}>();

const RETRIEVAL_MODES = [
  {
    value: "auto",
    label: "智能",
    hint: "元问题直接回答；简单问题走向量检索，复杂问题自动多路检索",
  },
  { value: "unified", label: "深度", hint: "向量 + 关键词 + 图谱 + 多模态，覆盖更全" },
  { value: "dense", label: "快速", hint: "仅向量检索，速度更快" },
] as const;

const query = ref("");
const inputElement = ref<HTMLTextAreaElement | null>(null);
const canSubmit = computed(() => props.canSend && query.value.trim().length > 0);

function resize(): void {
  const element = inputElement.value;
  if (element === null) {
    return;
  }
  element.style.height = "auto";
  element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
}

function submit(): void {
  console.log("PROBE submit", props.disabled, props.canSend, JSON.stringify(query.value));
  if (!canSubmit.value) {
    return;
  }
  const value = query.value.trim();
  query.value = "";
  void nextTick(resize);
  emit("send", value);
}

function fill(text: string): void {
  query.value = text;
  void nextTick(() => {
    resize();
    inputElement.value?.focus();
  });
}

function onKeydown(event: KeyboardEvent): void {
  console.log("PROBE keydown", event.key, props.disabled);
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submit();
  }
}

defineExpose({ fill, focus: () => inputElement.value?.focus() });
</script>

<template>
  <div class="composer-area">
    <div class="composer">
      <textarea
        ref="inputElement"
        v-model="query"
        class="composer__input"
        rows="1"
        maxlength="4000"
        :placeholder="placeholder ?? '给知识库提问，Enter 发送'"
        aria-label="输入问题"
        :disabled="disabled"
        @input="resize"
        @keydown="onKeydown"
      />
      <button
        v-if="disabled"
        class="composer__button composer__button--stop"
        type="button"
        aria-label="停止生成"
        title="停止生成"
        @click="emit('stop')"
      >
        ■
      </button>
      <button
        v-else
        class="composer__button"
        type="button"
        aria-label="发送问题"
        title="发送"
        :disabled="!canSubmit"
        @click="submit"
      >
        ↑
      </button>
    </div>
    <div class="composer__footer">
      <div
        class="composer__modes"
        role="radiogroup"
        aria-label="检索范围"
      >
        <button
          v-for="mode in RETRIEVAL_MODES"
          :key="mode.value"
          type="button"
          role="radio"
          class="composer__mode"
          :class="{ 'is-active': retrievalMode === mode.value }"
          :aria-checked="retrievalMode === mode.value"
          :title="mode.hint"
          @click="emit('update:retrievalMode', mode.value)"
        >
          {{ mode.label }}
        </button>
      </div>
      <p class="composer__hint">
        Enter 发送 · Shift + Enter 换行
      </p>
    </div>
  </div>
</template>

<style scoped>
.composer-area {
  padding: 0 24px 20px;
  background: linear-gradient(to top, var(--ks-surface) 60%, transparent);
}

.composer {
  display: flex;
  width: min(820px, 100%);
  margin: 0 auto;
  align-items: flex-end;
  gap: 10px;
  padding: 10px 12px 10px 16px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border-strong);
  border-radius: 22px;
  box-shadow: 0 10px 28px rgb(32 38 34 / 6%);
  transition: border-color var(--ks-duration-fast) var(--ks-ease-out),
    box-shadow var(--ks-duration-fast) var(--ks-ease-out);
}

.composer:focus-within {
  border-color: var(--ks-accent);
  box-shadow: 0 0 0 3px rgb(55 116 102 / 10%), 0 10px 28px rgb(32 38 34 / 6%);
}

.composer__input {
  display: block;
  width: 100%;
  min-height: 26px;
  max-height: 200px;
  padding: 4px 0;
  color: var(--ks-ink);
  font-size: 15px;
  line-height: 1.7;
  overflow-y: auto;
  resize: none;
  background: transparent;
  border: 0;
  outline: 0;
}

.composer__input::placeholder {
  color: var(--ks-faint);
}

.composer__input:disabled {
  cursor: wait;
  opacity: 0.7;
}

.composer__button {
  display: grid;
  width: 34px;
  height: 34px;
  flex: 0 0 34px;
  place-items: center;
  color: var(--ks-surface);
  font-size: 15px;
  background: var(--ks-accent);
  border: 0;
  border-radius: 50%;
  cursor: pointer;
  transition: background-color var(--ks-duration-fast) var(--ks-ease-out);
}

.composer__button:hover:not(:disabled) {
  background: var(--ks-accent-strong);
}

.composer__button:disabled {
  color: var(--ks-faint);
  background: var(--ks-surface-muted);
  cursor: not-allowed;
}

.composer__button--stop {
  color: var(--ks-danger);
  font-size: 11px;
  background: #f8eae8;
}

.composer__footer {
  display: flex;
  width: min(820px, 100%);
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin: 8px auto 0;
}

.composer__modes {
  display: inline-flex;
  flex: 0 0 auto;
  padding: 2px;
  background: var(--ks-surface-subtle);
  border: 1px solid var(--ks-border);
  border-radius: 999px;
}

.composer__mode {
  padding: 4px 12px;
  color: var(--ks-muted);
  font-size: 11px;
  font-weight: 650;
  background: transparent;
  border: 0;
  border-radius: 999px;
  cursor: pointer;
  transition: color var(--ks-duration-fast) var(--ks-ease-out),
    background-color var(--ks-duration-fast) var(--ks-ease-out);
}

.composer__mode:hover {
  color: var(--ks-ink);
}

.composer__mode.is-active {
  color: var(--ks-surface);
  background: var(--ks-accent);
}

.composer__hint {
  margin: 0;
  color: var(--ks-faint);
  font-size: 11px;
  text-align: right;
}

@media (max-width: 720px) {
  .composer-area {
    padding: 0 16px 16px;
  }
}
</style>
