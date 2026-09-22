<script setup lang="ts">
import { computed, ref } from "vue";

const props = defineProps<{
  disabled: boolean;
  loading: boolean;
  canSend: boolean;
}>();

const emit = defineEmits<{
  send: [question: string];
  stop: [];
}>();

const question = ref("");
const hasQuestion = computed(() => question.value.trim().length > 0);

function submit(): void {
  const normalized = question.value.trim();
  if (!normalized || !props.canSend || props.disabled) {
    return;
  }
  emit("send", normalized);
  question.value = "";
}

function handleKeydown(event: KeyboardEvent): void {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submit();
  }
}
</script>

<template>
  <form
    class="analysis-composer"
    @submit.prevent="submit"
  >
    <textarea
      v-model="question"
      class="analysis-composer__input"
      rows="4"
      maxlength="10000"
      :disabled="disabled"
      placeholder="例如：统计每个地区的客户数量，并按数量从高到低排列。"
      aria-label="分析问题"
      @keydown="handleKeydown"
    />
    <div class="analysis-composer__footer">
      <span>Enter 发送 · Shift + Enter 换行</span>
      <button
        v-if="loading"
        class="analysis-composer__button analysis-composer__button--stop"
        type="button"
        @click="emit('stop')"
      >
        停止
      </button>
      <button
        v-else
        class="analysis-composer__button"
        type="submit"
        :disabled="disabled || !hasQuestion || !canSend"
      >
        开始分析
      </button>
    </div>
  </form>
</template>

<style scoped>
.analysis-composer {
  padding: 14px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-lg);
  box-shadow: var(--ks-shadow-sm);
}

.analysis-composer__input {
  display: block;
  width: 100%;
  min-height: 94px;
  resize: vertical;
  padding: 10px 12px;
  color: var(--ks-ink);
  line-height: 1.6;
  background: var(--ks-surface-subtle);
  border: 1px solid var(--ks-border);
  border-radius: 9px;
}

.analysis-composer__input:focus {
  border-color: var(--ks-accent);
  outline: 2px solid rgb(55 116 102 / 12%);
  outline-offset: 1px;
}

.analysis-composer__footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 10px;
  color: var(--ks-muted);
  font-size: 11px;
}

.analysis-composer__button {
  min-height: 36px;
  padding: 0 16px;
  color: var(--ks-surface);
  font-size: 13px;
  font-weight: 700;
  background: var(--ks-accent);
  border: 0;
  border-radius: 8px;
  cursor: pointer;
}

.analysis-composer__button:hover:not(:disabled) {
  background: var(--ks-accent-strong);
}

.analysis-composer__button:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.analysis-composer__button--stop {
  color: var(--ks-danger);
  background: #fff1ef;
}

@media (max-width: 560px) {
  .analysis-composer__footer {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
