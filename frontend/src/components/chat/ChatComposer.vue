<script setup lang="ts">
import { computed, ref } from "vue";

const props = defineProps<{
  disabled: boolean;
  canSend: boolean;
}>();

const emit = defineEmits<{
  send: [query: string];
  stop: [];
}>();

const query = ref("");
const canSubmit = computed(() => props.canSend && query.value.trim().length > 0);

function submit(): void {
  if (!canSubmit.value) {
    return;
  }
  const value = query.value.trim();
  query.value = "";
  emit("send", value);
}

function onKeydown(event: KeyboardEvent): void {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submit();
  }
}
</script>

<template>
  <div class="composer-wrap">
    <div class="composer">
      <textarea
        v-model="query"
        class="composer__input"
        rows="1"
        maxlength="4000"
        placeholder="询问知识库中的内容…"
        aria-label="输入问题"
        :disabled="disabled"
        @keydown="onKeydown"
      />
      <div class="composer__footer">
        <span class="composer__hint">Enter 发送 · Shift + Enter 换行</span>
        <button
          v-if="disabled"
          class="composer__button composer__button--stop"
          type="button"
          aria-label="停止生成"
          @click="emit('stop')"
        >
          停止
        </button>
        <button
          v-else
          class="composer__button"
          type="button"
          :disabled="!canSubmit"
          @click="submit"
        >
          发送
          <span aria-hidden="true">↗</span>
        </button>
      </div>
    </div>
    <p class="composer__notice">
      回答来自当前知识库中的资料，请结合引用来源判断。
    </p>
  </div>
</template>

<style scoped>
.composer-wrap {
  padding: 0 clamp(24px, 5vw, 72px) 22px;
}

.composer {
  padding: 12px 14px 10px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border-strong);
  border-radius: 14px;
  box-shadow: 0 6px 18px rgb(32 38 34 / 5%);
}

.composer:focus-within {
  border-color: var(--ks-accent);
  box-shadow: 0 0 0 3px rgb(55 116 102 / 10%), 0 6px 18px rgb(32 38 34 / 5%);
}

.composer__input {
  display: block;
  width: 100%;
  min-height: 46px;
  max-height: 160px;
  padding: 5px 2px;
  color: var(--ks-ink);
  font-size: 14px;
  line-height: 1.7;
  resize: vertical;
  background: transparent;
  border: 0;
  outline: 0;
}

.composer__input::placeholder {
  color: var(--ks-faint);
}

.composer__input:disabled {
  cursor: wait;
  opacity: 0.65;
}

.composer__footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.composer__hint,
.composer__notice {
  color: var(--ks-faint);
  font-size: 11px;
}

.composer__button {
  display: inline-flex;
  min-width: 72px;
  min-height: 32px;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 0 12px;
  color: var(--ks-surface);
  font-size: 12px;
  font-weight: 700;
  background: var(--ks-accent);
  border: 0;
  border-radius: 7px;
  cursor: pointer;
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
  background: #f8eae8;
}

.composer__notice {
  margin: 8px 0 0;
  text-align: center;
}
</style>
