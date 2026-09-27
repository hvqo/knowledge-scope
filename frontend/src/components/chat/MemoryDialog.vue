<script setup lang="ts">
import { computed, ref, watch } from "vue";

const props = defineProps<{
  open: boolean;
  memories: { id: string; content: string; created_at: string }[];
  knowledgeBaseName: string | null;
  loading: boolean;
}>();

const emit = defineEmits<{
  close: [];
  forget: [id: string];
  remember: [content: string];
}>();

const draft = ref("");

const sorted = computed(() =>
  [...props.memories].sort(
    (first, second) => new Date(second.created_at).getTime() - new Date(first.created_at).getTime(),
  ),
);

watch(
  () => props.open,
  (open) => {
    if (open) {
      draft.value = "";
    }
  },
);

function remember(): void {
  const content = draft.value.trim();
  if (content === "") {
    return;
  }
  emit("remember", content);
  draft.value = "";
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return new Intl.DateTimeFormat("zh-CN", { month: "numeric", day: "numeric" }).format(date);
}
</script>

<template>
  <div
    v-if="open"
    class="memory-dialog"
    role="dialog"
    aria-modal="true"
    aria-label="长期记忆"
  >
    <button
      class="memory-dialog__backdrop"
      type="button"
      aria-label="关闭长期记忆"
      @click="emit('close')"
    />
    <div
      class="memory-dialog__panel"
      role="document"
    >
      <header class="memory-dialog__head">
        <div>
          <span class="memory-dialog__eyebrow">跨对话记住</span>
          <h2>长期记忆</h2>
        </div>
        <button
          class="memory-dialog__close"
          type="button"
          aria-label="关闭"
          @click="emit('close')"
        >
          ✕
        </button>
      </header>

      <p class="memory-dialog__hint">
        这些是关于你的长期记忆，只在「{{ knowledgeBaseName ?? "当前知识库" }}」的对话间共享，
        会作为背景提供给回答，但不会当成资料来源。
      </p>

      <form
        class="memory-dialog__form"
        @submit.prevent="remember"
      >
        <input
          v-model="draft"
          class="memory-dialog__input"
          type="text"
          maxlength="200"
          placeholder="手动添加一条记忆，例如「我在准备中考化学」"
          aria-label="新增长期记忆"
        >
        <button
          class="memory-dialog__add"
          type="submit"
          :disabled="draft.trim() === ''"
        >
          记住
        </button>
      </form>

      <div
        v-if="loading"
        class="memory-dialog__state"
        role="status"
      >
        正在加载…
      </div>
      <div
        v-else-if="sorted.length === 0"
        class="memory-dialog__state"
      >
        还没有长期记忆。随着对话积累，值得记住的信息会自动出现在这里。
      </div>
      <ul
        v-else
        class="memory-dialog__list"
      >
        <li
          v-for="memory in sorted"
          :key="memory.id"
          class="memory-dialog__item"
        >
          <span class="memory-dialog__content">{{ memory.content }}</span>
          <span class="memory-dialog__meta">{{ formatDate(memory.created_at) }}</span>
          <button
            type="button"
            class="memory-dialog__forget"
            :aria-label="`忘记：${memory.content}`"
            @click="emit('forget', memory.id)"
          >
            忘记
          </button>
        </li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
.memory-dialog {
  position: fixed;
  inset: 0;
  z-index: 60;
  display: grid;
  place-items: center;
  padding: 20px;
}

.memory-dialog__backdrop {
  position: absolute;
  inset: 0;
  background: rgb(15 32 28 / 32%);
}

.memory-dialog__panel {
  position: relative;
  display: flex;
  width: min(520px, 100%);
  max-height: min(70vh, 560px);
  flex-direction: column;
  gap: 12px;
  padding: 20px;
  overflow: hidden;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-lg);
  box-shadow: var(--ks-shadow-md);
}

.memory-dialog__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
}

.memory-dialog__eyebrow {
  color: var(--ks-accent);
  font-size: 11px;
  font-weight: 720;
  letter-spacing: 0.08em;
}

.memory-dialog__head h2 {
  margin: 4px 0 0;
  color: var(--ks-ink);
  font-size: 18px;
  letter-spacing: -0.03em;
}

.memory-dialog__close {
  display: grid;
  width: 30px;
  height: 30px;
  place-items: center;
  color: var(--ks-muted);
  background: transparent;
  border: 1px solid var(--ks-border);
  border-radius: 8px;
}

.memory-dialog__hint {
  margin: 0;
  color: var(--ks-muted);
  font-size: 12px;
  line-height: 1.6;
}

.memory-dialog__form {
  display: flex;
  gap: 8px;
}

.memory-dialog__input {
  flex: 1;
  min-height: 36px;
  padding: 0 10px;
  color: var(--ks-ink);
  font-size: 13px;
  background: var(--ks-surface-subtle);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-sm);
}

.memory-dialog__add {
  padding: 0 14px;
  color: var(--ks-surface);
  font-size: 12px;
  font-weight: 680;
  background: var(--ks-accent);
  border: 0;
  border-radius: var(--ks-radius-sm);
}

.memory-dialog__add:disabled {
  opacity: 0.5;
}

.memory-dialog__state {
  padding: 16px 4px;
  color: var(--ks-muted);
  font-size: 12px;
  text-align: center;
}

.memory-dialog__list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin: 0;
  padding: 0;
  list-style: none;
  overflow-y: auto;
}

.memory-dialog__item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 10px;
  background: var(--ks-surface-subtle);
  border-radius: var(--ks-radius-sm);
}

.memory-dialog__content {
  flex: 1;
  color: var(--ks-text);
  font-size: 13px;
  line-height: 1.6;
}

.memory-dialog__meta {
  color: var(--ks-faint);
  font-size: 10px;
}

.memory-dialog__forget {
  flex: 0 0 auto;
  padding: 3px 8px;
  color: var(--ks-muted);
  font-size: 11px;
  background: transparent;
  border: 1px solid var(--ks-border);
  border-radius: 6px;
}

.memory-dialog__forget:hover {
  color: var(--ks-danger);
  border-color: var(--ks-danger);
}
</style>
