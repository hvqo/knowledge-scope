<script setup lang="ts">
import { ref, watch } from "vue";

import type {
  ReportAIDraftResponse,
  ReportAIOutlineItem,
  ReportAIOutlineResponse,
} from "../../api/types";

type Preview =
  | { kind: "outline"; value: ReportAIOutlineResponse }
  | { kind: "draft"; value: ReportAIDraftResponse }
  | null;

const props = defineProps<{
  preview: Preview;
  busy: boolean;
  disabled: boolean;
  error: string | null;
}>();

const emit = defineEmits<{
  generateOutline: [instruction: string];
  generateSection: [instruction: string];
  edit: [operation: "rewrite" | "expand" | "summarize", instruction: string];
  acceptOutline: [items: ReportAIOutlineItem[]];
  acceptDraft: [draft: ReportAIDraftResponse];
  discard: [];
}>();

const instruction = ref("");
const outlineItems = ref<ReportAIOutlineItem[]>([]);
const draftContent = ref("");

watch(
  () => props.preview,
  (preview) => {
    outlineItems.value = preview?.kind === "outline"
      ? preview.value.items.map((item) => ({ ...item }))
      : [];
    draftContent.value = preview?.kind === "draft" ? preview.value.content : "";
  },
  { immediate: true },
);

function submitGenerateOutline(): void {
  emit("generateOutline", instruction.value.trim());
}

function submitGenerateSection(): void {
  emit("generateSection", instruction.value.trim());
}

function submitEdit(operation: "rewrite" | "expand" | "summarize"): void {
  emit("edit", operation, instruction.value.trim());
}

function accept(): void {
  if (props.preview?.kind === "outline") {
    emit("acceptOutline", outlineItems.value);
  } else if (props.preview?.kind === "draft") {
    emit("acceptDraft", { ...props.preview.value, content: draftContent.value.trim() });
  }
}
</script>

<template>
  <section
    class="report-ai-panel"
    aria-label="AI 辅助写作"
  >
    <header class="report-ai-panel__header">
      <div>
        <span class="report-ai-panel__eyebrow">辅助写作</span>
        <h2>AI 预览</h2>
      </div>
      <button
        v-if="preview"
        type="button"
        class="report-ai-panel__close"
        :disabled="busy"
        @click="emit('discard')"
      >
        关闭
      </button>
    </header>

    <p class="report-ai-panel__note">
      生成内容只会进入预览。接受后仍需保存章节；引用只来自当前报告已有来源。
    </p>
    <label
      class="report-ai-panel__label"
      for="report-ai-instruction"
    >补充要求（可选）</label>
    <textarea
      id="report-ai-instruction"
      v-model="instruction"
      class="report-ai-panel__instruction"
      :disabled="disabled || busy"
      maxlength="2000"
      placeholder="例如：突出变化趋势，保持客观。"
    />
    <div class="report-ai-panel__actions">
      <button
        type="button"
        :disabled="disabled || busy"
        @click="submitGenerateOutline"
      >
        {{ busy ? "生成中…" : "生成大纲" }}
      </button>
      <button
        type="button"
        :disabled="disabled || busy"
        @click="submitGenerateSection"
      >
        生成本章节
      </button>
      <button
        type="button"
        :disabled="disabled || busy"
        @click="submitEdit('rewrite')"
      >
        改写
      </button>
      <button
        type="button"
        :disabled="disabled || busy"
        @click="submitEdit('expand')"
      >
        扩写
      </button>
      <button
        type="button"
        :disabled="disabled || busy"
        @click="submitEdit('summarize')"
      >
        精简
      </button>
    </div>
    <p
      v-if="error"
      class="report-ai-panel__error"
      role="alert"
    >
      {{ error }}
    </p>

    <div
      v-if="preview?.kind === 'outline'"
      class="report-ai-panel__preview"
    >
      <h3>大纲预览</h3>
      <p class="report-ai-panel__hint">
        可以先调整标题和摘要，再追加到报告。
      </p>
      <div
        v-for="(item, index) in outlineItems"
        :key="index"
        class="report-ai-panel__outline-item"
      >
        <input
          v-model="item.title"
          maxlength="200"
          aria-label="大纲标题"
        >
        <textarea
          v-model="item.summary"
          maxlength="1000"
          aria-label="大纲摘要"
        />
      </div>
    </div>
    <div
      v-else-if="preview?.kind === 'draft'"
      class="report-ai-panel__preview"
    >
      <h3>章节预览</h3>
      <textarea
        v-model="draftContent"
        class="report-ai-panel__draft"
        maxlength="100000"
      />
      <ul
        v-if="preview.value.citations.length"
        class="report-ai-panel__citations"
      >
        <li
          v-for="citation in preview.value.citations"
          :key="citation.marker"
        >
          [{{ citation.marker }}] {{ citation.title }}
        </li>
      </ul>
    </div>
    <div
      v-if="preview"
      class="report-ai-panel__footer"
    >
      <button
        type="button"
        class="report-ai-panel__discard"
        :disabled="busy"
        @click="emit('discard')"
      >
        放弃预览
      </button>
      <button
        type="button"
        class="report-ai-panel__accept"
        :disabled="busy"
        @click="accept"
      >
        {{ preview.kind === "outline" ? "追加到报告" : "放入当前草稿" }}
      </button>
    </div>
  </section>
</template>

<style scoped>
.report-ai-panel {
  padding: 18px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-lg);
  box-shadow: var(--ks-shadow-sm);
}

.report-ai-panel__header,
.report-ai-panel__footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.report-ai-panel__eyebrow {
  color: var(--ks-accent);
  font-size: 11px;
  font-weight: 760;
  letter-spacing: 0.1em;
}

.report-ai-panel h2,
.report-ai-panel h3 {
  margin: 4px 0 0;
  color: var(--ks-ink);
}

.report-ai-panel h2 { font-size: 18px; }
.report-ai-panel h3 { font-size: 14px; }

.report-ai-panel__close,
.report-ai-panel__discard {
  color: var(--ks-muted);
  background: transparent;
  border: 0;
  cursor: pointer;
}

.report-ai-panel__note,
.report-ai-panel__hint {
  color: var(--ks-muted);
  font-size: 12px;
  line-height: 1.6;
}

.report-ai-panel__label {
  display: block;
  margin: 13px 0 5px;
  color: var(--ks-text);
  font-size: 12px;
  font-weight: 650;
}

.report-ai-panel__instruction,
.report-ai-panel__draft,
.report-ai-panel__outline-item textarea,
.report-ai-panel__outline-item input {
  box-sizing: border-box;
  width: 100%;
  color: var(--ks-text);
  background: var(--ks-surface-subtle);
  border: 1px solid var(--ks-border);
  border-radius: 7px;
  outline: 0;
}

.report-ai-panel__instruction { min-height: 62px; padding: 8px; resize: vertical; }
.report-ai-panel__instruction:focus,
.report-ai-panel__draft:focus,
.report-ai-panel__outline-item textarea:focus,
.report-ai-panel__outline-item input:focus { border-color: var(--ks-accent); }

.report-ai-panel__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 11px 0;
}

.report-ai-panel__actions button,
.report-ai-panel__accept {
  min-height: 31px;
  padding: 0 9px;
  color: var(--ks-surface);
  font-size: 12px;
  background: var(--ks-accent);
  border: 1px solid var(--ks-accent);
  border-radius: 7px;
  cursor: pointer;
}

.report-ai-panel__actions button:disabled,
.report-ai-panel__accept:disabled { cursor: not-allowed; opacity: 0.55; }

.report-ai-panel__error { color: var(--ks-danger); font-size: 12px; }
.report-ai-panel__preview { padding-top: 9px; border-top: 1px solid var(--ks-border); }
.report-ai-panel__outline-item { display: grid; gap: 6px; margin-top: 10px; }
.report-ai-panel__outline-item input { min-height: 32px; padding: 6px 8px; font-weight: 650; }
.report-ai-panel__outline-item textarea { min-height: 52px; padding: 7px 8px; resize: vertical; }
.report-ai-panel__draft { min-height: 220px; padding: 10px; margin-top: 10px; line-height: 1.7; resize: vertical; }
.report-ai-panel__citations { padding-left: 20px; color: var(--ks-muted); font-size: 11px; line-height: 1.8; }
.report-ai-panel__footer { padding-top: 12px; margin-top: 12px; border-top: 1px solid var(--ks-border); }

@media (prefers-reduced-motion: reduce) { .report-ai-panel * { scroll-behavior: auto; } }
</style>
