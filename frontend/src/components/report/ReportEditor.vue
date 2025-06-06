<script setup lang="ts">
defineProps<{
  sectionTitle: string;
  content: string;
  preview: boolean;
  saveState: "idle" | "saving" | "saved" | "error";
  disabled: boolean;
}>();

const emit = defineEmits<{
  "update:sectionTitle": [value: string];
  "update:content": [value: string];
  save: [];
  togglePreview: [];
}>();
</script>

<template>
  <section class="report-editor">
    <header class="report-editor__toolbar">
      <div>
        <span class="report-editor__eyebrow">当前章节</span>
        <div
          class="report-editor__status"
          role="status"
        >
          <span v-if="saveState === 'saving'">正在保存…</span>
          <span v-else-if="saveState === 'saved'">已保存</span>
          <span v-else-if="saveState === 'error'">保存失败</span>
          <span v-else>未保存的编辑</span>
        </div>
      </div>
      <div class="report-editor__actions">
        <button
          type="button"
          class="report-editor__preview"
          :disabled="disabled"
          @click="emit('togglePreview')"
        >
          {{ preview ? "返回编辑" : "预览报告" }}
        </button>
        <button
          v-if="!preview"
          type="button"
          class="report-editor__save"
          :disabled="disabled || saveState === 'saving'"
          @click="emit('save')"
        >
          保存
        </button>
      </div>
    </header>

    <template v-if="preview">
      <div class="report-editor__preview-sheet">
        <h2>{{ sectionTitle || "未命名章节" }}</h2>
        <p v-if="content.trim()">
          {{ content }}
        </p>
        <p
          v-else
          class="report-editor__preview-empty"
        >
          当前章节还没有正文。
        </p>
      </div>
    </template>
    <template v-else>
      <input
        class="report-editor__section-title"
        :value="sectionTitle"
        :disabled="disabled"
        aria-label="章节标题"
        placeholder="章节标题"
        @input="emit('update:sectionTitle', ($event.target as HTMLInputElement).value)"
      >
      <textarea
        class="report-editor__textarea"
        :value="content"
        :disabled="disabled"
        aria-label="章节正文"
        placeholder="从这里开始写作。可以将右侧的证据或分析结果插入到当前章节。"
        @input="emit('update:content', ($event.target as HTMLTextAreaElement).value)"
      />
      <p class="report-editor__hint">
        支持纯文本编辑，后续可以继续接入 Markdown 或导出格式。
      </p>
    </template>
  </section>
</template>

<style scoped>
.report-editor {
  display: flex;
  min-width: 0;
  min-height: 540px;
  flex: 1;
  flex-direction: column;
  padding: 26px clamp(24px, 4vw, 54px);
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-lg);
  box-shadow: var(--ks-shadow-sm);
}

.report-editor__toolbar {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
  padding-bottom: 17px;
  border-bottom: 1px solid var(--ks-border);
}

.report-editor__eyebrow {
  color: var(--ks-accent);
  font-size: 11px;
  font-weight: 760;
  letter-spacing: 0.1em;
}

.report-editor__status {
  margin-top: 5px;
  color: var(--ks-muted);
  font-size: 12px;
}

.report-editor__actions {
  display: flex;
  gap: 8px;
}

.report-editor__preview,
.report-editor__save {
  min-height: 34px;
  padding: 0 12px;
  font-size: 12px;
  font-weight: 680;
  border-radius: 7px;
  cursor: pointer;
}

.report-editor__preview {
  color: var(--ks-text);
  background: var(--ks-surface-subtle);
  border: 1px solid var(--ks-border);
}

.report-editor__save {
  color: var(--ks-surface);
  background: var(--ks-accent);
  border: 1px solid var(--ks-accent);
}

.report-editor__preview:disabled,
.report-editor__save:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.report-editor__section-title {
  width: 100%;
  padding: 24px 0 12px;
  color: var(--ks-ink);
  font-size: 28px;
  font-weight: 720;
  letter-spacing: -0.05em;
  background: transparent;
  border: 0;
  border-bottom: 1px solid var(--ks-border);
  outline: 0;
}

.report-editor__section-title:focus {
  border-color: var(--ks-accent);
}

.report-editor__textarea {
  min-height: 360px;
  flex: 1;
  padding: 24px 0;
  color: var(--ks-text);
  font-size: 15px;
  line-height: 1.9;
  resize: vertical;
  background: transparent;
  border: 0;
  outline: 0;
}

.report-editor__textarea:focus {
  box-shadow: inset 0 -2px var(--ks-accent);
}

.report-editor__hint {
  margin: 0;
  color: var(--ks-faint);
  font-size: 11px;
}

.report-editor__preview-sheet {
  max-width: 720px;
  padding: 42px 24px 60px;
  margin: 18px auto 0;
  background: #fff;
  border: 1px solid var(--ks-border);
  box-shadow: var(--ks-shadow-sm);
}

.report-editor__preview-sheet h2 {
  margin: 0 0 28px;
  color: var(--ks-ink);
  font-size: 29px;
  letter-spacing: -0.05em;
}

.report-editor__preview-sheet p {
  margin: 0;
  color: var(--ks-text);
  font-size: 15px;
  line-height: 2;
  white-space: pre-wrap;
}

.report-editor__preview-empty {
  color: var(--ks-muted) !important;
}

@media (max-width: 980px) {
  .report-editor {
    min-height: 500px;
    padding: 22px;
  }
}
</style>
