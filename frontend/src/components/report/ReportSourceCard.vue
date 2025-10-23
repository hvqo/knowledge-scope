<script setup lang="ts">
import type { ReportSourceReference } from "../../api/types";

defineProps<{
  source: ReportSourceReference;
  removable?: boolean;
}>();

const emit = defineEmits<{ remove: [] }>();

function pageLabel(source: ReportSourceReference): string {
  if (source.page_start === null) {
    return "";
  }
  return source.page_start === source.page_end
    ? `第 ${source.page_start} 页`
    : `第 ${source.page_start}–${source.page_end} 页`;
}
</script>

<template>
  <article class="report-source-card">
    <header class="report-source-card__header">
      <div>
        <span class="report-source-card__kind">
          {{ source.kind === "rag_evidence" ? "知识库证据" : "数据分析" }}
        </span>
        <h3>{{ source.title }}</h3>
      </div>
      <button
        v-if="removable"
        type="button"
        class="report-source-card__remove"
        aria-label="移除来源"
        @click="emit('remove')"
      >
        移除
      </button>
    </header>

    <template v-if="source.kind === 'rag_evidence'">
      <div class="report-source-card__meta">
        <span>{{ source.document_title || "来源文档" }}</span>
        <span v-if="pageLabel(source)">{{ pageLabel(source) }}</span>
        <span v-if="source.evidence_type">{{ source.evidence_type }}</span>
      </div>
      <p class="report-source-card__snippet">
        {{ source.snippet || "当前来源没有可展示的片段。" }}
      </p>
      <p
        v-if="source.section_path.length > 0"
        class="report-source-card__path"
      >
        {{ source.section_path.join(" / ") }}
      </p>
    </template>
    <template v-else>
      <div class="report-source-card__meta">
        <span>{{ source.document_title || "业务数据源" }}</span>
        <span v-if="source.row_count !== null">{{ source.row_count }} 行</span>
      </div>
      <p
        v-if="source.question"
        class="report-source-card__question"
      >
        {{ source.question }}
      </p>
      <p
        v-if="source.answer"
        class="report-source-card__snippet"
      >
        {{ source.answer }}
      </p>
      <div
        v-if="source.columns.length > 0"
        class="report-source-card__table-summary"
      >
        已保存 {{ source.columns.length }} 列结果，可在预览中继续查看。
      </div>
    </template>
  </article>
</template>

<style scoped>
.report-source-card {
  padding: 14px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: 10px;
}

.report-source-card__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
}

.report-source-card__kind {
  color: var(--ks-accent);
  font-size: 10px;
  font-weight: 760;
  letter-spacing: 0.08em;
}

.report-source-card h3 {
  margin: 4px 0 0;
  color: var(--ks-ink);
  font-size: 13px;
  line-height: 1.45;
}

.report-source-card__remove {
  flex: 0 0 auto;
  padding: 3px 0;
  color: var(--ks-danger);
  font-size: 11px;
  background: transparent;
  border: 0;
  cursor: pointer;
}

.report-source-card__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 10px;
  color: var(--ks-muted);
  font-size: 11px;
}

.report-source-card__meta span {
  padding: 3px 6px;
  background: var(--ks-surface-subtle);
  border-radius: 5px;
}

.report-source-card__snippet,
.report-source-card__question,
.report-source-card__path,
.report-source-card__table-summary {
  margin: 10px 0 0;
  color: var(--ks-text);
  font-size: 12px;
  line-height: 1.65;
}

.report-source-card__snippet {
  white-space: pre-wrap;
}

.report-source-card__question {
  color: var(--ks-ink);
  font-weight: 680;
}

.report-source-card__path,
.report-source-card__table-summary {
  color: var(--ks-muted);
  font-size: 11px;
}
</style>
