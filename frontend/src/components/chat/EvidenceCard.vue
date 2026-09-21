<script setup lang="ts">
import { computed } from "vue";

import type { RAGCitation } from "../../api/types";

const props = defineProps<{
  citation: RAGCitation;
  selected: boolean;
  documentTitle: string | null;
}>();

const emit = defineEmits<{ select: [] }>();

const modalityLabel = computed(() => {
  if (props.citation.candidate_kind === "chunk") {
    return "文本片段";
  }
  return {
    text: "文本证据",
    table: "表格证据",
    image: "图片证据",
    formula: "公式证据",
  }[props.citation.modality ?? "text"];
});

const pageLabel = computed(() =>
  props.citation.page_start === props.citation.page_end
    ? `第 ${props.citation.page_start} 页`
    : `第 ${props.citation.page_start}–${props.citation.page_end} 页`,
);

const sectionLabel = computed(() => props.citation.section_path.join(" / "));
const branchLabel = computed(() => {
  const branches = props.citation.branch_provenance.map((item) => item.branch);
  return branches.length > 0 ? branches.join(" + ") : "资料检索";
});
</script>

<template>
  <button
    class="evidence-card"
    :class="{ 'is-selected': selected }"
    type="button"
    @click="emit('select')"
  >
    <div class="evidence-card__topline">
      <span class="evidence-card__marker">{{ citation.marker }}</span>
      <span class="evidence-card__type">{{ modalityLabel }}</span>
      <span class="evidence-card__branch">{{ branchLabel }}</span>
    </div>
    <strong class="evidence-card__title">
      {{ documentTitle || "当前知识库文档" }}
    </strong>
    <span class="evidence-card__section-title">
      {{ citation.section_title || "未命名章节" }}
    </span>
    <div class="evidence-card__meta">
      <span>{{ pageLabel }}</span>
      <span>{{ citation.source_block_ids.length }} 个来源块</span>
    </div>
    <p
      v-if="sectionLabel"
      class="evidence-card__section"
    >
      {{ sectionLabel }}
    </p>
    <p class="evidence-card__note">
      <template v-if="citation.snippet">
        <span class="evidence-card__snippet-label">
          {{ citation.snippet_kind === "representation" ? "检索表示" : "原始片段" }}
        </span>
        <span class="evidence-card__snippet">{{ citation.snippet }}</span>
        <span v-if="citation.snippet_kind === 'representation'">
          检索表示不替代原始来源。
        </span>
      </template>
      <template v-else-if="citation.candidate_kind === 'evidence'">
        该卡片锚定原始 Evidence；当前接口未提供可展示的表示片段。
      </template>
      <template v-else>
        当前接口未提供可展示的原始文本片段。
      </template>
    </p>
    <div
      v-if="citation.candidate_kind === 'evidence'"
      class="evidence-card__detail"
    >
      <span v-if="citation.asset_refs.length > 0">含原始资产引用</span>
      <span v-if="citation.representation_ids.length > 0">
        {{ citation.representation_ids.length }} 个检索表示
      </span>
    </div>
  </button>
</template>

<style scoped>
.evidence-card {
  display: block;
  width: 100%;
  padding: 14px;
  color: var(--ks-text);
  text-align: left;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-md);
  cursor: pointer;
}

.evidence-card:hover,
.evidence-card.is-selected {
  border-color: var(--ks-accent);
  box-shadow: 0 4px 12px rgb(55 116 102 / 8%);
}

.evidence-card.is-selected {
  background: #fbfdfb;
}

.evidence-card__topline,
.evidence-card__meta,
.evidence-card__detail {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 7px;
}

.evidence-card__topline {
  margin-bottom: 10px;
}

.evidence-card__marker {
  padding: 2px 5px;
  color: var(--ks-accent-strong);
  font-size: 11px;
  font-weight: 740;
  background: var(--ks-accent-soft);
  border-radius: 4px;
}

.evidence-card__type {
  color: var(--ks-text);
  font-size: 11px;
  font-weight: 680;
}

.evidence-card__branch {
  margin-left: auto;
  color: var(--ks-faint);
  font-size: 10px;
}

.evidence-card__title {
  display: block;
  overflow: hidden;
  color: var(--ks-ink);
  font-size: 13px;
  font-weight: 680;
  line-height: 1.5;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.evidence-card__section-title {
  display: block;
  margin-top: 3px;
  overflow: hidden;
  color: var(--ks-text);
  font-size: 11px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.evidence-card__meta,
.evidence-card__section,
.evidence-card__note,
.evidence-card__detail {
  color: var(--ks-muted);
  font-size: 11px;
}

.evidence-card__meta {
  margin-top: 8px;
}

.evidence-card__section {
  margin: 7px 0 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.evidence-card__note {
  margin: 10px 0 0;
  line-height: 1.55;
}

.evidence-card__snippet-label {
  display: block;
  margin-bottom: 4px;
  color: var(--ks-accent-strong);
  font-weight: 680;
}

.evidence-card__snippet {
  display: -webkit-box;
  max-height: 74px;
  overflow: hidden;
  color: var(--ks-text);
  line-height: 1.55;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 3;
}

.evidence-card__detail {
  margin-top: 8px;
  color: var(--ks-accent-strong);
}
</style>
