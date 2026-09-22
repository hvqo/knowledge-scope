<script setup lang="ts">
import type { Report } from "../../api/types";
import ResultChart from "../analysis/ResultChart.vue";
import ResultTable from "../analysis/ResultTable.vue";
import ReportSourceCard from "./ReportSourceCard.vue";

defineProps<{ report: Report }>();
</script>

<template>
  <article class="report-preview">
    <header class="report-preview__cover">
      <span>报告预览</span>
      <h1>{{ report.title }}</h1>
      <p>由 KnowledgeScope 中的资料和业务分析结果整理。</p>
    </header>
    <section
      v-for="section in report.sections"
      :key="section.id"
      class="report-preview__section"
    >
      <div class="report-preview__section-heading">
        <span>0{{ section.position + 1 }}</span>
        <h2>{{ section.title }}</h2>
      </div>
      <p
        v-if="section.content.trim()"
        class="report-preview__content"
      >
        {{ section.content }}
      </p>
      <p
        v-else
        class="report-preview__empty"
      >
        当前章节还没有正文。
      </p>
      <div
        v-if="section.sources.length > 0"
        class="report-preview__sources"
      >
        <ReportSourceCard
          v-for="source in section.sources"
          :key="source.id"
          :source="source"
        />
        <template
          v-for="source in section.sources"
          :key="`${source.id}-detail`"
        >
          <ResultTable
            v-if="source.kind === 'chatbi_result' && source.columns.length > 0"
            :columns="source.columns"
            :rows="source.rows"
          />
          <ResultChart
            v-if="source.kind === 'chatbi_result' && source.chart_spec"
            :spec="{
              kind: source.chart_spec.kind,
              categoryColumn: source.chart_spec.category_column,
              valueColumn: source.chart_spec.value_column,
              points: source.chart_spec.points,
            }"
          />
        </template>
      </div>
    </section>
    <p
      v-if="report.sections.length === 0"
      class="report-preview__empty report-preview__empty--page"
    >
      报告还没有章节。
    </p>
  </article>
</template>

<style scoped>
.report-preview {
  max-width: 820px;
  padding: 42px clamp(24px, 5vw, 72px) 72px;
  margin: 0 auto;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  box-shadow: var(--ks-shadow-sm);
}

.report-preview__cover {
  padding-bottom: 42px;
  border-bottom: 1px solid var(--ks-border);
}

.report-preview__cover > span,
.report-preview__section-heading > span {
  color: var(--ks-accent);
  font-size: 11px;
  font-weight: 760;
  letter-spacing: 0.1em;
}

.report-preview__cover h1 {
  margin: 11px 0 12px;
  color: var(--ks-ink);
  font-size: clamp(30px, 4vw, 44px);
  letter-spacing: -0.06em;
}

.report-preview__cover p {
  margin: 0;
  color: var(--ks-muted);
  font-size: 13px;
}

.report-preview__section {
  padding: 34px 0;
  border-bottom: 1px solid var(--ks-border);
}

.report-preview__section-heading {
  display: flex;
  align-items: baseline;
  gap: 12px;
}

.report-preview__section-heading h2 {
  margin: 0;
  color: var(--ks-ink);
  font-size: 23px;
  letter-spacing: -0.04em;
}

.report-preview__content,
.report-preview__empty {
  margin: 20px 0 0;
  color: var(--ks-text);
  font-size: 15px;
  line-height: 2;
  white-space: pre-wrap;
}

.report-preview__empty {
  color: var(--ks-muted);
}

.report-preview__empty--page {
  padding: 40px 0;
  text-align: center;
}

.report-preview__sources {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding-top: 22px;
}

.report-preview__sources :deep(.result-table-wrap) {
  max-height: 360px;
}
</style>
