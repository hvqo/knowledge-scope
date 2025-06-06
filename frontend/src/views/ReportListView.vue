<script setup lang="ts">
import { useMutation, useQuery } from "@tanstack/vue-query";
import { computed, ref } from "vue";
import { useRouter } from "vue-router";

import {
  createReport,
  fetchChatBIDataSources,
  fetchKnowledgeBases,
  fetchReports,
  getUserFacingError,
} from "../api/client";
import type { ReportCreateRequest } from "../api/types";

const router = useRouter();
const title = ref("");
const knowledgeBaseId = ref("");
const datasourceId = ref("");

const reportsQuery = useQuery({
  queryKey: ["reports"],
  queryFn: () => fetchReports({ limit: 100 }),
});
const knowledgeBasesQuery = useQuery({
  queryKey: ["report-knowledge-bases"],
  queryFn: () => fetchKnowledgeBases({ limit: 100 }),
});
const dataSourcesQuery = useQuery({
  queryKey: ["report-data-sources"],
  queryFn: () => fetchChatBIDataSources({ limit: 100 }),
});

const reports = computed(() => reportsQuery.data.value?.items ?? []);
const knowledgeBases = computed(() => knowledgeBasesQuery.data.value?.items ?? []);
const dataSources = computed(() => dataSourcesQuery.data.value?.items ?? []);
const isLoading = computed(
  () => reportsQuery.isPending.value || knowledgeBasesQuery.isPending.value || dataSourcesQuery.isPending.value,
);
const loadError = computed(() =>
  reportsQuery.error.value || knowledgeBasesQuery.error.value || dataSourcesQuery.error.value,
);

const createMutation = useMutation({
  mutationFn: (payload: ReportCreateRequest) => createReport(payload),
  onSuccess: (report) => {
    void router.push({ name: "report-workspace", params: { id: report.id } });
  },
});

function submit(): void {
  const normalizedTitle = title.value.trim();
  if (!normalizedTitle || createMutation.isPending.value) {
    return;
  }
  createMutation.mutate({
    title: normalizedTitle,
    knowledge_base_id: knowledgeBaseId.value || null,
    datasource_id: datasourceId.value || null,
  });
}

function openReport(id: string): void {
  void router.push({ name: "report-workspace", params: { id } });
}

function retry(): void {
  void reportsQuery.refetch();
  void knowledgeBasesQuery.refetch();
  void dataSourcesQuery.refetch();
}
</script>

<template>
  <section class="report-list-page">
    <header class="report-list-page__heading">
      <div>
        <span class="report-list-page__eyebrow">工作区</span>
        <h1>报告创作</h1>
        <p>把文档证据和业务分析整理成一份可继续编辑的报告。</p>
      </div>
    </header>

    <div class="report-list-page__create">
      <div>
        <span class="report-list-page__eyebrow">新建报告</span>
        <h2>从一个清晰的标题开始</h2>
      </div>
      <form
        class="report-list-page__form"
        @submit.prevent="submit"
      >
        <input
          v-model="title"
          aria-label="报告标题"
          placeholder="例如：2026 年度质量分析报告"
          :disabled="createMutation.isPending.value"
        >
        <select
          v-model="knowledgeBaseId"
          aria-label="关联知识库"
          :disabled="createMutation.isPending.value"
        >
          <option value="">
            不关联知识库
          </option>
          <option
            v-for="knowledgeBase in knowledgeBases"
            :key="knowledgeBase.id"
            :value="knowledgeBase.id"
          >
            知识库：{{ knowledgeBase.name }}
          </option>
        </select>
        <select
          v-model="datasourceId"
          aria-label="关联数据源"
          :disabled="createMutation.isPending.value"
        >
          <option value="">
            不关联数据源
          </option>
          <option
            v-for="dataSource in dataSources"
            :key="dataSource.id"
            :value="dataSource.id"
          >
            数据源：{{ dataSource.display_name }}
          </option>
        </select>
        <button
          type="submit"
          :disabled="createMutation.isPending.value || !title.trim()"
        >
          {{ createMutation.isPending.value ? "正在创建…" : "创建报告" }}
        </button>
      </form>
      <p
        v-if="createMutation.error.value"
        class="report-list-page__error"
        role="alert"
      >
        {{ getUserFacingError(createMutation.error.value, "报告创建失败，请稍后重试。") }}
      </p>
    </div>

    <div
      v-if="isLoading"
      class="report-list-page__state"
      role="status"
    >
      正在读取报告…
    </div>
    <div
      v-else-if="loadError"
      class="report-list-page__state report-list-page__state--error"
      role="alert"
    >
      <span>{{ getUserFacingError(loadError, "报告列表暂时无法加载，请稍后重试。") }}</span>
      <button
        type="button"
        @click="retry"
      >
        重试
      </button>
    </div>
    <div
      v-else-if="reports.length === 0"
      class="report-list-page__state"
    >
      <strong>还没有报告</strong>
      <span>创建一份报告，把知识库证据和数据分析结果放在一起。</span>
    </div>
    <div
      v-else
      class="report-list-page__grid"
    >
      <button
        v-for="report in reports"
        :key="report.id"
        type="button"
        class="report-list-page__card"
        @click="openReport(report.id)"
      >
        <span class="report-list-page__card-mark">报告</span>
        <strong>{{ report.title }}</strong>
        <span>
          {{ report.knowledge_base_id ? "已关联知识库" : "未关联知识库" }}
          ·
          {{ report.datasource_id ? "已关联数据源" : "未关联数据源" }}
        </span>
        <small>最近更新 {{ new Date(report.updated_at).toLocaleDateString("zh-CN") }}</small>
      </button>
    </div>
  </section>
</template>

<style scoped>
.report-list-page {
  display: flex;
  min-height: calc(100vh - 104px);
  flex-direction: column;
  gap: 24px;
}

.report-list-page__heading {
  padding: 10px 2px 0;
}

.report-list-page__eyebrow {
  color: var(--ks-accent);
  font-size: 11px;
  font-weight: 760;
  letter-spacing: 0.1em;
}

.report-list-page h1 {
  margin: 6px 0 0;
  color: var(--ks-ink);
  font-size: clamp(28px, 4vw, 38px);
  letter-spacing: -0.06em;
}

.report-list-page__heading p {
  margin: 8px 0 0;
  color: var(--ks-muted);
  font-size: 14px;
}

.report-list-page__create {
  display: grid;
  grid-template-columns: minmax(220px, 0.7fr) minmax(0, 1.3fr);
  align-items: center;
  gap: 24px;
  padding: 20px 22px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-lg);
}

.report-list-page__create h2 {
  margin: 6px 0 0;
  color: var(--ks-ink);
  font-size: 17px;
  letter-spacing: -0.03em;
}

.report-list-page__form {
  display: grid;
  grid-template-columns: minmax(180px, 1.3fr) repeat(2, minmax(150px, 1fr)) auto;
  gap: 8px;
}

.report-list-page__form input,
.report-list-page__form select {
  min-width: 0;
  min-height: 38px;
  padding: 0 10px;
  color: var(--ks-text);
  background: var(--ks-surface-subtle);
  border: 1px solid var(--ks-border);
  border-radius: 8px;
}

.report-list-page__form input:focus,
.report-list-page__form select:focus {
  border-color: var(--ks-accent);
  outline: 2px solid rgb(55 116 102 / 12%);
}

.report-list-page__form button,
.report-list-page__state button {
  min-height: 38px;
  padding: 0 13px;
  color: var(--ks-surface);
  font-size: 12px;
  font-weight: 680;
  white-space: nowrap;
  background: var(--ks-accent);
  border: 1px solid var(--ks-accent);
  border-radius: 8px;
  cursor: pointer;
}

.report-list-page__form button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.report-list-page__error {
  grid-column: 1 / -1;
  margin: 0;
  color: var(--ks-danger);
  font-size: 12px;
}

.report-list-page__state {
  display: flex;
  min-height: 240px;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 8px;
  color: var(--ks-muted);
  text-align: center;
  background: var(--ks-surface);
  border: 1px dashed var(--ks-border-strong);
  border-radius: var(--ks-radius-lg);
}

.report-list-page__state strong {
  color: var(--ks-ink);
  font-size: 18px;
}

.report-list-page__state--error {
  color: var(--ks-danger);
}

.report-list-page__state--error button {
  margin-top: 5px;
}

.report-list-page__grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 14px;
}

.report-list-page__card {
  display: flex;
  min-height: 160px;
  align-items: flex-start;
  flex-direction: column;
  gap: 9px;
  padding: 18px;
  color: var(--ks-text);
  text-align: left;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-lg);
  cursor: pointer;
}

.report-list-page__card:hover {
  border-color: var(--ks-accent);
  box-shadow: var(--ks-shadow-md);
}

.report-list-page__card-mark {
  padding: 4px 7px;
  color: var(--ks-accent-strong);
  font-size: 10px;
  font-weight: 760;
  background: var(--ks-accent-soft);
  border-radius: 5px;
}

.report-list-page__card strong {
  color: var(--ks-ink);
  font-size: 16px;
}

.report-list-page__card span:not(.report-list-page__card-mark),
.report-list-page__card small {
  color: var(--ks-muted);
  font-size: 12px;
}

.report-list-page__card small {
  margin-top: auto;
}

@media (max-width: 1120px) {
  .report-list-page__create {
    grid-template-columns: 1fr;
  }

  .report-list-page__form {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .report-list-page__form button {
    grid-column: span 2;
  }
}
</style>
