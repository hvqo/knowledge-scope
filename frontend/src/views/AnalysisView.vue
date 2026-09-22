<script setup lang="ts">
import { useQuery } from "@tanstack/vue-query";
import { computed, onBeforeUnmount, ref, watch } from "vue";

import { askChatBI, fetchChatBIDataSources, getUserFacingError } from "../api/client";
import type { ChatBIDataSource } from "../api/types";
import AnalysisComposer from "../components/analysis/AnalysisComposer.vue";
import AnalysisResult from "../components/analysis/AnalysisResult.vue";
import AnalysisSidebar from "../components/analysis/AnalysisSidebar.vue";
import { useAnalysisStore } from "../stores/analysis";

const analysisStore = useAnalysisStore();
const activeController = ref<AbortController | null>(null);
const activeRun = ref(0);
const isUnmounting = ref(false);

const dataSourcesQuery = useQuery({
  queryKey: ["chatbi-data-sources"],
  queryFn: () => fetchChatBIDataSources({ limit: 100 }),
});

const dataSources = computed(() => dataSourcesQuery.data.value?.items ?? []);
const availableDataSources = computed(() => dataSources.value.filter((item) => item.enabled));
const dataSourcesLoading = computed(() => dataSourcesQuery.isLoading.value);
const dataSourcesError = computed(() => dataSourcesQuery.isError.value);
const activeSession = computed(() => analysisStore.activeSession);
const activeResult = computed(() => activeSession.value?.result ?? null);
const isLoading = computed(() => activeSession.value?.status === "loading");
const selectedDataSource = computed<ChatBIDataSource | null>(() =>
  availableDataSources.value.find((item) => item.id === analysisStore.selectedDatasourceId) ?? null,
);
const dataSourcesErrorMessage = computed(() =>
  getUserFacingError(dataSourcesQuery.error.value, "暂时无法读取数据源，请稍后重试。"),
);

watch(
  availableDataSources,
  (items) => {
    const selected = analysisStore.selectedDatasourceId;
    if (items.length === 0) {
      analysisStore.setDatasource(null);
      return;
    }
    if (selected === null || !items.some((item) => item.id === selected)) {
      analysisStore.setDatasource(items[0].id);
    }
  },
  { immediate: true },
);

function selectDatasource(id: string | null): void {
  if (isLoading.value) {
    return;
  }
  const changed = analysisStore.selectedDatasourceId !== id;
  const hasQuestion = (activeSession.value?.question.trim().length ?? 0) > 0;
  analysisStore.setDatasource(id);
  if (changed && hasQuestion) {
    analysisStore.newAnalysis(id);
  }
}

function newAnalysis(): void {
  if (!isLoading.value) {
    analysisStore.newAnalysis();
  }
}

function selectSession(id: string): void {
  if (!isLoading.value) {
    analysisStore.selectSession(id);
  }
}

function stopAnalysis(): void {
  if (activeController.value === null) {
    return;
  }
  activeRun.value += 1;
  activeController.value.abort();
  const session = activeSession.value;
  if (session !== null) {
    analysisStore.updateSession(session.id, (item) => {
      item.status = "error";
      item.errorMessage = "已停止本次分析。";
    });
  }
  activeController.value = null;
}

async function sendQuestion(question: string): Promise<void> {
  const datasourceId = analysisStore.selectedDatasourceId;
  if (!datasourceId || !question.trim() || isLoading.value) {
    return;
  }
  const session = analysisStore.ensureActiveSession();
  const sessionId = session.id;
  analysisStore.updateSession(sessionId, (item) => {
    item.datasourceId = datasourceId;
    item.question = question.trim();
    item.status = "loading";
    item.result = null;
    item.errorMessage = null;
  });
  const controller = new AbortController();
  activeController.value = controller;
  const run = ++activeRun.value;
  try {
    const result = await askChatBI(datasourceId, question, controller.signal);
    if (run !== activeRun.value || isUnmounting.value) {
      return;
    }
    analysisStore.updateSession(sessionId, (item) => {
      item.result = result;
      item.status = "complete";
      item.errorMessage = null;
    });
  } catch (error) {
    if (run !== activeRun.value || isUnmounting.value || controller.signal.aborted) {
      return;
    }
    analysisStore.updateSession(sessionId, (item) => {
      item.status = "error";
      item.errorMessage = getUserFacingError(error, "分析暂时无法完成，请稍后重试。");
    });
  } finally {
    if (run === activeRun.value) {
      activeController.value = null;
    }
  }
}

function retryDataSources(): void {
  void dataSourcesQuery.refetch();
}

onBeforeUnmount(() => {
  isUnmounting.value = true;
  activeRun.value += 1;
  activeController.value?.abort();
});
</script>

<template>
  <section class="analysis-page">
    <header class="analysis-page__heading">
      <div>
        <span class="analysis-page__eyebrow">业务数据</span>
        <h1>数据分析</h1>
        <p>选择已注册的数据源，用自然语言查看结构化结果。</p>
      </div>
      <div class="analysis-page__scope">
        <span
          class="analysis-page__scope-dot"
          aria-hidden="true"
        />
        <span v-if="selectedDataSource">当前使用 {{ selectedDataSource.display_name }}</span>
        <span v-else>请选择一个可用数据源</span>
      </div>
    </header>

    <div class="analysis-workspace">
      <AnalysisSidebar
        :data-sources="availableDataSources"
        :selected-datasource-id="analysisStore.selectedDatasourceId"
        :data-sources-loading="dataSourcesLoading"
        :data-sources-error="dataSourcesError"
        :data-sources-error-message="dataSourcesErrorMessage"
        :sessions="analysisStore.sessions"
        :active-session-id="analysisStore.activeSessionId"
        :disabled="isLoading"
        @select-datasource="selectDatasource"
        @new-analysis="newAnalysis"
        @select-session="selectSession"
        @retry-data-sources="retryDataSources"
      />

      <main class="analysis-main">
        <div
          v-if="activeResult"
          class="analysis-main__result"
        >
          <AnalysisResult
            :result="activeResult"
            :datasource-name="selectedDataSource?.display_name || '业务数据源'"
          />
        </div>
        <div
          v-else-if="activeSession?.status === 'error'"
          class="analysis-main__empty analysis-main__empty--error"
          role="alert"
        >
          <span
            class="analysis-main__empty-icon"
            aria-hidden="true"
          >!</span>
          <strong>{{ activeSession.errorMessage }}</strong>
          <span>可以修改问题后重新分析。</span>
        </div>
        <div
          v-else
          class="analysis-main__empty"
        >
          <span
            class="analysis-main__empty-icon"
            aria-hidden="true"
          >↗</span>
          <strong>从一个业务问题开始</strong>
          <span>结果会以表格呈现；适合的数据也会生成简洁图表。</span>
        </div>

        <AnalysisComposer
          :disabled="isLoading || availableDataSources.length === 0"
          :loading="isLoading"
          :can-send="selectedDataSource !== null"
          @send="sendQuestion"
          @stop="stopAnalysis"
        />
      </main>
    </div>
  </section>
</template>

<style scoped>
.analysis-page {
  display: flex;
  min-height: calc(100vh - 104px);
  flex-direction: column;
  gap: 18px;
}

.analysis-page__heading {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 20px;
}

.analysis-page__eyebrow {
  color: var(--ks-accent);
  font-size: 11px;
  font-weight: 760;
  letter-spacing: 0.08em;
}

.analysis-page h1 {
  margin: 4px 0 0;
  color: var(--ks-ink);
  font-size: clamp(25px, 3vw, 32px);
  letter-spacing: -0.06em;
}

.analysis-page__heading p {
  margin: 7px 0 0;
  color: var(--ks-muted);
  font-size: 13px;
}

.analysis-page__scope {
  display: flex;
  align-items: center;
  gap: 7px;
  padding-bottom: 4px;
  color: var(--ks-muted);
  font-size: 12px;
}

.analysis-page__scope-dot {
  width: 7px;
  height: 7px;
  background: var(--ks-success);
  border-radius: 50%;
}

.analysis-workspace {
  display: grid;
  min-width: 0;
  grid-template-columns: 248px minmax(0, 1fr);
  align-items: start;
  gap: 18px;
}

.analysis-main {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 16px;
}

.analysis-main__result {
  min-width: 0;
}

.analysis-main__empty {
  display: flex;
  min-height: 300px;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 9px;
  padding: 32px;
  color: var(--ks-muted);
  text-align: center;
  background: var(--ks-surface);
  border: 1px dashed var(--ks-border-strong);
  border-radius: var(--ks-radius-lg);
}

.analysis-main__empty strong {
  color: var(--ks-ink);
  font-size: 15px;
}

.analysis-main__empty span:last-child {
  font-size: 13px;
}

.analysis-main__empty--error {
  color: var(--ks-danger);
  border-style: solid;
  border-color: #efccc8;
  background: #fffaf9;
}

.analysis-main__empty--error strong {
  color: #8d443e;
}

.analysis-main__empty-icon {
  display: grid;
  width: 42px;
  height: 42px;
  place-items: center;
  color: var(--ks-accent-strong);
  font-size: 20px;
  background: var(--ks-accent-soft);
  border-radius: 12px;
}

.analysis-main__empty--error .analysis-main__empty-icon {
  color: var(--ks-danger);
  background: #ffe7e4;
}

@media (max-width: 880px) {
  .analysis-page__heading {
    align-items: flex-start;
    flex-direction: column;
  }

  .analysis-workspace {
    grid-template-columns: 1fr;
  }
}
</style>
