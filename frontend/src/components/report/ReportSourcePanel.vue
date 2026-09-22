<script setup lang="ts">
import { computed, ref } from "vue";

import type {
  ChatBIAnalysisResult,
  RAGCitation,
  ReportChartSpec,
  ReportSourceReference,
} from "../../api/types";
import EvidenceCard from "../chat/EvidenceCard.vue";
import ResultChart from "../analysis/ResultChart.vue";
import ResultTable from "../analysis/ResultTable.vue";
import { inferAnalysisChart } from "../analysis/chartInference";
import ReportSourceCard from "./ReportSourceCard.vue";

const props = defineProps<{
  knowledgeBaseId: string | null;
  datasourceId: string | null;
  documentTitles: Record<string, string>;
  references: ReportSourceReference[];
  ragAnswer: string;
  ragCitations: RAGCitation[];
  ragLoading: boolean;
  ragError: string | null;
  chatBIResult: ChatBIAnalysisResult | null;
  chatBILoading: boolean;
  chatBIError: string | null;
  disabled: boolean;
}>();

const emit = defineEmits<{
  searchRag: [question: string];
  insertRag: [citation: RAGCitation];
  askChatBI: [question: string];
  insertChatBI: [question: string, chartSpec: ReportChartSpec | null];
  removeReference: [id: string];
}>();

const activeTab = ref<"rag" | "chatbi" | "references">("rag");
const ragQuestion = ref("");
const chatBIQuestion = ref("");
const lastChatBIQuestion = ref("");
const chart = computed(() =>
  props.chatBIResult === null ? null : inferAnalysisChart(props.chatBIResult),
);
const sourceCount = computed(() => props.references.length);

function submitRag(): void {
  const question = ragQuestion.value.trim();
  if (question && props.knowledgeBaseId !== null && !props.ragLoading) {
    emit("searchRag", question);
  }
}

function submitChatBI(): void {
  const question = chatBIQuestion.value.trim();
  if (question && props.datasourceId !== null && !props.chatBILoading) {
    lastChatBIQuestion.value = question;
    emit("askChatBI", question);
  }
}

function documentTitle(citation: RAGCitation): string | null {
  return props.documentTitles[citation.document_id] ?? null;
}

function insertCurrentChatBI(): void {
  const question = lastChatBIQuestion.value;
  if (!question || props.chatBIResult === null) {
    return;
  }
  const currentChart = chart.value;
  emit(
    "insertChatBI",
    question,
    currentChart === null
      ? null
      : {
          kind: currentChart.kind,
          category_column: currentChart.categoryColumn,
          value_column: currentChart.valueColumn,
          points: currentChart.points,
        },
  );
}
</script>

<template>
  <aside class="report-source-panel">
    <header class="report-source-panel__header">
      <div>
        <span class="report-source-panel__eyebrow">来源</span>
        <h2>把资料放进报告</h2>
      </div>
      <span class="report-source-panel__count">{{ sourceCount }}</span>
    </header>

    <div
      class="report-source-panel__tabs"
      role="tablist"
      aria-label="报告来源"
    >
      <button
        type="button"
        role="tab"
        :aria-selected="activeTab === 'rag'"
        :class="{ 'is-active': activeTab === 'rag' }"
        @click="activeTab = 'rag'"
      >
        知识库证据
      </button>
      <button
        type="button"
        role="tab"
        :aria-selected="activeTab === 'chatbi'"
        :class="{ 'is-active': activeTab === 'chatbi' }"
        @click="activeTab = 'chatbi'"
      >
        数据分析
      </button>
      <button
        type="button"
        role="tab"
        :aria-selected="activeTab === 'references'"
        :class="{ 'is-active': activeTab === 'references' }"
        @click="activeTab = 'references'"
      >
        已引用 {{ sourceCount }}
      </button>
    </div>

    <div
      v-if="activeTab === 'rag'"
      class="report-source-panel__body"
    >
      <p
        v-if="knowledgeBaseId === null"
        class="report-source-panel__empty"
      >
        请先为报告选择知识库。
      </p>
      <template v-else>
        <form
          class="report-source-panel__form"
          @submit.prevent="submitRag"
        >
          <label for="report-rag-question">研究问题</label>
          <textarea
            id="report-rag-question"
            v-model="ragQuestion"
            rows="3"
            :disabled="disabled || ragLoading"
            placeholder="例如：总结文档中关于质量控制的主要观点"
          />
          <button
            type="submit"
            :disabled="disabled || ragLoading || !ragQuestion.trim()"
          >
            {{ ragLoading ? "正在查找…" : "查找证据" }}
          </button>
        </form>
        <p
          v-if="ragError"
          class="report-source-panel__error"
          role="alert"
        >
          {{ ragError }}
        </p>
        <p
          v-if="ragAnswer"
          class="report-source-panel__answer"
        >
          {{ ragAnswer }}
        </p>
        <div
          v-if="ragCitations.length > 0"
          class="report-source-panel__results"
        >
          <div
            v-for="citation in ragCitations"
            :key="citation.marker"
            class="report-source-panel__rag-result"
          >
            <EvidenceCard
              :citation="citation"
              :selected="false"
              :document-title="documentTitle(citation)"
            />
            <button
              type="button"
              class="report-source-panel__insert"
              :disabled="disabled"
              @click="emit('insertRag', citation)"
            >
              插入到当前章节
            </button>
          </div>
        </div>
        <p
          v-else-if="ragAnswer && !ragLoading"
          class="report-source-panel__empty"
        >
          这次回答没有返回可插入的证据。
        </p>
      </template>
    </div>

    <div
      v-else-if="activeTab === 'chatbi'"
      class="report-source-panel__body"
    >
      <p
        v-if="datasourceId === null"
        class="report-source-panel__empty"
      >
        请先为报告选择数据源。
      </p>
      <template v-else>
        <form
          class="report-source-panel__form"
          @submit.prevent="submitChatBI"
        >
          <label for="report-chatbi-question">分析问题</label>
          <textarea
            id="report-chatbi-question"
            v-model="chatBIQuestion"
            rows="3"
            :disabled="disabled || chatBILoading"
            placeholder="例如：统计每个地区的客户数量"
          />
          <button
            type="submit"
            :disabled="disabled || chatBILoading || !chatBIQuestion.trim()"
          >
            {{ chatBILoading ? "正在分析…" : "开始分析" }}
          </button>
        </form>
        <p
          v-if="chatBIError"
          class="report-source-panel__error"
          role="alert"
        >
          {{ chatBIError }}
        </p>
        <template v-if="chatBIResult">
          <p
            v-if="chatBIResult.answer"
            class="report-source-panel__answer"
          >
            {{ chatBIResult.answer }}
          </p>
          <ResultChart
            v-if="chart"
            :spec="chart"
          />
          <ResultTable
            :columns="chatBIResult.columns"
            :rows="chatBIResult.rows"
          />
          <button
            type="button"
            class="report-source-panel__insert"
            :disabled="disabled || chatBIResult.execution_status !== 'succeeded'"
            @click="insertCurrentChatBI"
          >
            插入当前分析结果
          </button>
        </template>
      </template>
    </div>

    <div
      v-else
      class="report-source-panel__body report-source-panel__references"
    >
      <p
        v-if="references.length === 0"
        class="report-source-panel__empty"
      >
        插入的证据和分析结果会出现在这里。
      </p>
      <ReportSourceCard
        v-for="source in references"
        :key="source.id"
        :source="source"
        :removable="!disabled"
        @remove="emit('removeReference', source.id)"
      />
    </div>
  </aside>
</template>

<style scoped>
.report-source-panel {
  display: flex;
  min-width: 0;
  flex-direction: column;
  padding: 18px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-lg);
}

.report-source-panel__header,
.report-source-panel__tabs {
  display: flex;
  align-items: center;
}

.report-source-panel__header {
  justify-content: space-between;
  gap: 10px;
  padding-bottom: 14px;
}

.report-source-panel__eyebrow {
  color: var(--ks-accent);
  font-size: 11px;
  font-weight: 760;
  letter-spacing: 0.1em;
}

.report-source-panel h2 {
  margin: 4px 0 0;
  color: var(--ks-ink);
  font-size: 17px;
  letter-spacing: -0.03em;
}

.report-source-panel__count {
  display: grid;
  width: 26px;
  height: 26px;
  place-items: center;
  color: var(--ks-accent-strong);
  font-size: 11px;
  background: var(--ks-accent-soft);
  border-radius: 7px;
}

.report-source-panel__tabs {
  gap: 4px;
  padding: 4px;
  background: var(--ks-surface-subtle);
  border-radius: 8px;
}

.report-source-panel__tabs button {
  min-width: 0;
  flex: 1;
  padding: 7px 4px;
  color: var(--ks-muted);
  font-size: 11px;
  font-weight: 680;
  background: transparent;
  border: 0;
  border-radius: 6px;
  cursor: pointer;
}

.report-source-panel__tabs button.is-active {
  color: var(--ks-accent-strong);
  background: var(--ks-surface);
  box-shadow: var(--ks-shadow-sm);
}

.report-source-panel__body {
  display: flex;
  min-height: 0;
  flex-direction: column;
  gap: 13px;
  padding-top: 15px;
  overflow-y: auto;
}

.report-source-panel__form {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.report-source-panel__form label {
  color: var(--ks-text);
  font-size: 12px;
  font-weight: 680;
}

.report-source-panel__form textarea {
  width: 100%;
  min-height: 72px;
  padding: 9px;
  color: var(--ks-text);
  font-size: 12px;
  line-height: 1.55;
  resize: vertical;
  background: var(--ks-surface-subtle);
  border: 1px solid var(--ks-border);
  border-radius: 8px;
}

.report-source-panel__form textarea:focus {
  border-color: var(--ks-accent);
  outline: 2px solid rgb(55 116 102 / 12%);
}

.report-source-panel__form button,
.report-source-panel__insert {
  min-height: 34px;
  padding: 0 10px;
  color: var(--ks-surface);
  font-size: 12px;
  font-weight: 680;
  background: var(--ks-accent);
  border: 1px solid var(--ks-accent);
  border-radius: 7px;
  cursor: pointer;
}

.report-source-panel__form button:disabled,
.report-source-panel__insert:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.report-source-panel__error {
  padding: 9px;
  margin: 0;
  color: var(--ks-danger);
  font-size: 12px;
  line-height: 1.5;
  background: #fff1ef;
  border-radius: 7px;
}

.report-source-panel__answer {
  padding: 10px;
  margin: 0;
  color: var(--ks-text);
  font-size: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
  background: var(--ks-accent-soft);
  border-radius: 7px;
}

.report-source-panel__results,
.report-source-panel__references {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.report-source-panel__rag-result {
  display: flex;
  flex-direction: column;
  gap: 7px;
}

.report-source-panel__rag-result :deep(.evidence-card) {
  cursor: default;
}

.report-source-panel__insert {
  width: 100%;
}

.report-source-panel__empty {
  padding: 18px 4px;
  margin: 0;
  color: var(--ks-muted);
  font-size: 12px;
  line-height: 1.65;
  text-align: center;
}

.report-source-panel__references {
  overflow-y: auto;
}

.report-source-panel :deep(.result-table-wrap) {
  max-height: 220px;
}

.report-source-panel :deep(.result-table) {
  min-width: 360px;
  font-size: 11px;
}

.report-source-panel :deep(.result-table th),
.report-source-panel :deep(.result-table td) {
  padding: 8px;
}

.report-source-panel :deep(.result-chart) {
  padding: 10px;
}

.report-source-panel :deep(.result-chart__plot svg) {
  height: 150px;
}
</style>
