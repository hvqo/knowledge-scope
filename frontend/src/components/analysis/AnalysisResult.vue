<script setup lang="ts">
import { computed, ref } from "vue";

import type { ChatBIAnalysisResult } from "../../api/types";
import ResultChart from "./ResultChart.vue";
import ResultTable from "./ResultTable.vue";
import SqlDisclosure from "./SqlDisclosure.vue";
import { inferAnalysisChart } from "./chartInference";

const props = defineProps<{
  result: ChatBIAnalysisResult;
  datasourceName: string;
}>();

const viewMode = ref<"table" | "chart">("table");
const chart = computed(() => inferAnalysisChart(props.result));
const isSuccessful = computed(() => props.result.execution_status === "succeeded");
const isEligibilityTerminal = computed(() =>
  ["clarify", "refuse", "eligibility_unavailable"].includes(props.result.execution_status),
);
const statusLabel = computed(() => {
  if (props.result.execution_status === "succeeded") return "已完成";
  if (props.result.execution_status === "clarify") return "需要补充说明";
  if (props.result.execution_status === "refuse") return "无法分析";
  if (props.result.execution_status === "eligibility_unavailable") return "暂时无法判断";
  return "未完成";
});

function showChart(): void {
  if (chart.value !== null) {
    viewMode.value = "chart";
  }
}
</script>

<template>
  <section
    class="analysis-result"
    aria-live="polite"
  >
    <div class="analysis-result__meta">
      <div>
        <span class="analysis-result__eyebrow">分析结果</span>
        <h2>{{ datasourceName }}</h2>
      </div>
      <div class="analysis-result__meta-items">
        <span>{{ statusLabel }}</span>
        <span v-if="isSuccessful">{{ result.row_count }} 行</span>
        <span>{{ result.elapsed_ms.toFixed(0) }} ms</span>
      </div>
    </div>

    <div
      v-if="isEligibilityTerminal"
      class="analysis-result__decision"
      :class="`is-${result.execution_status}`"
      role="status"
    >
      <strong>{{ result.eligibility?.user_message || result.error_message || "暂时无法分析该问题。" }}</strong>
      <span v-if="result.eligibility?.clarification_question">
        {{ result.eligibility.clarification_question }}
      </span>
    </div>

    <div
      v-else-if="result.error_message"
      class="analysis-result__error"
      role="alert"
    >
      {{ result.error_message }}
    </div>

    <template v-if="isSuccessful">
      <div
        v-if="result.answer"
        class="analysis-result__answer"
      >
        <span class="analysis-result__eyebrow">结论</span>
        <p>{{ result.answer }}</p>
      </div>

      <div
        v-if="result.warnings.length > 0"
        class="analysis-result__warning"
        role="status"
      >
        <span aria-hidden="true">!</span>
        <span>{{ result.warnings.join(" ") }}</span>
      </div>

      <div class="analysis-result__viewbar">
        <div>
          <span class="analysis-result__eyebrow">明细</span>
          <strong>{{ result.row_count }} 行结果</strong>
        </div>
        <div
          v-if="chart"
          class="analysis-result__view-toggle"
          role="tablist"
          aria-label="结果视图"
        >
          <button
            type="button"
            :class="{ 'is-active': viewMode === 'table' }"
            role="tab"
            :aria-selected="viewMode === 'table'"
            @click="viewMode = 'table'"
          >
            表格
          </button>
          <button
            type="button"
            :class="{ 'is-active': viewMode === 'chart' }"
            role="tab"
            :aria-selected="viewMode === 'chart'"
            @click="showChart"
          >
            图表
          </button>
        </div>
      </div>

      <ResultChart
        v-if="chart && viewMode === 'chart'"
        :spec="chart"
      />
      <ResultTable
        v-else
        :columns="result.columns"
        :rows="result.rows"
      />
      <SqlDisclosure :sql="result.redacted_sql" />
    </template>
  </section>
</template>

<style scoped>
.analysis-result {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 22px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-lg);
  box-shadow: var(--ks-shadow-sm);
}

.analysis-result__meta,
.analysis-result__viewbar {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.analysis-result__meta h2 {
  margin: 5px 0 0;
  color: var(--ks-ink);
  font-size: 18px;
  letter-spacing: -0.03em;
}

.analysis-result__eyebrow {
  color: var(--ks-accent);
  font-size: 11px;
  font-weight: 760;
  letter-spacing: 0.08em;
}

.analysis-result__meta-items {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
  color: var(--ks-muted);
  font-size: 12px;
}

.analysis-result__meta-items span {
  padding: 5px 8px;
  background: var(--ks-surface-subtle);
  border-radius: 6px;
}

.analysis-result__answer {
  padding: 16px;
  background: var(--ks-accent-soft);
  border-radius: 10px;
}

.analysis-result__answer p {
  margin: 8px 0 0;
  color: var(--ks-ink);
  line-height: 1.7;
  white-space: pre-wrap;
}

.analysis-result__decision,
.analysis-result__error,
.analysis-result__warning {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 16px;
  line-height: 1.6;
  border-radius: 10px;
}

.analysis-result__decision span {
  color: var(--ks-muted);
  font-size: 13px;
}

.analysis-result__decision.is-clarify {
  background: #fff8e9;
}

.analysis-result__decision.is-refuse,
.analysis-result__error {
  color: #8d443e;
  background: #fff1ef;
}

.analysis-result__decision.is-eligibility_unavailable {
  color: #8b642c;
  background: #fff8e9;
}

.analysis-result__warning {
  flex-direction: row;
  color: #7e5d2d;
  background: #fff8e9;
  font-size: 12px;
}

.analysis-result__warning > span:first-child {
  display: grid;
  width: 18px;
  height: 18px;
  flex: 0 0 18px;
  place-items: center;
  color: var(--ks-surface);
  font-weight: 800;
  background: var(--ks-warning);
  border-radius: 50%;
}

.analysis-result__viewbar {
  align-items: center;
}

.analysis-result__viewbar strong {
  display: block;
  margin-top: 5px;
  color: var(--ks-ink);
  font-size: 14px;
}

.analysis-result__view-toggle {
  display: inline-flex;
  padding: 3px;
  background: var(--ks-surface-subtle);
  border: 1px solid var(--ks-border);
  border-radius: 8px;
}

.analysis-result__view-toggle button {
  padding: 6px 10px;
  color: var(--ks-muted);
  font-size: 12px;
  background: transparent;
  border: 0;
  border-radius: 6px;
  cursor: pointer;
}

.analysis-result__view-toggle button.is-active {
  color: var(--ks-accent-strong);
  font-weight: 700;
  background: var(--ks-surface);
  box-shadow: var(--ks-shadow-sm);
}

@media (max-width: 560px) {
  .analysis-result__meta,
  .analysis-result__viewbar {
    align-items: flex-start;
    flex-direction: column;
  }

  .analysis-result__meta-items {
    justify-content: flex-start;
  }
}
</style>
