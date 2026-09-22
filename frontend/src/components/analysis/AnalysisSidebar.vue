<script setup lang="ts">
import type { ChatBIDataSource } from "../../api/types";
import type { AnalysisSession } from "../../stores/analysis";

defineProps<{
  dataSources: ChatBIDataSource[];
  selectedDatasourceId: string | null;
  dataSourcesLoading: boolean;
  dataSourcesError: boolean;
  dataSourcesErrorMessage: string;
  sessions: AnalysisSession[];
  activeSessionId: string | null;
  disabled: boolean;
}>();

const emit = defineEmits<{
  selectDatasource: [id: string | null];
  newAnalysis: [];
  selectSession: [id: string];
  retryDataSources: [];
}>();

function sessionLabel(session: AnalysisSession): string {
  return session.question.trim().slice(0, 28) || "新分析";
}
</script>

<template>
  <aside class="analysis-sidebar">
    <div class="analysis-sidebar__topline">
      <span class="analysis-sidebar__eyebrow">ChatBI</span>
      <button
        class="analysis-sidebar__new"
        type="button"
        :disabled="disabled"
        @click="emit('newAnalysis')"
      >
        <span aria-hidden="true">＋</span>
        新分析
      </button>
    </div>

    <label
      class="analysis-sidebar__label"
      for="analysis-datasource"
    >数据源</label>
    <select
      id="analysis-datasource"
      class="analysis-sidebar__select"
      :value="selectedDatasourceId ?? ''"
      :disabled="disabled || dataSourcesLoading || dataSources.length === 0"
      @change="emit('selectDatasource', ($event.target as HTMLSelectElement).value || null)"
    >
      <option
        value=""
        disabled
      >
        选择业务数据源
      </option>
      <option
        v-for="dataSource in dataSources"
        :key="dataSource.id"
        :value="dataSource.id"
      >
        {{ dataSource.display_name }}
      </option>
    </select>
    <p
      v-if="dataSourcesLoading"
      class="analysis-sidebar__hint"
    >
      正在读取数据源…
    </p>
    <div
      v-else-if="dataSourcesError"
      class="analysis-sidebar__error"
      role="alert"
    >
      <span>{{ dataSourcesErrorMessage }}</span>
      <button
        type="button"
        @click="emit('retryDataSources')"
      >
        重试
      </button>
    </div>
    <p
      v-else-if="dataSources.length === 0"
      class="analysis-sidebar__hint"
    >
      暂无可用的数据源，请先完成注册。
    </p>

    <div class="analysis-sidebar__history-head">
      <span>本次会话</span>
      <span>{{ sessions.length }}</span>
    </div>
    <div
      v-if="sessions.length === 0"
      class="analysis-sidebar__empty"
    >
      你的分析记录会保留在当前页面中。
    </div>
    <nav
      v-else
      class="analysis-sidebar__history"
      aria-label="分析记录"
    >
      <button
        v-for="session in sessions"
        :key="session.id"
        class="analysis-sidebar__history-item"
        :class="{ 'is-active': session.id === activeSessionId }"
        type="button"
        :disabled="disabled"
        @click="emit('selectSession', session.id)"
      >
        <span class="analysis-sidebar__history-title">{{ sessionLabel(session) }}</span>
        <span class="analysis-sidebar__history-status">
          {{ session.status === "loading" ? "分析中" : session.status === "error" ? "未完成" : "" }}
        </span>
      </button>
    </nav>
  </aside>
</template>

<style scoped>
.analysis-sidebar {
  display: flex;
  min-width: 220px;
  width: 248px;
  flex-direction: column;
  gap: 12px;
  padding: 18px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-lg);
}

.analysis-sidebar__topline,
.analysis-sidebar__history-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.analysis-sidebar__eyebrow {
  color: var(--ks-accent);
  font-size: 11px;
  font-weight: 760;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.analysis-sidebar__new {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 6px 8px;
  color: var(--ks-accent-strong);
  font-size: 12px;
  font-weight: 700;
  background: var(--ks-accent-soft);
  border: 1px solid transparent;
  border-radius: 7px;
  cursor: pointer;
}

.analysis-sidebar__new:disabled,
.analysis-sidebar__history-item:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.analysis-sidebar__label {
  color: var(--ks-muted);
  font-size: 12px;
  font-weight: 680;
}

.analysis-sidebar__select {
  width: 100%;
  min-height: 38px;
  padding: 0 10px;
  color: var(--ks-ink);
  background: var(--ks-surface-subtle);
  border: 1px solid var(--ks-border);
  border-radius: 8px;
}

.analysis-sidebar__hint,
.analysis-sidebar__empty {
  margin: -4px 0 0;
  color: var(--ks-muted);
  font-size: 12px;
  line-height: 1.55;
}

.analysis-sidebar__error {
  display: flex;
  flex-direction: column;
  gap: 7px;
  color: var(--ks-danger);
  font-size: 12px;
  line-height: 1.5;
}

.analysis-sidebar__error button {
  align-self: flex-start;
  padding: 0;
  color: var(--ks-accent-strong);
  background: transparent;
  border: 0;
  cursor: pointer;
}

.analysis-sidebar__history-head {
  margin-top: 10px;
  color: var(--ks-muted);
  font-size: 12px;
  font-weight: 680;
}

.analysis-sidebar__history-head span:last-child {
  color: var(--ks-faint);
  font-weight: 500;
}

.analysis-sidebar__history {
  display: flex;
  max-height: 360px;
  flex-direction: column;
  gap: 4px;
  overflow-y: auto;
}

.analysis-sidebar__history-item {
  display: flex;
  min-width: 0;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 9px 10px;
  color: var(--ks-text);
  text-align: left;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 8px;
  cursor: pointer;
}

.analysis-sidebar__history-item:hover,
.analysis-sidebar__history-item.is-active {
  color: var(--ks-accent-strong);
  background: var(--ks-accent-soft);
}

.analysis-sidebar__history-title {
  min-width: 0;
  overflow: hidden;
  font-size: 13px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.analysis-sidebar__history-status {
  flex: 0 0 auto;
  color: var(--ks-muted);
  font-size: 11px;
}

@media (max-width: 880px) {
  .analysis-sidebar {
    width: auto;
  }

  .analysis-sidebar__history {
    max-height: 180px;
  }
}
</style>
