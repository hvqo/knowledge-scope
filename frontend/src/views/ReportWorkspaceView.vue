<script setup lang="ts">
import { useQuery } from "@tanstack/vue-query";
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { onBeforeRouteLeave, onBeforeRouteUpdate, useRoute, useRouter } from "vue-router";

import {
  askChatBI,
  createReportSection,
  deleteReportSection,
  deleteReportSource,
  fetchChatBIDataSources,
  fetchDocuments,
  fetchKnowledgeBases,
  fetchReport,
  getUserFacingError,
  insertReportSource,
  streamRagQuery,
  updateReport,
  updateReportSection,
} from "../api/client";
import type {
  ChatBIAnalysisResult,
  Document,
  RAGCitation,
  ReportChartSpec,
  ReportRagSourceCreateRequest,
  ReportSection,
} from "../api/types";
import ReportEditor from "../components/report/ReportEditor.vue";
import ReportOutline from "../components/report/ReportOutline.vue";
import ReportPreview from "../components/report/ReportPreview.vue";
import ReportSourcePanel from "../components/report/ReportSourcePanel.vue";

const route = useRoute();
const router = useRouter();
const reportId = computed(() => (typeof route.params.id === "string" ? route.params.id : ""));

const activeSectionId = ref<string | null>(null);
const draftReportTitle = ref("");
const draftSectionTitle = ref("");
const draftContent = ref("");
const preview = ref(false);
const saveState = ref<"idle" | "saving" | "saved" | "error">("idle");
const actionError = ref<string | null>(null);
const sectionBusy = ref(false);
const sourceBusy = ref(false);

const reportQuery = useQuery({
  queryKey: computed(() => ["report", reportId.value]),
  queryFn: () => fetchReport(reportId.value),
  enabled: computed(() => reportId.value.length > 0),
});
const report = computed(() => reportQuery.data.value ?? null);
const knowledgeBasesQuery = useQuery({
  queryKey: ["report-knowledge-bases"],
  queryFn: () => fetchKnowledgeBases({ limit: 100 }),
});
const dataSourcesQuery = useQuery({
  queryKey: ["report-data-sources"],
  queryFn: () => fetchChatBIDataSources({ limit: 100 }),
});
const documentsQuery = useQuery({
  queryKey: computed(() => ["report-documents", report.value?.knowledge_base_id]),
  queryFn: () => fetchDocuments(report.value?.knowledge_base_id ?? "", { limit: 100 }),
  enabled: computed(() => report.value?.knowledge_base_id !== null && report.value !== undefined),
});

const activeSection = computed<ReportSection | null>(() =>
  report.value?.sections.find((section) => section.id === activeSectionId.value) ?? null,
);
const knowledgeBases = computed(() => knowledgeBasesQuery.data.value?.items ?? []);
const dataSources = computed(() => dataSourcesQuery.data.value?.items ?? []);
const documentTitles = computed<Record<string, string>>(() => {
  const documents: Document[] = documentsQuery.data.value?.items ?? [];
  return Object.fromEntries(documents.map((document) => [document.id, document.original_filename]));
});
const reportKnowledgeBaseName = computed(
  () => knowledgeBases.value.find((item) => item.id === report.value?.knowledge_base_id)?.name ?? null,
);
const reportDatasourceName = computed(
  () => dataSources.value.find((item) => item.id === report.value?.datasource_id)?.display_name ?? null,
);
const isDirty = computed(() => {
  const section = activeSection.value;
  return (
    section !== null &&
    (draftReportTitle.value !== (report.value?.title ?? "") ||
      draftSectionTitle.value !== section.title ||
      draftContent.value !== section.content)
  );
});
const isBusy = computed(
  () =>
    reportQuery.isPending.value ||
    sectionBusy.value ||
    sourceBusy.value ||
    saveState.value === "saving",
);
const loadErrorMessage = computed(() =>
  getUserFacingError(reportQuery.error.value, "报告暂时无法加载，请稍后重试。"),
);

const ragAnswer = ref("");
const ragCitations = ref<RAGCitation[]>([]);
const ragLoading = ref(false);
const ragError = ref<string | null>(null);
const chatBIResult = ref<ChatBIAnalysisResult | null>(null);
const chatBILoading = ref(false);
const chatBIError = ref<string | null>(null);
const activeRagController = ref<AbortController | null>(null);
const activeChatBIController = ref<AbortController | null>(null);
let ragRun = 0;
let chatBIRun = 0;
let workspaceRun = 0;
let isUnmounting = false;

function syncDrafts(): void {
  const currentReport = report.value;
  if (currentReport === null || currentReport.id !== reportId.value) {
    return;
  }
  if (
    activeSectionId.value === null ||
    !currentReport.sections.some((section) => section.id === activeSectionId.value)
  ) {
    activeSectionId.value = currentReport.sections[0]?.id ?? null;
  }
  const section = currentReport.sections.find((item) => item.id === activeSectionId.value);
  draftReportTitle.value = currentReport.title;
  draftSectionTitle.value = section?.title ?? "";
  draftContent.value = section?.content ?? "";
  saveState.value = "saved";
}

watch(report, syncDrafts, { immediate: true });

function cancelSourceRequests(): void {
  ragRun += 1;
  chatBIRun += 1;
  activeRagController.value?.abort();
  activeChatBIController.value?.abort();
  activeRagController.value = null;
  activeChatBIController.value = null;
  ragLoading.value = false;
  chatBILoading.value = false;
}

function clearSourceResults(): void {
  ragAnswer.value = "";
  ragCitations.value = [];
  ragError.value = null;
  chatBIResult.value = null;
  chatBIError.value = null;
}

watch(reportId, (nextId, previousId) => {
  if (previousId !== undefined && nextId !== previousId) {
    workspaceRun += 1;
    cancelSourceRequests();
    clearSourceResults();
    activeSectionId.value = null;
    preview.value = false;
    actionError.value = null;
    draftReportTitle.value = "";
    draftSectionTitle.value = "";
    draftContent.value = "";
    saveState.value = "idle";
    sourceBusy.value = false;
  }
});

function updateDraftReportTitle(value: string): void {
  draftReportTitle.value = value;
  saveState.value = "idle";
}

function updateDraftSectionTitle(value: string): void {
  draftSectionTitle.value = value;
  saveState.value = "idle";
}

function updateDraftContent(value: string): void {
  draftContent.value = value;
  saveState.value = "idle";
}

async function selectSection(id: string): Promise<void> {
  if (isBusy.value || id === activeSectionId.value) {
    return;
  }
  if (isDirty.value && !(await saveDraft())) {
    return;
  }
  if (id === activeSectionId.value) {
    return;
  }
  activeSectionId.value = id;
  preview.value = false;
  syncDrafts();
}

async function saveDraft(): Promise<boolean> {
  const currentReport = report.value;
  const section = activeSection.value;
  const run = workspaceRun;
  const targetReportId = reportId.value;
  const targetSectionId = section?.id;
  if (
    currentReport === null ||
    currentReport.id !== targetReportId ||
    section === null ||
    targetSectionId === undefined ||
    !draftReportTitle.value.trim()
  ) {
    return false;
  }
  saveState.value = "saving";
  actionError.value = null;
  try {
    await updateReport(targetReportId, { title: draftReportTitle.value.trim() });
    await updateReportSection(targetReportId, targetSectionId, {
      title: draftSectionTitle.value.trim(),
      content: draftContent.value,
    });
    if (run !== workspaceRun || isUnmounting) {
      return false;
    }
    await reportQuery.refetch();
    if (run !== workspaceRun || isUnmounting) {
      return false;
    }
    saveState.value = "saved";
    return true;
  } catch (error) {
    if (run !== workspaceRun || isUnmounting) {
      return false;
    }
    saveState.value = "error";
    actionError.value = getUserFacingError(error, "保存报告失败，请稍后重试。");
    return false;
  }
}

async function protectDirtyDraft(): Promise<boolean> {
  if (!isDirty.value) {
    return true;
  }
  return saveDraft();
}

onBeforeRouteUpdate(async (to, from) => {
  if (String(to.params.id ?? "") === String(from.params.id ?? "")) {
    return true;
  }
  return protectDirtyDraft();
});

onBeforeRouteLeave(() => protectDirtyDraft());

function handleBeforeUnload(event: BeforeUnloadEvent): void {
  if (!isDirty.value) {
    return;
  }
  event.preventDefault();
  event.returnValue = "";
}

onMounted(() => {
  window.addEventListener("beforeunload", handleBeforeUnload);
});

async function addSection(): Promise<void> {
  if (report.value === null || sectionBusy.value) {
    return;
  }
  sectionBusy.value = true;
  actionError.value = null;
  try {
    const section = await createReportSection(reportId.value, { title: "新章节" });
    await reportQuery.refetch();
    activeSectionId.value = section.id;
    preview.value = false;
    syncDrafts();
  } catch (error) {
    actionError.value = getUserFacingError(error, "无法添加章节，请稍后重试。");
  } finally {
    sectionBusy.value = false;
  }
}

async function removeSection(): Promise<void> {
  const section = activeSection.value;
  if (section === null || sectionBusy.value) {
    return;
  }
  if (!window.confirm(`确定删除“${section.title}”吗？章节中的来源引用也会移除。`)) {
    return;
  }
  if (isDirty.value && !(await saveDraft())) {
    return;
  }
  sectionBusy.value = true;
  actionError.value = null;
  try {
    await deleteReportSection(reportId.value, section.id);
    activeSectionId.value = null;
    await reportQuery.refetch();
    preview.value = false;
    syncDrafts();
  } catch (error) {
    actionError.value = getUserFacingError(error, "无法删除章节，请稍后重试。");
  } finally {
    sectionBusy.value = false;
  }
}

function ragFailureMessage(category: string): string {
  if (category === "retrieval") {
    return "暂时无法读取当前知识库，请稍后重试。";
  }
  if (category === "request") {
    return "问题格式不正确，请调整后重试。";
  }
  if (category === "timeout") {
    return "证据检索超时，请稍后重试。";
  }
  return "证据检索未完成，请稍后重试。";
}

async function searchRag(question: string): Promise<void> {
  const knowledgeBaseId = report.value?.knowledge_base_id;
  if (knowledgeBaseId === null || knowledgeBaseId === undefined || ragLoading.value) {
    return;
  }
  ragRun += 1;
  const run = ragRun;
  activeRagController.value?.abort();
  const controller = new AbortController();
  activeRagController.value = controller;
  ragLoading.value = true;
  ragError.value = null;
  ragAnswer.value = "";
  ragCitations.value = [];
  let completed = false;
  try {
    for await (const event of streamRagQuery(
      { query: question, knowledge_base_id: knowledgeBaseId, retrieval_mode: "dense" },
      controller.signal,
    )) {
      if (run !== ragRun || isUnmounting) {
        return;
      }
      if (event.event === "answer_delta") {
        ragAnswer.value += event.data.text;
      } else if (event.event === "citations") {
        ragCitations.value = event.data.items;
      } else if (event.event === "error") {
        ragError.value = ragFailureMessage(event.data.category);
        break;
      } else {
        completed = true;
        if (event.data.status === "error") {
          ragError.value = "证据检索未完成，请稍后重试。";
        }
      }
    }
    if (run === ragRun && !completed && ragError.value === null) {
      ragError.value = "连接已中断，证据没有完成返回。";
    }
  } catch (error) {
    if (run === ragRun && !isUnmounting && !controller.signal.aborted) {
      ragError.value = getUserFacingError(error, "知识库暂时无法查询，请稍后重试。");
    }
  } finally {
    if (run === ragRun) {
      ragLoading.value = false;
      activeRagController.value = null;
    }
  }
}

async function askDataSource(question: string): Promise<void> {
  const datasourceId = report.value?.datasource_id;
  if (datasourceId === null || datasourceId === undefined || chatBILoading.value) {
    return;
  }
  chatBIRun += 1;
  const run = chatBIRun;
  activeChatBIController.value?.abort();
  const controller = new AbortController();
  activeChatBIController.value = controller;
  chatBILoading.value = true;
  chatBIError.value = null;
  chatBIResult.value = null;
  try {
    const result = await askChatBI(datasourceId, question, controller.signal);
    if (run === chatBIRun && !isUnmounting) {
      chatBIResult.value = result;
    }
  } catch (error) {
    if (run === chatBIRun && !isUnmounting && !controller.signal.aborted) {
      chatBIError.value = getUserFacingError(error, "数据分析暂时无法完成，请稍后重试。");
    }
  } finally {
    if (run === chatBIRun) {
      chatBILoading.value = false;
      activeChatBIController.value = null;
    }
  }
}

function appendSourceText(content: string, heading: string, body: string): string {
  const entry = [heading, body.trim()].filter(Boolean).join("\n");
  return content.trim() ? `${content.trim()}\n\n${entry}` : entry;
}

function copyResultRows(value: unknown): unknown[][] {
  if (typeof value !== "object" || value === null || !("rows" in value)) {
    return [];
  }
  const rows = (value as { rows?: unknown }).rows;
  return Array.isArray(rows)
    ? rows.map((row) => (Array.isArray(row) ? [...row] : []))
    : [];
}

async function insertRag(citation: RAGCitation): Promise<void> {
  const currentReport = report.value;
  const section = activeSection.value;
  const run = workspaceRun;
  const targetReportId = reportId.value;
  const targetSectionId = section?.id;
  const wrongKnowledgeBase =
    citation.knowledge_base_id !== null &&
    citation.knowledge_base_id !== currentReport?.knowledge_base_id;
  if (
    currentReport === null ||
    currentReport.id !== targetReportId ||
    section === null ||
    targetSectionId === undefined ||
    currentReport.knowledge_base_id === null ||
    wrongKnowledgeBase ||
    sourceBusy.value
  ) {
    return;
  }
  if (isDirty.value && !(await saveDraft())) {
    return;
  }
  if (run !== workspaceRun || isUnmounting) {
    return;
  }
  sourceBusy.value = true;
  actionError.value = null;
  const documentTitle = documentTitles.value[citation.document_id] ?? "来源文档";
  const payload: ReportRagSourceCreateRequest = {
    kind: "rag_evidence",
    title: citation.section_title || documentTitle,
    knowledge_base_id: currentReport.knowledge_base_id,
    document_id: citation.document_id,
    evidence_id: citation.evidence_id,
    chunk_id: citation.chunk_id,
    document_title: documentTitle,
    page_start: citation.page_start,
    page_end: citation.page_end,
    evidence_type: citation.modality ?? "text",
    snippet: citation.snippet,
    section_path: citation.section_path,
    source_block_ids: citation.source_block_ids,
  };
  try {
    const body = citation.snippet ?? `${documentTitle}，${citation.page_start}–${citation.page_end} 页`;
    await insertReportSource(targetReportId, targetSectionId, {
      source: payload,
      content_append: appendSourceText("", `【${payload.title}】`, body),
    });
    if (run === workspaceRun && !isUnmounting) {
      await reportQuery.refetch();
    }
  } catch (error) {
    if (run === workspaceRun && !isUnmounting) {
      actionError.value = getUserFacingError(error, "无法插入知识库证据，请稍后重试。");
    }
  } finally {
    if (run === workspaceRun) {
      sourceBusy.value = false;
    }
  }
}

async function insertChatBI(question: string, chartSpec: ReportChartSpec | null): Promise<void> {
  const currentReport = report.value;
  const section = activeSection.value;
  const result = chatBIResult.value;
  const run = workspaceRun;
  const targetReportId = reportId.value;
  const targetSectionId = section?.id;
  if (
    currentReport === null ||
    currentReport.id !== targetReportId ||
    section === null ||
    targetSectionId === undefined ||
    currentReport.datasource_id === null ||
    result === null ||
    result.execution_status !== "succeeded" ||
    sourceBusy.value
  ) {
    return;
  }
  if (isDirty.value && !(await saveDraft())) {
    return;
  }
  if (run !== workspaceRun || isUnmounting) {
    return;
  }
  sourceBusy.value = true;
  actionError.value = null;
  const datasourceName = reportDatasourceName.value ?? "业务数据源";
  const rows = copyResultRows(result as unknown);
  const payload = {
    kind: "chatbi_result" as const,
    title: question || "数据分析结果",
    datasource_id: currentReport.datasource_id,
    datasource_name: datasourceName,
    question: question || "数据分析结果",
    answer: result.answer,
    columns: result.columns,
    rows,
    row_count: result.row_count,
    truncated: result.truncated,
    truncation_reason: result.truncation_reason,
    redacted_sql: result.redacted_sql,
    chart_spec: chartSpec,
  };
  try {
    const body = result.answer ?? `${result.row_count} 行结果`;
    await insertReportSource(targetReportId, targetSectionId, {
      source: payload,
      content_append: appendSourceText("", `【${payload.title}】`, body),
    });
    if (run === workspaceRun && !isUnmounting) {
      await reportQuery.refetch();
    }
  } catch (error) {
    if (run === workspaceRun && !isUnmounting) {
      actionError.value = getUserFacingError(error, "无法插入分析结果，请稍后重试。");
    }
  } finally {
    if (run === workspaceRun) {
      sourceBusy.value = false;
    }
  }
}

async function removeReference(sourceId: string): Promise<void> {
  if (sourceBusy.value) {
    return;
  }
  const run = workspaceRun;
  const targetReportId = reportId.value;
  sourceBusy.value = true;
  actionError.value = null;
  try {
    await deleteReportSource(targetReportId, sourceId);
    if (run === workspaceRun && !isUnmounting) {
      await reportQuery.refetch();
    }
  } catch (error) {
    if (run === workspaceRun && !isUnmounting) {
      actionError.value = getUserFacingError(error, "无法移除来源，请稍后重试。");
    }
  } finally {
    if (run === workspaceRun) {
      sourceBusy.value = false;
    }
  }
}

function retry(): void {
  void reportQuery.refetch();
}

function goBack(): void {
  void router.push({ name: "reports" });
}

onBeforeUnmount(() => {
  isUnmounting = true;
  window.removeEventListener("beforeunload", handleBeforeUnload);
  workspaceRun += 1;
  cancelSourceRequests();
  clearSourceResults();
});
</script>

<template>
  <section class="report-workspace-page">
    <header class="report-workspace-page__header">
      <button
        type="button"
        class="report-workspace-page__back"
        @click="goBack"
      >
        ← 报告列表
      </button>
      <div
        v-if="report"
        class="report-workspace-page__context"
      >
        <span>{{ reportKnowledgeBaseName ?? "未关联知识库" }}</span>
        <span>·</span>
        <span>{{ reportDatasourceName ?? "未关联数据源" }}</span>
      </div>
    </header>

    <div
      v-if="reportQuery.isPending.value"
      class="report-workspace-page__state"
      role="status"
    >
      正在打开报告…
    </div>
    <div
      v-else-if="reportQuery.isError.value"
      class="report-workspace-page__state report-workspace-page__state--error"
      role="alert"
    >
      <strong>{{ loadErrorMessage }}</strong>
      <button
        type="button"
        @click="retry"
      >
        重试
      </button>
    </div>
    <div
      v-else-if="report"
      class="report-workspace"
    >
      <ReportOutline
        :title="draftReportTitle"
        :sections="report.sections"
        :active-section-id="activeSectionId"
        :disabled="isBusy"
        @update:title="updateDraftReportTitle"
        @select="selectSection"
        @add="addSection"
        @delete="removeSection"
      />

      <main class="report-workspace__editor">
        <div
          v-if="actionError"
          class="report-workspace__error"
          role="alert"
        >
          {{ actionError }}
        </div>
        <ReportPreview
          v-if="preview"
          :report="report"
        />
        <ReportEditor
          v-else-if="activeSection"
          :section-title="draftSectionTitle"
          :content="draftContent"
          :preview="false"
          :save-state="saveState"
          :disabled="isBusy"
          @update:section-title="updateDraftSectionTitle"
          @update:content="updateDraftContent"
          @save="saveDraft"
          @toggle-preview="preview = true"
        />
        <div
          v-else
          class="report-workspace__empty"
        >
          <strong>报告还没有章节</strong>
          <span>从左侧添加一个章节开始编辑。</span>
          <button
            type="button"
            :disabled="isBusy"
            @click="addSection"
          >
            添加章节
          </button>
        </div>
        <button
          v-if="preview"
          type="button"
          class="report-workspace__return-edit"
          @click="preview = false"
        >
          返回编辑
        </button>
      </main>

      <ReportSourcePanel
        :key="reportId"
        :knowledge-base-id="report.knowledge_base_id"
        :datasource-id="report.datasource_id"
        :document-titles="documentTitles"
        :references="report.sections.flatMap((section) => section.sources)"
        :rag-answer="ragAnswer"
        :rag-citations="ragCitations"
        :rag-loading="ragLoading"
        :rag-error="ragError"
        :chat-b-i-result="chatBIResult"
        :chat-b-i-loading="chatBILoading"
        :chat-b-i-error="chatBIError"
        :disabled="isBusy"
        @search-rag="searchRag"
        @insert-rag="insertRag"
        @ask-chat-b-i="askDataSource"
        @insert-chat-b-i="insertChatBI"
        @remove-reference="removeReference"
      />
    </div>
  </section>
</template>

<style scoped>
.report-workspace-page {
  display: flex;
  min-height: calc(100vh - 104px);
  flex-direction: column;
  gap: 15px;
}

.report-workspace-page__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.report-workspace-page__back {
  padding: 6px 0;
  color: var(--ks-muted);
  font-size: 13px;
  background: transparent;
  border: 0;
  cursor: pointer;
}

.report-workspace-page__back:hover {
  color: var(--ks-accent-strong);
}

.report-workspace-page__context {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  color: var(--ks-muted);
  font-size: 12px;
}

.report-workspace {
  display: grid;
  min-width: 0;
  grid-template-columns: 238px minmax(0, 1fr) 350px;
  align-items: start;
  gap: 16px;
}

.report-workspace__editor {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 10px;
}

.report-workspace__error {
  padding: 10px 13px;
  color: var(--ks-danger);
  font-size: 12px;
  background: var(--ks-danger-soft);
  border: 1px solid rgb(176 75 75 / 18%);
  border-radius: 8px;
}

.report-workspace__empty,
.report-workspace-page__state {
  display: flex;
  min-height: 540px;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 10px;
  padding: 30px;
  color: var(--ks-muted);
  text-align: center;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-lg);
}

.report-workspace__empty strong,
.report-workspace-page__state strong {
  color: var(--ks-ink);
  font-size: 16px;
}

.report-workspace__empty button,
.report-workspace-page__state button {
  min-height: 34px;
  padding: 0 13px;
  color: var(--ks-surface);
  background: var(--ks-accent);
  border: 0;
  border-radius: 7px;
  cursor: pointer;
}

.report-workspace-page__state--error {
  color: var(--ks-danger);
}

.report-workspace__return-edit {
  align-self: center;
  padding: 7px 12px;
  color: var(--ks-accent-strong);
  background: var(--ks-accent-soft);
  border: 0;
  border-radius: 7px;
  cursor: pointer;
}

@media (max-width: 1120px) {
  .report-workspace {
    grid-template-columns: 210px minmax(0, 1fr);
  }

  .report-source-panel {
    grid-column: 1 / -1;
  }
}

@media (max-width: 720px) {
  .report-workspace-page__header {
    align-items: flex-start;
    flex-direction: column;
  }

  .report-workspace {
    display: flex;
    flex-direction: column;
  }

  .report-outline,
  .report-workspace__editor,
  .report-source-panel {
    width: 100%;
  }
}
</style>
