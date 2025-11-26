<script setup lang="ts">
import { useQuery } from "@tanstack/vue-query";
import { ElDialog } from "element-plus";
import { computed, ref, watch } from "vue";

import {
  analyzeVision,
  ApiError,
  documentFileUrl,
  evidenceAssetUrl,
  fetchEvidenceDetail,
  getUserFacingError,
} from "../../api/client";
import type {
  EvidenceRepresentationType,
  RAGCitation,
  VisionAnalysisResult,
  VisionTaskType,
} from "../../api/types";
import AnswerMath from "./AnswerMath.vue";

const props = defineProps<{
  modelValue: boolean;
  knowledgeBaseId: string | null;
  citation: RAGCitation | null;
  documentTitle: string | null;
}>();

const emit = defineEmits<{ "update:modelValue": [value: boolean] }>();

const imageFailed = ref(false);

const visionTask = ref<VisionTaskType | null>(null);
const visionQuestion = ref("");
const visionLoading = ref(false);
const visionResult = ref<VisionAnalysisResult | null>(null);
const visionError = ref<string | null>(null);

const visionTasks: { value: VisionTaskType; label: string }[] = [
  { value: "describe_image", label: "解读图片" },
  { value: "analyze_chart", label: "分析图表" },
  { value: "extract_table", label: "识别表格" },
];

const documentId = computed(() => props.citation?.document_id ?? null);
const evidenceId = computed(() => props.citation?.evidence_id ?? null);

const detailQuery = useQuery({
  queryKey: computed(() => [
    "evidence-detail",
    props.knowledgeBaseId,
    documentId.value,
    evidenceId.value,
  ]),
  queryFn: () => {
    const knowledgeBaseId = props.knowledgeBaseId;
    const currentDocumentId = documentId.value;
    const currentEvidenceId = evidenceId.value;
    if (knowledgeBaseId === null || currentDocumentId === null || currentEvidenceId === null) {
      return Promise.reject(new Error("该来源没有可访问的 Evidence 素材"));
    }
    return fetchEvidenceDetail(knowledgeBaseId, currentDocumentId, currentEvidenceId);
  },
  enabled: computed(
    () => props.modelValue && props.knowledgeBaseId !== null && evidenceId.value !== null,
  ),
  staleTime: 60_000,
  retry: false,
});

const detail = computed(() => detailQuery.data.value ?? null);

const modalityLabel = computed(() => {
  if (detail.value === null) {
    return props.citation?.candidate_kind === "chunk" ? "文本片段" : "来源证据";
  }
  return {
    text: "文本证据",
    table: "表格证据",
    image: "图片证据",
    formula: "公式证据",
  }[detail.value.modality];
});

const pageLabel = computed(() => {
  const citation = props.citation;
  if (citation === null) {
    return "";
  }
  return citation.page_start === citation.page_end
    ? `第 ${citation.page_start} 页`
    : `第 ${citation.page_start}–${citation.page_end} 页`;
});

const sectionLabel = computed(() => {
  const citation = props.citation;
  if (citation === null || citation.section_path.length === 0) {
    return "";
  }
  return citation.section_path.join(" / ");
});

const snippet = computed(() => props.citation?.snippet ?? null);

function firstRepresentationContent(type: EvidenceRepresentationType): string | null {
  const representation = detail.value?.representations.find(
    (item) => item.representation_type === type && item.content !== null,
  );
  return representation?.content ?? null;
}

const caption = computed(() => firstRepresentationContent("caption"));
const contextText = computed(() => firstRepresentationContent("context"));
const markdown = computed(() => firstRepresentationContent("markdown"));
const html = computed(() => firstRepresentationContent("html"));
const latex = computed(() => firstRepresentationContent("latex"));
const primaryText = computed(
  () => firstRepresentationContent("text") ?? firstRepresentationContent("context"),
);

const imageAsset = computed(() => detail.value?.assets.find((asset) => asset.available) ?? null);
const imageSrc = computed(() => {
  const knowledgeBaseId = props.knowledgeBaseId;
  const currentDocumentId = documentId.value;
  const currentEvidenceId = evidenceId.value;
  if (
    knowledgeBaseId === null ||
    currentDocumentId === null ||
    currentEvidenceId === null ||
    imageAsset.value === null
  ) {
    return null;
  }
  return evidenceAssetUrl(knowledgeBaseId, currentDocumentId, currentEvidenceId, imageAsset.value.index);
});

function parseMarkdownTable(value: string): string[][] {
  const rows = value
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.includes("|"))
    .map((line) =>
      line
        .replace(/^\|/, "")
        .replace(/\|$/, "")
        .split("|")
        .map((cell) => cell.trim()),
    );
  return rows.filter(
    (row) => !(row.length > 0 && row.every((cell) => /^:?-{2,}:?$/.test(cell))),
  );
}

function parseHtmlTable(value: string): string[][] {
  const documentNode = new DOMParser().parseFromString(value, "text/html");
  const table = documentNode.querySelector("table");
  if (table === null) {
    return [];
  }
  return Array.from(table.querySelectorAll("tr")).map((row) =>
    Array.from(row.querySelectorAll("th, td")).map(
      (cell) => (cell.textContent ?? "").trim(),
    ),
  );
}

const tableRows = computed<string[][]>(() => {
  const htmlValue = html.value;
  if (htmlValue !== null) {
    const rows = parseHtmlTable(htmlValue);
    if (rows.length > 0) {
      return rows;
    }
  }
  const markdownValue = markdown.value;
  return markdownValue !== null ? parseMarkdownTable(markdownValue) : [];
});

const errorMessage = computed(() => {
  const error = detailQuery.error.value;
  if (error === null) {
    return null;
  }
  if (error instanceof ApiError && error.detail) {
    return error.detail;
  }
  return getUserFacingError(error, "无法读取该来源的原始素材。");
});

function openSourcePage(): void {
  const knowledgeBaseId = props.knowledgeBaseId;
  const currentDocumentId = documentId.value;
  if (knowledgeBaseId === null || currentDocumentId === null) {
    return;
  }
  window.open(
    documentFileUrl(knowledgeBaseId, currentDocumentId, props.citation?.page_start ?? 1),
    "_blank",
    "noopener,noreferrer",
  );
}

const canUseVision = computed(() => {
  const modality = detail.value?.modality;
  if (modality !== "image" && modality !== "table") {
    return false;
  }
  return imageSrc.value !== null && !imageFailed.value;
});

const visibleVisionTasks = computed(() => {
  if (detail.value?.modality === "table") {
    return visionTasks.filter((task) => task.value === "extract_table");
  }
  return visionTasks;
});

const activeAssetIndex = computed(() => imageAsset.value?.index ?? 0);

const visionChart = computed(() => visionResult.value?.chart_analysis ?? null);
const visionTable = computed(() => visionResult.value?.extracted_table ?? null);
const visionWarnings = computed(() => visionResult.value?.warnings ?? []);
const chartSections = computed(() => {
  const chart = visionChart.value;
  if (chart === null) {
    return [];
  }
  return [
    { title: "图表类型", items: chart.chart_type === null ? [] : [chart.chart_type] },
    { title: "坐标轴与分类", items: chart.axes },
    { title: "数据系列", items: chart.series },
    { title: "主要趋势", items: chart.trends },
    { title: "峰值与低谷", items: chart.peaks },
    { title: "显著异常", items: chart.anomalies },
    { title: "图表限制", items: chart.limitations },
  ].filter((section) => section.items.length > 0);
});

function startVisionTask(task: VisionTaskType): void {
  if (visionLoading.value) {
    return;
  }
  visionTask.value = task;
  visionQuestion.value = "";
  visionResult.value = null;
  visionError.value = null;
}

async function runVisionAnalysis(): Promise<void> {
  const knowledgeBaseId = props.knowledgeBaseId;
  const currentDocumentId = documentId.value;
  const currentEvidenceId = evidenceId.value;
  const task = visionTask.value;
  if (
    knowledgeBaseId === null ||
    currentDocumentId === null ||
    currentEvidenceId === null ||
    task === null
  ) {
    return;
  }
  visionLoading.value = true;
  visionError.value = null;
  visionResult.value = null;
  try {
    visionResult.value = await analyzeVision({
      knowledge_base_id: knowledgeBaseId,
      document_id: currentDocumentId,
      evidence_id: currentEvidenceId,
      asset_index: activeAssetIndex.value,
      task_type: task,
      question: visionQuestion.value.trim() || null,
    });
  } catch (error) {
    if (error instanceof ApiError && error.detail) {
      visionError.value = error.detail;
    } else {
      visionError.value = getUserFacingError(error, "视觉模型暂时不可用，请稍后重试。");
    }
  } finally {
    visionLoading.value = false;
  }
}

watch(
  () => [props.modelValue, evidenceId.value, imageSrc.value],
  () => {
    imageFailed.value = false;
    visionTask.value = null;
    visionQuestion.value = "";
    visionLoading.value = false;
    visionResult.value = null;
    visionError.value = null;
  },
);
</script>

<template>
  <ElDialog
    :model-value="modelValue"
    title="来源原始素材"
    width="min(880px, 94vw)"
    top="6vh"
    destroy-on-close
    @update:model-value="emit('update:modelValue', $event)"
  >
    <div class="asset-preview">
      <header class="asset-preview__header">
        <div class="asset-preview__heading">
          <strong class="asset-preview__title">{{ documentTitle || "当前知识库文档" }}</strong>
          <p class="asset-preview__meta">
            <span>{{ modalityLabel }}</span>
            <span v-if="pageLabel">{{ pageLabel }}</span>
            <span v-if="sectionLabel">{{ sectionLabel }}</span>
          </p>
        </div>
        <span
          v-if="citation"
          class="asset-preview__marker"
        >{{ citation.marker }}</span>
      </header>

      <div
        v-if="detailQuery.isLoading.value"
        class="asset-preview__state"
      >
        正在读取来源素材…
      </div>

      <div
        v-else-if="errorMessage"
        class="asset-preview__state asset-preview__state--warning"
      >
        <p>{{ errorMessage }}</p>
        <p class="asset-preview__hint">
          历史文档的原始素材可能未随解析产物保留，回答与引用信息仍可正常使用。
        </p>
        <p
          v-if="snippet"
          class="asset-preview__snippet"
        >
          {{ snippet }}
        </p>
      </div>

      <template v-else-if="detail">
        <div
          v-if="detail.modality === 'image'"
          class="asset-preview__asset"
        >
          <template v-if="imageSrc && !imageFailed">
            <img
              class="asset-preview__image"
              :src="imageSrc"
              :alt="caption || '来源图片'"
              @error="imageFailed = true"
            >
            <p
              v-if="caption"
              class="asset-preview__caption"
            >
              {{ caption }}
            </p>
          </template>
          <div
            v-else
            class="asset-preview__state asset-preview__state--warning"
          >
            <p>该来源的原始图片未能保留，暂时无法预览原图。</p>
            <p
              v-if="caption"
              class="asset-preview__snippet"
            >
              {{ caption }}
            </p>
          </div>
          <p class="asset-preview__hint">
            原始素材来自解析产物，仅用于核验来源。
          </p>
        </div>

        <section
          v-if="canUseVision"
          class="asset-preview__vision"
        >
          <div class="asset-preview__vision-actions">
            <button
              v-for="task in visibleVisionTasks"
              :key="task.value"
              type="button"
              class="asset-preview__button asset-preview__button--ghost"
              :class="{ 'is-active': visionTask === task.value }"
              :disabled="visionLoading"
              @click="startVisionTask(task.value)"
            >
              {{ task.label }}
            </button>
          </div>
          <div
            v-if="visionTask"
            class="asset-preview__vision-form"
          >
            <input
              v-model="visionQuestion"
              class="asset-preview__input"
              type="text"
              :placeholder="
                visionTask === 'extract_table'
                  ? '可选：说明需要提取的表格范围'
                  : '可选：针对这张图片提问'
              "
              :disabled="visionLoading"
            >
            <button
              type="button"
              class="asset-preview__button"
              :disabled="visionLoading"
              @click="runVisionAnalysis"
            >
              {{ visionLoading ? '解读中…' : '开始解读' }}
            </button>
          </div>
          <p
            v-if="visionLoading"
            class="asset-preview__state"
          >
            视觉模型正在解读图片，可能需要数秒…
          </p>
          <p
            v-else-if="visionError"
            class="asset-preview__state asset-preview__state--warning"
          >
            {{ visionError }}
          </p>
          <template v-else-if="visionResult">
            <div
              v-if="visionResult.status === 'not_configured'"
              class="asset-preview__state asset-preview__state--warning"
            >
              <p>{{ visionResult.message || '视觉模型未启用，无法解读图片。' }}</p>
              <p class="asset-preview__hint">
                配置 KNOWLEDGE_SCOPE_VLM_ENABLED、KNOWLEDGE_SCOPE_VLM_BASE_URL、KNOWLEDGE_SCOPE_VLM_MODEL 后可启用该能力。
              </p>
            </div>
            <div
              v-else-if="visionResult.status === 'failed'"
              class="asset-preview__state asset-preview__state--warning"
            >
              <p>{{ visionResult.message || '视觉模型解读失败。' }}</p>
              <p class="asset-preview__hint">
                原始图片与引用信息不受影响，可稍后重试。
              </p>
            </div>
            <template v-else>
              <p class="asset-preview__hint">
                由视觉模型 {{ visionResult.provider || '未知服务' }} / {{ visionResult.model || '未知模型' }} 生成，仅供参考；原始图片与引用信息未改变。
              </p>
              <p
                v-if="visionResult.status === 'needs_review'"
                class="asset-preview__warning"
              >
                视觉模型结果需人工核对。
              </p>
              <p
                v-if="visionResult.answer"
                class="asset-preview__text"
              >
                {{ visionResult.answer }}
              </p>
              <dl
                v-if="visionChart"
                class="asset-preview__chart"
              >
                <template
                  v-for="section in chartSections"
                  :key="section.title"
                >
                  <dt>{{ section.title }}</dt>
                  <dd>
                    <ul>
                      <li
                        v-for="(item, itemIndex) in section.items"
                        :key="itemIndex"
                      >
                        {{ item }}
                      </li>
                    </ul>
                  </dd>
                </template>
              </dl>
              <div
                v-if="visionTable"
                class="asset-preview__vision-table"
              >
                <p
                  v-if="visionTable.title"
                  class="asset-preview__caption"
                >
                  {{ visionTable.title }}<span v-if="visionTable.unit">（单位：{{ visionTable.unit }}）</span>
                </p>
                <table
                  v-if="visionTable.columns.length > 0 && visionTable.rows.length > 0"
                  class="asset-preview__table"
                >
                  <thead>
                    <tr>
                      <th
                        v-for="(column, columnIndex) in visionTable.columns"
                        :key="columnIndex"
                      >
                        {{ column }}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="(row, rowIndex) in visionTable.rows"
                      :key="rowIndex"
                    >
                      <td
                        v-for="(cell, cellIndex) in row"
                        :key="cellIndex"
                      >
                        {{ cell === null || cell === '' ? '—' : cell }}
                      </td>
                    </tr>
                  </tbody>
                </table>
                <p class="asset-preview__warning">
                  视觉模型提取，数值需核对，不能作为精确统计数据使用。
                </p>
                <p
                  v-if="visionTable.merged_cell_info.length > 0"
                  class="asset-preview__hint"
                >
                  合并单元格：{{ visionTable.merged_cell_info.join('；') }}
                </p>
                <ul
                  v-if="visionTable.warnings.length > 0"
                  class="asset-preview__warnings"
                >
                  <li
                    v-for="(warning, warningIndex) in visionTable.warnings"
                    :key="warningIndex"
                  >
                    {{ warning }}
                  </li>
                </ul>
              </div>
              <ul
                v-if="visionWarnings.length > 0"
                class="asset-preview__warnings"
              >
                <li
                  v-for="(warning, warningIndex) in visionWarnings"
                  :key="warningIndex"
                >
                  {{ warning }}
                </li>
              </ul>
            </template>
          </template>
        </section>

        <div
          v-if="detail.modality === 'table'"
          class="asset-preview__asset"
        >
          <table
            v-if="tableRows.length > 0"
            class="asset-preview__table"
          >
            <tbody>
              <tr
                v-for="(row, rowIndex) in tableRows"
                :key="rowIndex"
              >
                <td
                  v-for="(cell, cellIndex) in row"
                  :key="cellIndex"
                >
                  {{ cell }}
                </td>
              </tr>
            </tbody>
          </table>
          <div
            v-else-if="imageSrc && !imageFailed"
            class="asset-preview__table-fallback"
          >
            <p class="asset-preview__hint">
              未找到表格结构化文本，以下为表格原始截图。
            </p>
            <img
              class="asset-preview__image"
              :src="imageSrc"
              :alt="caption || '表格截图'"
              @error="imageFailed = true"
            >
          </div>
          <div
            v-else
            class="asset-preview__state asset-preview__state--warning"
          >
            <p>该来源的表格内容与截图未能保留。</p>
          </div>
          <p
            v-if="caption"
            class="asset-preview__caption"
          >
            {{ caption }}
          </p>
          <p class="asset-preview__hint">
            表格内容按解析产物中的结构化文本展示，不执行任何 HTML。
          </p>
          <p
            v-if="contextText"
            class="asset-preview__snippet"
          >
            {{ contextText }}
          </p>
          <details
            v-if="tableRows.length > 0 && imageSrc && !imageFailed"
            class="asset-preview__details"
          >
            <summary>查看表格原始截图</summary>
            <img
              class="asset-preview__image"
              :src="imageSrc"
              :alt="caption || '表格截图'"
              @error="imageFailed = true"
            >
          </details>
        </div>

        <div
          v-if="detail.modality === 'formula'"
          class="asset-preview__asset"
        >
          <div
            v-if="latex"
            class="asset-preview__formula"
          >
            <AnswerMath
              :latex="latex"
              display
            />
          </div>
          <p
            v-if="contextText"
            class="asset-preview__text"
          >
            {{ contextText }}
          </p>
          <p class="asset-preview__hint">
            公式按解析产物中的 LaTeX 内容展示。
          </p>
        </div>

        <div
          v-if="detail.modality === 'text'"
          class="asset-preview__asset"
        >
          <p
            v-if="primaryText"
            class="asset-preview__text"
          >
            {{ primaryText }}
          </p>
          <p
            v-else
            class="asset-preview__state"
          >
            该证据没有可展示的文本内容。
          </p>
          <p class="asset-preview__hint">
            文本证据直接展示原始解析文本。
          </p>
        </div>

        <p
          v-if="snippet && detail.modality !== 'image' && detail.modality !== 'table'"
          class="asset-preview__snippet"
        >
          {{ snippet }}
        </p>
      </template>

      <footer class="asset-preview__footer">
        <button
          type="button"
          class="asset-preview__button asset-preview__button--ghost"
          @click="emit('update:modelValue', false)"
        >
          关闭
        </button>
        <button
          v-if="knowledgeBaseId && documentId"
          type="button"
          class="asset-preview__button"
          @click="openSourcePage"
        >
          打开原文 PDF（第 {{ citation?.page_start ?? 1 }} 页）
        </button>
      </footer>
    </div>
  </ElDialog>
</template>

<style scoped>
.asset-preview {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.asset-preview__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.asset-preview__heading {
  min-width: 0;
}

.asset-preview__title {
  display: block;
  overflow: hidden;
  color: var(--ks-ink);
  font-size: 14px;
  font-weight: 680;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.asset-preview__meta {
  display: flex;
  margin: 5px 0 0;
  color: var(--ks-muted);
  font-size: 12px;
  flex-wrap: wrap;
  gap: 10px;
}

.asset-preview__marker {
  padding: 2px 7px;
  color: var(--ks-accent-strong);
  font-size: 11px;
  font-weight: 740;
  background: var(--ks-accent-soft);
  border-radius: 4px;
}

.asset-preview__asset {
  display: flex;
  min-height: 180px;
  padding: 14px;
  flex-direction: column;
  gap: 10px;
  background: var(--ks-surface-subtle);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-md);
  overflow: auto;
}

.asset-preview__image {
  display: block;
  max-width: 100%;
  max-height: 58vh;
  margin: 0 auto;
  object-fit: contain;
  border-radius: var(--ks-radius-sm);
}

.asset-preview__table {
  width: 100%;
  border-collapse: collapse;
  color: var(--ks-text);
  font-size: 12px;
}

.asset-preview__table td {
  padding: 6px 9px;
  border: 1px solid var(--ks-border);
  vertical-align: top;
}

.asset-preview__table tr:first-child td {
  color: var(--ks-ink);
  font-weight: 660;
  background: var(--ks-accent-soft);
}

.asset-preview__formula {
  overflow-x: auto;
  padding: 8px 4px;
  color: var(--ks-ink);
}

.asset-preview__text {
  margin: 0;
  color: var(--ks-text);
  font-size: 13px;
  line-height: 1.7;
  white-space: pre-wrap;
}

.asset-preview__caption {
  margin: 0;
  color: var(--ks-muted);
  font-size: 12px;
  line-height: 1.6;
}

.asset-preview__snippet {
  margin: 0;
  padding: 10px 12px;
  color: var(--ks-muted);
  font-size: 12px;
  line-height: 1.65;
  white-space: pre-wrap;
  background: var(--ks-surface-subtle);
  border-left: 3px solid var(--ks-accent);
  border-radius: var(--ks-radius-sm);
}

.asset-preview__hint {
  margin: 0;
  color: var(--ks-faint);
  font-size: 11px;
  line-height: 1.6;
}

.asset-preview__state {
  padding: 26px 14px;
  color: var(--ks-muted);
  font-size: 13px;
  text-align: center;
}

.asset-preview__state--warning {
  text-align: left;
  background: #fff8ec;
  border: 1px solid #f0d9b5;
  border-radius: var(--ks-radius-md);
}

.asset-preview__state--warning p {
  margin: 0 0 8px;
}

.asset-preview__details {
  margin-top: 4px;
}

.asset-preview__details summary {
  color: var(--ks-accent-strong);
  font-size: 12px;
  cursor: pointer;
}

.asset-preview__details img {
  margin-top: 10px;
}

.asset-preview__footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

.asset-preview__button {
  padding: 6px 14px;
  color: #fff;
  font: inherit;
  font-size: 12px;
  font-weight: 620;
  background: var(--ks-accent);
  border: 0;
  border-radius: var(--ks-radius-sm);
  cursor: pointer;
}

.asset-preview__button:hover {
  background: var(--ks-accent-strong);
}

.asset-preview__button--ghost {
  color: var(--ks-text);
  background: transparent;
  border: 1px solid var(--ks-border);
}

.asset-preview__button--ghost:hover {
  background: var(--ks-surface-subtle);
}

.asset-preview__button:focus-visible {
  outline: 2px solid var(--ks-accent);
  outline-offset: 2px;
}

.asset-preview__button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.asset-preview__vision {
  display: flex;
  padding: 12px 14px;
  flex-direction: column;
  gap: 10px;
  border: 1px dashed var(--ks-border);
  border-radius: var(--ks-radius-md);
}

.asset-preview__vision-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.asset-preview__vision-actions .asset-preview__button.is-active {
  color: var(--ks-accent-strong);
  background: var(--ks-accent-soft);
  border-color: var(--ks-accent);
}

.asset-preview__vision-form {
  display: flex;
  gap: 8px;
}

.asset-preview__input {
  min-width: 0;
  padding: 6px 10px;
  color: var(--ks-text);
  font: inherit;
  font-size: 12px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-sm);
  flex: 1 1 auto;
}

.asset-preview__input:focus-visible {
  outline: 2px solid var(--ks-accent);
  outline-offset: 1px;
}

.asset-preview__warning {
  margin: 0;
  padding: 8px 10px;
  color: #8a5a00;
  font-size: 12px;
  line-height: 1.6;
  background: #fff8ec;
  border: 1px solid #f0d9b5;
  border-radius: var(--ks-radius-sm);
}

.asset-preview__chart {
  display: grid;
  margin: 0;
  grid-template-columns: max-content 1fr;
  gap: 6px 12px;
}

.asset-preview__chart dt {
  color: var(--ks-ink);
  font-size: 12px;
  font-weight: 660;
}

.asset-preview__chart dd {
  margin: 0;
  color: var(--ks-text);
  font-size: 12px;
  line-height: 1.7;
}

.asset-preview__chart ul {
  margin: 0;
  padding-left: 16px;
}

.asset-preview__vision-table {
  display: flex;
  flex-direction: column;
  gap: 8px;
  overflow-x: auto;
}

.asset-preview__table th {
  padding: 6px 9px;
  color: var(--ks-ink);
  font-weight: 660;
  text-align: left;
  background: var(--ks-accent-soft);
  border: 1px solid var(--ks-border);
}

.asset-preview__warnings {
  margin: 0;
  padding-left: 18px;
  color: var(--ks-muted);
  font-size: 12px;
  line-height: 1.7;
}
</style>
