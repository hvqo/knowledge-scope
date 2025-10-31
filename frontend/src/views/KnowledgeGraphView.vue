<script setup lang="ts">
import { useQuery } from "@tanstack/vue-query";
import { ElButton, ElInput, ElOption, ElSelect } from "element-plus";
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { RouterLink, useRoute, useRouter } from "vue-router";

import {
  fetchDocuments,
  fetchGraphEntityDetail,
  fetchGraphOverview,
  fetchKnowledgeBases,
  getUserFacingError,
} from "../api/client";
import type { GraphNode } from "../api/types";
import EntityDetailPanel from "../components/graph/EntityDetailPanel.vue";
import GraphCanvas from "../components/graph/GraphCanvas.vue";
import { colorForType } from "../components/graph/graphPalette";

const NODE_LIMIT_OPTIONS = [100, 200, 300, 500] as const;
const SEARCH_DEBOUNCE_MS = 350;
const LEGEND_LIMIT = 8;

const route = useRoute();
const router = useRouter();
const selectedKnowledgeBaseId = ref<string | null>(
  typeof route.query.kb === "string" ? route.query.kb : null,
);
const searchInput = ref("");
const searchTerm = ref("");
const selectedType = ref("");
const selectedDocumentId = ref("");
const documentChoiceMade = ref(false);
const nodeLimit = ref<number>(NODE_LIMIT_OPTIONS[1]);
const selectedEntityId = ref<string | null>(null);
const canvasRef = ref<InstanceType<typeof GraphCanvas> | null>(null);
let searchTimer: number | undefined;

const knowledgeBasesQuery = useQuery({
  queryKey: ["graph-knowledge-bases"],
  queryFn: () => fetchKnowledgeBases({ limit: 100 }),
  staleTime: 60_000,
});

const graphQuery = useQuery({
  queryKey: computed(() => [
    "graph-overview",
    selectedKnowledgeBaseId.value,
    searchTerm.value,
    selectedType.value,
    selectedDocumentId.value,
    nodeLimit.value,
  ]),
  queryFn: () =>
    fetchGraphOverview(selectedKnowledgeBaseId.value ?? "", {
      search: searchTerm.value || undefined,
      entityType: selectedType.value || undefined,
      documentId: selectedDocumentId.value || undefined,
      limit: nodeLimit.value,
    }),
  enabled: computed(() => selectedKnowledgeBaseId.value !== null),
  staleTime: 60_000,
});

const entityDetailQuery = useQuery({
  queryKey: computed(() => [
    "graph-entity",
    selectedKnowledgeBaseId.value,
    selectedEntityId.value,
  ]),
  queryFn: () =>
    fetchGraphEntityDetail(selectedKnowledgeBaseId.value ?? "", selectedEntityId.value ?? ""),
  enabled: computed(
    () => selectedKnowledgeBaseId.value !== null && selectedEntityId.value !== null,
  ),
  staleTime: 60_000,
});

const documentsQuery = useQuery({
  queryKey: computed(() => ["graph-documents", selectedKnowledgeBaseId.value]),
  queryFn: () => fetchDocuments(selectedKnowledgeBaseId.value ?? "", { limit: 100 }),
  enabled: computed(() => selectedKnowledgeBaseId.value !== null),
  staleTime: 120_000,
});

const knowledgeBases = computed(() => knowledgeBasesQuery.data.value?.items ?? []);
const knowledgeBasesLoading = computed(() => knowledgeBasesQuery.isPending.value);
const knowledgeBasesError = computed(() => knowledgeBasesQuery.isError.value);
const knowledgeBasesErrorMessage = computed(() =>
  getUserFacingError(knowledgeBasesQuery.error.value, "知识库暂时无法加载，请稍后重试。"),
);

const overview = computed(() => graphQuery.data.value ?? null);
const nodes = computed<GraphNode[]>(() => overview.value?.nodes ?? []);
const edges = computed(() => overview.value?.edges ?? []);
const typeOptions = computed(() => overview.value?.entity_types ?? []);
const isLoadingGraph = computed(
  () => graphQuery.isPending.value && selectedKnowledgeBaseId.value !== null,
);
const isRefetchingGraph = computed(() => graphQuery.isFetching.value && !isLoadingGraph.value);
const graphError = computed(() => graphQuery.isError.value);
const graphErrorMessage = computed(() =>
  getUserFacingError(graphQuery.error.value, "图谱暂时无法加载，请稍后重试。"),
);
const isTruncated = computed(() => overview.value?.truncated ?? false);
const legendItems = computed(() =>
  typeOptions.value.slice(0, LEGEND_LIMIT).map((item) => ({
    ...item,
    color: colorForType(item.entity_type),
  })),
);
const documents = computed(() => documentsQuery.data.value?.items ?? []);
const documentNames = computed<Record<string, string>>(() =>
  Object.fromEntries(
    documents.value.map((document) => [document.id, document.original_filename]),
  ),
);
const hasVisibleEdges = computed(() => edges.value.length > 0);
const showsWholeKnowledgeBase = computed(() => selectedDocumentId.value === "");
const mergeEdgeCount = computed(
  () => edges.value.filter((edge) => edge.kind === "canonical_bridge").length,
);
const entityDetail = computed(() => entityDetailQuery.data.value ?? null);
const detailPending = computed(() => entityDetailQuery.isFetching.value);
const hasActiveFilter = computed(() => searchTerm.value !== "" || selectedType.value !== "");

watch(
  knowledgeBases,
  (items) => {
    if (items.length === 0) {
      selectedKnowledgeBaseId.value = null;
      return;
    }
    const current = selectedKnowledgeBaseId.value;
    if (current === null || !items.some((item) => item.id === current)) {
      selectedKnowledgeBaseId.value = items[0].id;
    }
  },
  { immediate: true },
);

watch(selectedKnowledgeBaseId, (knowledgeBaseId) => {
  selectedEntityId.value = null;
  selectedType.value = "";
  searchInput.value = "";
  searchTerm.value = "";
  documentChoiceMade.value = false;
  if (knowledgeBaseId !== null) {
    void router.replace({ query: { ...route.query, kb: knowledgeBaseId } });
  }
});

watch(
  documents,
  (items) => {
    if (!documentChoiceMade.value && items.length > 0) {
      selectedDocumentId.value = items[0].id;
      documentChoiceMade.value = true;
    }
  },
  { immediate: true },
);

function onSearchInput(value: string | number): void {
  searchInput.value = String(value);
  if (searchTimer !== undefined) {
    window.clearTimeout(searchTimer);
  }
  searchTimer = window.setTimeout(() => {
    searchTerm.value = searchInput.value.trim();
  }, SEARCH_DEBOUNCE_MS);
}

function clearFilters(): void {
  if (searchTimer !== undefined) {
    window.clearTimeout(searchTimer);
  }
  searchInput.value = "";
  searchTerm.value = "";
  selectedType.value = "";
}

function selectDocument(documentId: string | number | null): void {
  documentChoiceMade.value = true;
  selectedDocumentId.value = documentId === null ? "" : String(documentId);
  selectedEntityId.value = null;
}

function selectNode(entityId: string | null): void {
  selectedEntityId.value = entityId;
}

function focusEntity(entityId: string): void {
  selectedEntityId.value = entityId;
  canvasRef.value?.focusSelected();
}

async function retryKnowledgeBases(): Promise<void> {
  await knowledgeBasesQuery.refetch();
}

async function retryGraph(): Promise<void> {
  await graphQuery.refetch();
}

onBeforeUnmount(() => {
  if (searchTimer !== undefined) {
    window.clearTimeout(searchTimer);
  }
});
</script>

<template>
  <section class="knowledge-graph-page">
    <header class="page-heading">
      <div class="page-heading-copy">
        <h1>知识图谱</h1>
        <p class="page-description">
          查看文档抽取出的实体与关系，点击节点可追溯来源证据。
        </p>
      </div>
    </header>

    <div
      v-if="knowledgeBasesLoading"
      class="state-panel loading-state"
      role="status"
      aria-busy="true"
    >
      <span class="sr-only">正在加载知识库</span>
      <div class="skeleton-block" />
    </div>

    <div
      v-else-if="knowledgeBasesError"
      class="state-panel error-state"
      role="alert"
    >
      <div class="state-symbol state-symbol-error">
        !
      </div>
      <div class="state-copy">
        <h3>知识库暂时无法加载</h3>
        <p>{{ knowledgeBasesErrorMessage }}</p>
      </div>
      <el-button @click="retryKnowledgeBases">
        重试
      </el-button>
    </div>

    <div
      v-else-if="knowledgeBases.length === 0"
      class="state-panel empty-state"
    >
      <div class="state-symbol state-symbol-empty">
        KB
      </div>
      <div class="state-copy">
        <h3>还没有知识库</h3>
        <p>先创建一个知识库并上传文档，图谱会随文档整理逐步生成。</p>
      </div>
      <RouterLink
        class="empty-action"
        :to="{ name: 'knowledge-bases' }"
      >
        去创建知识库
      </RouterLink>
    </div>

    <template v-else>
      <div class="filter-surface">
        <div class="filter-field is-wide">
          <label
            class="filter-label"
            for="graph-knowledge-base"
          >知识库</label>
          <ElSelect
            id="graph-knowledge-base"
            v-model="selectedKnowledgeBaseId"
            class="filter-control"
            placeholder="选择知识库"
          >
            <ElOption
              v-for="knowledgeBase in knowledgeBases"
              :key="knowledgeBase.id"
              :label="knowledgeBase.name"
              :value="knowledgeBase.id"
            />
          </ElSelect>
        </div>

        <div class="filter-field">
          <label
            class="filter-label"
            for="graph-document"
          >文档范围</label>
          <ElSelect
            id="graph-document"
            :model-value="selectedDocumentId"
            class="filter-control"
            placeholder="全部文档"
            @update:model-value="selectDocument"
          >
            <ElOption
              label="全部文档"
              value=""
            />
            <ElOption
              v-for="document in documents"
              :key="document.id"
              :label="document.original_filename"
              :value="document.id"
            />
          </ElSelect>
        </div>

        <div class="filter-field">
          <label
            class="filter-label"
            for="graph-search"
          >搜索实体</label>
          <ElInput
            id="graph-search"
            :model-value="searchInput"
            class="filter-control"
            placeholder="按名称或别名查找"
            clearable
            @update:model-value="onSearchInput"
          />
        </div>

        <div class="filter-field">
          <label
            class="filter-label"
            for="graph-entity-type"
          >实体类型</label>
          <ElSelect
            id="graph-entity-type"
            v-model="selectedType"
            class="filter-control"
            placeholder="全部类型"
          >
            <ElOption
              label="全部类型"
              value=""
            />
            <ElOption
              v-for="item in typeOptions"
              :key="item.entity_type"
              :label="`${item.entity_type}（${item.entity_count}）`"
              :value="item.entity_type"
            />
          </ElSelect>
        </div>

        <div class="filter-field is-narrow">
          <label
            class="filter-label"
            for="graph-limit"
          >显示数量</label>
          <ElSelect
            id="graph-limit"
            v-model="nodeLimit"
            class="filter-control"
          >
            <ElOption
              v-for="option in NODE_LIMIT_OPTIONS"
              :key="option"
              :label="`${option} 个实体`"
              :value="option"
            />
          </ElSelect>
        </div>

        <el-button
          v-if="hasActiveFilter"
          class="filter-clear"
          @click="clearFilters"
        >
          清空筛选
        </el-button>
      </div>

      <div class="graph-layout">
        <section
          class="canvas-surface"
          aria-label="知识图谱画布区域"
        >
          <div
            v-if="isLoadingGraph"
            class="state-panel loading-state"
            role="status"
            aria-busy="true"
          >
            <span class="sr-only">正在加载图谱</span>
            <div class="skeleton-canvas" />
          </div>

          <div
            v-else-if="graphError"
            class="state-panel error-state"
            role="alert"
          >
            <div class="state-symbol state-symbol-error">
              !
            </div>
            <div class="state-copy">
              <h3>图谱暂时无法加载</h3>
              <p>{{ graphErrorMessage }}</p>
            </div>
            <el-button @click="retryGraph">
              重试
            </el-button>
          </div>

          <div
            v-else-if="nodes.length === 0"
            class="state-panel empty-state"
          >
            <div class="state-symbol state-symbol-empty">
              ◎
            </div>
            <div class="state-copy">
              <h3>这个知识库还没有图谱内容</h3>
              <p>图谱随文档整理逐步生成，文档处理完成后再来查看即可。</p>
            </div>
          </div>

          <template v-else>
            <GraphCanvas
              ref="canvasRef"
              :nodes="nodes"
              :edges="edges"
              :selected-id="selectedEntityId"
              @select="selectNode"
            />
            <p
              v-if="isRefetchingGraph"
              class="refresh-hint"
              role="status"
            >
              正在更新图谱…
            </p>
            <p
              v-if="isTruncated"
              class="truncation-hint"
            >
              图谱较大，这里显示的是连接最紧密的 {{ nodes.length }} 个实体；可用搜索或类型筛选缩小范围。
            </p>
            <p
              v-if="!hasVisibleEdges && showsWholeKnowledgeBase"
              class="truncation-hint"
            >
              实体关系按文档组织，整个知识库范围内通常没有直接连线；选择一个文档即可看到该文档内部的实体关系。
            </p>
          </template>

          <ul
            v-if="legendItems.length > 0 && nodes.length > 0"
            class="legend-list"
          >
            <li
              v-for="item in legendItems"
              :key="item.entity_type"
              class="legend-item"
            >
              <span
                class="legend-dot"
                aria-hidden="true"
                :style="{ backgroundColor: item.color }"
              />
              {{ item.entity_type }}
              <span class="legend-count">{{ item.entity_count }}</span>
            </li>
            <li
              v-if="mergeEdgeCount > 0"
              class="legend-item"
            >
              <span
                class="legend-line"
                aria-hidden="true"
              />
              虚线为跨文档归并的同一实体（{{ mergeEdgeCount }} 条）
            </li>
          </ul>
        </section>

        <div class="panel-dock">
          <EntityDetailPanel
            :detail="entityDetail"
            :pending="detailPending"
            :document-names="documentNames"
            @select="selectNode"
            @focus="focusEntity"
            @close="selectNode(null)"
          />
        </div>
      </div>
    </template>
  </section>
</template>

<style scoped>
.knowledge-graph-page {
  display: flex;
  flex-direction: column;
  gap: var(--ks-space-5);
}

.page-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--ks-space-4);
}

.page-heading-copy h1 {
  margin: 0;
  color: var(--ks-ink);
  font-size: 26px;
  font-weight: 720;
  letter-spacing: -0.02em;
}

.page-description {
  margin: 6px 0 0;
  color: var(--ks-muted);
  font-size: 14px;
}

.filter-surface {
  display: grid;
  grid-template-columns:
    minmax(200px, 1.3fr) minmax(180px, 1.3fr) minmax(150px, 1.1fr)
    minmax(140px, 1fr) 130px auto;
  align-items: end;
  gap: var(--ks-space-3);
  padding: var(--ks-space-4);
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-md);
}

.filter-field {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 6px;
}

.filter-label {
  color: var(--ks-muted);
  font-size: 12px;
  font-weight: 620;
}

.filter-control {
  width: 100%;
}

.filter-clear {
  align-self: flex-end;
}

.graph-layout {
  position: relative;
  display: block;
}

.canvas-surface {
  display: flex;
  flex-direction: column;
  gap: var(--ks-space-3);
  min-width: 0;
}

/* The detail panel floats over the canvas so the graph keeps the full width,
   starting below the canvas toolbar. */
.panel-dock {
  position: absolute;
  top: 58px;
  right: var(--ks-space-3);
  width: 300px;
  max-height: calc(100% - 74px);
  overflow-y: auto;
  box-shadow: 0 12px 32px rgb(32 38 34 / 10%);
  border-radius: var(--ks-radius-md);
}

.state-panel {
  display: flex;
  align-items: center;
  gap: var(--ks-space-4);
  padding: var(--ks-space-6);
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-md);
}

.loading-state,
.empty-state,
.error-state {
  flex-direction: column;
  align-items: flex-start;
  gap: var(--ks-space-3);
}

.state-symbol {
  display: grid;
  width: 40px;
  height: 40px;
  place-items: center;
  font-size: 14px;
  font-weight: 700;
  border-radius: var(--ks-radius-sm);
}

.state-symbol-empty {
  color: var(--ks-accent-strong);
  background: var(--ks-accent-soft);
}

.state-symbol-error {
  color: var(--ks-danger);
  background: #f7ecea;
}

.state-copy h3 {
  margin: 0 0 4px;
  color: var(--ks-ink);
  font-size: 16px;
  font-weight: 680;
}

.state-copy p {
  margin: 0;
  color: var(--ks-muted);
  font-size: 13px;
  line-height: 1.7;
}

.empty-action {
  display: inline-flex;
  align-items: center;
  min-height: 36px;
  padding: 0 16px;
  color: var(--ks-accent-strong);
  font-size: 13px;
  font-weight: 650;
  background: var(--ks-accent-soft);
  border-radius: var(--ks-radius-sm);
}

.skeleton-block,
.skeleton-canvas {
  width: 100%;
  background: var(--ks-surface-muted);
  border-radius: var(--ks-radius-sm);
  animation: skeleton-pulse 1.4s ease-in-out infinite;
}

.skeleton-block {
  height: 88px;
}

.skeleton-canvas {
  height: 420px;
}

.refresh-hint,
.truncation-hint {
  margin: 0;
  color: var(--ks-muted);
  font-size: 12px;
}

.legend-list {
  display: flex;
  flex-wrap: wrap;
  gap: var(--ks-space-2) var(--ks-space-4);
  margin: 0;
  padding: var(--ks-space-3) 0 0;
  list-style: none;
  border-top: 1px solid var(--ks-border);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--ks-text);
  font-size: 12px;
}

.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
}

.legend-count {
  color: var(--ks-faint);
}

.legend-line {
  width: 18px;
  height: 0;
  border-top: 1px dashed var(--ks-accent);
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
}

@keyframes skeleton-pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.55;
  }
}

@media (max-width: 1080px) {
  .filter-surface {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .panel-dock {
    position: static;
    width: auto;
    max-height: none;
    margin-top: var(--ks-space-3);
    overflow: visible;
    box-shadow: none;
  }
}

@media (max-width: 680px) {
  .filter-surface {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
