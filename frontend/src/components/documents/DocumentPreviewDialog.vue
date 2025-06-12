<script setup lang="ts">
import { useQuery } from "@tanstack/vue-query";
import { computed, ref, watch } from "vue";
import { ElButton, ElDialog } from "element-plus";

import {
  ApiError,
  documentFileUrl,
  fetchDocumentChunks,
  getUserFacingError,
  probeDocumentFile,
} from "../../api/client";
import type { Document, DocumentChunk } from "../../api/types";

const CONTENT_TYPE_LABELS: Record<string, string> = {
  title: "标题",
  text: "正文",
  table: "表格",
  formula: "公式",
  image: "图片",
};

const props = defineProps<{
  modelValue: boolean;
  knowledgeBaseId: string;
  document: Document | null;
}>();

const emit = defineEmits<{ "update:modelValue": [value: boolean] }>();

const visible = computed({
  get: () => props.modelValue,
  set: (value: boolean) => emit("update:modelValue", value),
});

const activePage = ref(1);
const documentId = computed(() => props.document?.id ?? "");

const chunksQuery = useQuery({
  queryKey: computed(() => ["document-chunks", props.knowledgeBaseId, documentId.value]),
  queryFn: () => fetchDocumentChunks(props.knowledgeBaseId, documentId.value),
  enabled: computed(() => props.modelValue && documentId.value !== ""),
});

const fileQuery = useQuery({
  queryKey: computed(() => ["document-file", props.knowledgeBaseId, documentId.value]),
  queryFn: () => probeDocumentFile(props.knowledgeBaseId, documentId.value),
  enabled: computed(() => props.modelValue && documentId.value !== ""),
  retry: false,
});

const isFileReady = computed(() => props.modelValue && fileQuery.isSuccess.value);
const fileError = computed(() =>
  props.modelValue && fileQuery.isError.value
    ? describeFileError(fileQuery.error.value)
    : null,
);

const chunks = computed(() => chunksQuery.data.value?.items ?? []);
const pageCount = computed(() => chunksQuery.data.value?.page_count ?? null);
const chunksLoading = computed(() => props.modelValue && chunksQuery.isPending.value);
const chunksError = computed(() =>
  props.modelValue && chunksQuery.isError.value
    ? getUserFacingError(chunksQuery.error.value, "文档切片暂时无法加载。")
    : null,
);
const previewUrl = computed(() =>
  props.document === null
    ? ""
    : documentFileUrl(props.knowledgeBaseId, props.document.id, activePage.value),
);
const canGoNext = computed(() => pageCount.value === null || activePage.value < pageCount.value);

watch(
  () => [props.modelValue, documentId.value] as const,
  ([isVisible]) => {
    if (isVisible) {
      activePage.value = 1;
    }
  },
);

function describeFileError(error: unknown): string {
  if (error instanceof ApiError && error.status === 404 && error.detail) {
    return error.detail;
  }
  return getUserFacingError(error, "该文档暂时无法预览，请稍后重试。");
}

function openPage(page: number): void {
  const lowerBounded = Math.max(1, Math.floor(page));
  activePage.value =
    pageCount.value === null ? lowerBounded : Math.min(lowerBounded, pageCount.value);
}

function selectChunk(chunk: DocumentChunk): void {
  openPage(chunk.page_start);
}

function isChunkActive(chunk: DocumentChunk): boolean {
  return activePage.value >= chunk.page_start && activePage.value <= chunk.page_end;
}

function pageLabel(chunk: DocumentChunk): string {
  return chunk.page_start === chunk.page_end
    ? "第 " + chunk.page_start + " 页"
    : "第 " + chunk.page_start + "–" + chunk.page_end + " 页";
}

function contentTypeLabel(contentType: string): string {
  return CONTENT_TYPE_LABELS[contentType] ?? contentType;
}

function excerpt(chunk: DocumentChunk): string {
  const normalized = chunk.text.replace(/\s+/g, " ").trim();
  if (normalized) {
    return normalized;
  }
  return chunk.asset_refs.length > 0 ? "该切片由图片或表格资源组成。" : "该切片没有文本内容。";
}
</script>

<template>
  <el-dialog
    v-model="visible"
    class="document-preview"
    :title="props.document?.original_filename ?? '文档预览'"
    width="min(1120px, 94vw)"
    top="5vh"
    destroy-on-close
  >
    <div
      v-if="props.document"
      class="document-preview__body"
    >
      <aside
        class="document-preview__chunks"
        aria-label="文档切片"
      >
        <div class="document-preview__chunks-head">
          <h3>切片</h3>
          <span v-if="pageCount !== null">共 {{ pageCount }} 页</span>
        </div>

        <p
          v-if="chunksLoading"
          class="document-preview__hint"
          role="status"
        >
          正在加载切片…
        </p>
        <p
          v-else-if="chunksError"
          class="document-preview__hint document-preview__hint--error"
          role="alert"
        >
          {{ chunksError }}
        </p>
        <p
          v-else-if="chunks.length === 0"
          class="document-preview__hint"
        >
          该文档还没有切片，仍可预览原始 PDF。
        </p>
        <ul
          v-else
          class="document-preview__chunk-list"
        >
          <li
            v-for="chunk in chunks"
            :key="chunk.chunk_id"
          >
            <button
              class="document-preview__chunk"
              :class="{ 'is-active': isChunkActive(chunk) }"
              type="button"
              @click="selectChunk(chunk)"
            >
              <span class="document-preview__chunk-top">
                <span class="document-preview__page">{{ pageLabel(chunk) }}</span>
                <span class="document-preview__types">
                  <span
                    v-for="contentType in chunk.content_types"
                    :key="contentType"
                  >{{ contentTypeLabel(contentType) }}</span>
                </span>
              </span>
              <span
                v-if="chunk.section_path.length > 0"
                class="document-preview__section"
              >{{ chunk.section_path.join(" / ") }}</span>
              <span class="document-preview__excerpt">{{ excerpt(chunk) }}</span>
            </button>
          </li>
        </ul>
      </aside>

      <section
        class="document-preview__viewer"
        aria-label="PDF 预览"
      >
        <div class="document-preview__viewer-bar">
          <span class="document-preview__viewer-page">
            第 {{ activePage }} 页<span v-if="pageCount !== null"> / 共 {{ pageCount }} 页</span>
          </span>
          <div class="document-preview__viewer-actions">
            <el-button
              size="small"
              :disabled="activePage <= 1"
              @click="openPage(activePage - 1)"
            >
              上一页
            </el-button>
            <el-button
              size="small"
              :disabled="!canGoNext"
              @click="openPage(activePage + 1)"
            >
              下一页
            </el-button>
            <el-button
              size="small"
              tag="a"
              :href="previewUrl"
              target="_blank"
              rel="noopener"
            >
              新窗口打开
            </el-button>
          </div>
        </div>
        <div
          v-if="isFileReady"
          class="document-preview__frame-wrap"
        >
          <iframe
            class="document-preview__frame"
            :src="previewUrl"
            :title="(props.document?.original_filename ?? 'PDF') + ' 预览'"
          />
        </div>
        <p
          v-else-if="fileError"
          class="document-preview__viewer-state document-preview__viewer-state--error"
          role="alert"
        >
          {{ fileError }}
        </p>
        <p
          v-else
          class="document-preview__viewer-state"
          role="status"
        >
          正在准备 PDF 预览…
        </p>
      </section>
    </div>
  </el-dialog>
</template>

<style scoped>
.document-preview__body {
  display: grid;
  grid-template-columns: minmax(240px, 300px) minmax(0, 1fr);
  gap: 16px;
  min-height: 0;
}

.document-preview__chunks {
  display: flex;
  max-height: min(72vh, 720px);
  flex-direction: column;
  gap: 10px;
  padding: 12px;
  overflow: hidden;
  background: var(--ks-surface-subtle);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-md);
}

.document-preview__chunks-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}

.document-preview__chunks-head h3 {
  margin: 0;
  color: var(--ks-ink);
  font-size: 13px;
  font-weight: 680;
}

.document-preview__chunks-head span {
  color: var(--ks-muted);
  font-size: 11px;
}

.document-preview__hint {
  margin: 0;
  padding: 10px 2px;
  color: var(--ks-muted);
  font-size: 12px;
  line-height: 1.6;
}

.document-preview__hint--error {
  color: var(--ks-danger);
}

.document-preview__chunk-list {
  display: flex;
  margin: 0;
  padding: 0 2px 0 0;
  flex-direction: column;
  gap: 6px;
  overflow-y: auto;
  list-style: none;
}

.document-preview__chunk {
  display: flex;
  width: 100%;
  padding: 9px 10px;
  flex-direction: column;
  gap: 4px;
  color: inherit;
  text-align: left;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-sm);
  cursor: pointer;
  transition: border-color var(--ks-duration-fast) var(--ks-ease-out),
    background-color var(--ks-duration-fast) var(--ks-ease-out);
}

.document-preview__chunk:hover {
  border-color: var(--ks-border-strong);
}

.document-preview__chunk.is-active {
  background: var(--ks-accent-soft);
  border-color: #c5dbd3;
}

.document-preview__chunk-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.document-preview__page {
  color: var(--ks-accent-strong);
  font-size: 11px;
  font-weight: 680;
}

.document-preview__types {
  display: flex;
  gap: 4px;
}

.document-preview__types span {
  padding: 1px 5px;
  color: var(--ks-muted);
  font-size: 10px;
  background: var(--ks-surface-muted);
  border-radius: 999px;
}

.document-preview__section {
  overflow: hidden;
  color: var(--ks-muted);
  font-size: 11px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.document-preview__excerpt {
  display: -webkit-box;
  overflow: hidden;
  color: var(--ks-text);
  font-size: 12px;
  line-height: 1.55;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 3;
}

.document-preview__viewer {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 10px;
}

.document-preview__viewer-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.document-preview__viewer-page {
  color: var(--ks-muted);
  font-size: 12px;
}

.document-preview__viewer-actions {
  display: flex;
  gap: 8px;
}

.document-preview__frame-wrap {
  display: flex;
  min-height: 0;
  flex: 1;
}

.document-preview__frame {
  width: 100%;
  height: min(72vh, 720px);
  background: var(--ks-surface-muted);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-md);
}

.document-preview__viewer-state {
  display: grid;
  height: min(72vh, 720px);
  margin: 0;
  padding: 24px;
  place-items: center;
  color: var(--ks-muted);
  font-size: 13px;
  line-height: 1.7;
  text-align: center;
  background: var(--ks-surface-subtle);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-md);
}

.document-preview__viewer-state--error {
  color: var(--ks-danger);
  background: #f8eae8;
  border-color: #edd7d4;
}

@media (max-width: 900px) {
  .document-preview__body {
    grid-template-columns: minmax(0, 1fr);
  }

  .document-preview__chunks {
    max-height: 240px;
  }

  .document-preview__frame {
    height: 60vh;
  }
}
</style>
