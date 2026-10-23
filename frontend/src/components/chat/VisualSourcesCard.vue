<script setup lang="ts">
import { computed, ref } from "vue";

import { documentFileUrl, evidenceAssetUrl } from "../../api/client";
import type { RAGFactStatus, RAGSafetyGate, RAGVisualSource } from "../../api/types";

const props = defineProps<{
  sources: RAGVisualSource[];
  notice: string | null;
  gate: RAGSafetyGate;
  factStatus: RAGFactStatus | null;
}>();

const emit = defineEmits<{ open: [source: RAGVisualSource] }>();

const failedAssets = ref<Set<string>>(new Set());

function assetKey(source: RAGVisualSource): string {
  return `${source.document_id}:${source.evidence_id}:${source.asset_index}`;
}

function imageUrl(source: RAGVisualSource): string {
  return evidenceAssetUrl(
    source.knowledge_base_id,
    source.document_id,
    source.evidence_id,
    source.asset_index,
  );
}

function imageFailed(source: RAGVisualSource): boolean {
  return failedAssets.value.has(assetKey(source));
}

function markFailed(source: RAGVisualSource, event: Event): void {
  event.stopPropagation();
  const next = new Set(failedAssets.value);
  next.add(assetKey(source));
  failedAssets.value = next;
}

function openSource(source: RAGVisualSource): void {
  if (!imageFailed(source)) {
    emit("open", source);
  }
}

function openOriginal(source: RAGVisualSource): void {
  window.open(
    documentFileUrl(source.knowledge_base_id, source.document_id, source.page_start),
    "_blank",
    "noopener,noreferrer",
  );
}

const noticeText = computed(() => {
  if (props.notice) {
    return props.notice;
  }
  if (props.gate === "visual_numeric_withheld") {
    return "视觉数值未经核验，已省略精确数字并标注为视觉估计";
  }
  if (props.gate === "vision_unavailable") {
    return "视觉分析未完成，已阻止未核验的数值输出";
  }
  return null;
});

function sourceLabel(source: RAGVisualSource): string {
  return source.label ?? `第 ${source.page_start} 页图片`;
}
</script>

<template>
  <section
    class="visual-sources"
    aria-label="本次分析的图片"
  >
    <p
      v-if="noticeText"
      class="visual-sources__notice"
      role="note"
    >
      ⚠️ {{ noticeText }}
    </p>
    <h4 class="visual-sources__title">
      本次分析图片
      <span
        v-if="factStatus === 'visual_estimate' || factStatus === 'unverified'"
        class="visual-sources__badge"
      >
        视觉分析结果需核对
      </span>
    </h4>
    <div class="visual-sources__grid">
      <article
        v-for="source in sources"
        :key="assetKey(source)"
        class="visual-sources__card"
      >
        <button
          v-if="!imageFailed(source)"
          type="button"
          class="visual-sources__thumb"
          :aria-label="`查看${sourceLabel(source)}`"
          @click="openSource(source)"
        >
          <img
            :src="imageUrl(source)"
            :alt="sourceLabel(source)"
            loading="lazy"
            @error="markFailed(source, $event)"
          >
        </button>
        <div
          v-else
          class="visual-sources__missing"
        >
          原图不可用，无法核验
        </div>
        <div class="visual-sources__meta">
          <span>{{ sourceLabel(source) }}</span>
          <span class="visual-sources__page">第 {{ source.page_start }} 页</span>
        </div>
        <div class="visual-sources__actions">
          <button
            v-if="!imageFailed(source)"
            type="button"
            class="visual-sources__action"
            @click="openSource(source)"
          >
            查看原图
          </button>
          <button
            type="button"
            class="visual-sources__action"
            @click="openOriginal(source)"
          >
            打开原文 PDF
          </button>
        </div>
      </article>
    </div>
  </section>
</template>

<style scoped>
.visual-sources {
  margin-top: 12px;
  padding: 12px;
  background: var(--ks-surface-muted, #f7f5ef);
  border: 1px solid var(--ks-border, #e3ded2);
  border-radius: 10px;
}

.visual-sources__notice {
  margin: 0 0 10px;
  padding: 8px 10px;
  color: var(--ks-warning-text, #8a5a00);
  background: var(--ks-warning-soft, #fdf3dd);
  border-radius: 8px;
  font-size: 13px;
  line-height: 1.5;
}

.visual-sources__title {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0 0 10px;
  color: var(--ks-text);
  font-size: 13px;
  font-weight: 600;
}

.visual-sources__badge {
  padding: 2px 6px;
  color: var(--ks-warning-text, #8a5a00);
  background: var(--ks-warning-soft, #fdf3dd);
  border-radius: 6px;
  font-size: 12px;
  font-weight: 500;
}

.visual-sources__grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
}

.visual-sources__card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px;
  background: var(--ks-surface, #fff);
  border: 1px solid var(--ks-border, #e3ded2);
  border-radius: 8px;
}

.visual-sources__thumb {
  overflow: hidden;
  padding: 0;
  background: transparent;
  border: none;
  border-radius: 6px;
  cursor: zoom-in;
}

.visual-sources__thumb img {
  display: block;
  width: 100%;
  max-height: 180px;
  object-fit: contain;
}

.visual-sources__thumb:focus-visible {
  outline: 2px solid var(--ks-accent);
  outline-offset: 2px;
}

.visual-sources__missing {
  display: grid;
  place-items: center;
  min-height: 90px;
  color: var(--ks-muted);
  background: var(--ks-surface-muted, #f7f5ef);
  border-radius: 6px;
  font-size: 12px;
  text-align: center;
}

.visual-sources__meta {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  color: var(--ks-muted);
  font-size: 12px;
}

.visual-sources__actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.visual-sources__action {
  padding: 4px 8px;
  color: var(--ks-text);
  background: transparent;
  border: 1px solid var(--ks-border, #e3ded2);
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
}

.visual-sources__action:hover {
  border-color: var(--ks-accent);
}

.visual-sources__action:focus-visible {
  outline: 2px solid var(--ks-accent);
  outline-offset: 2px;
}
</style>
