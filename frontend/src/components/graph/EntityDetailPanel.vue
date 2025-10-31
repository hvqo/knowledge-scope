<script setup lang="ts">
import { computed } from "vue";

import type { GraphEntityDetail, GraphNeighborDirection } from "../../api/types";
import { colorForType } from "./graphPalette";

const props = defineProps<{
  detail: GraphEntityDetail | null;
  pending: boolean;
  documentNames: Record<string, string>;
}>();

const emit = defineEmits<{
  select: [entityId: string];
  focus: [entityId: string];
  close: [];
}>();

const DIRECTION_LABELS: Record<GraphNeighborDirection, string> = {
  forward: "指向",
  reverse: "来自",
  canonical_bridge: "同一实体",
};

const entity = computed(() => props.detail?.entity ?? null);

const neighborSummary = computed(() => {
  const neighbors = props.detail?.neighbors ?? [];
  return neighbors.reduce<Record<string, { count: number; label: string }>>((accumulator, item) => {
    const label = item.relation_type ?? DIRECTION_LABELS[item.direction];
    const existing = accumulator[item.entity_id];
    if (existing) {
      existing.count += 1;
      return accumulator;
    }
    accumulator[item.entity_id] = { count: 1, label };
    return accumulator;
  }, {});
});

const uniqueNeighbors = computed(() => {
  const neighbors = props.detail?.neighbors ?? [];
  const seen = new Set<string>();
  return neighbors.filter((item) => {
    if (seen.has(item.entity_id)) {
      return false;
    }
    seen.add(item.entity_id);
    return true;
  });
});

const evidenceItems = computed(() => props.detail?.evidence ?? []);
const canonicalEntities = computed(() => props.detail?.canonical_entities ?? []);

function pageLabel(item: { page_start: number | null; page_end: number | null }): string {
  if (item.page_start === null || item.page_end === null) {
    return "";
  }
  return item.page_start === item.page_end
    ? `第 ${item.page_start} 页`
    : `第 ${item.page_start}-${item.page_end} 页`;
}

function documentLabel(documentId: string): string {
  return props.documentNames[documentId] ?? "";
}
</script>

<template>
  <aside
    class="entity-panel"
    aria-label="实体详情"
  >
    <div
      v-if="!entity"
      class="panel-empty"
    >
      <p class="panel-empty-title">
        选择一个节点
      </p>
      <p class="panel-empty-copy">
        点击图谱中的节点，这里会显示它的类型、别名、来源证据和相邻实体。
      </p>
    </div>

    <template v-else>
      <header class="panel-header">
        <div class="panel-heading">
          <span
            class="type-dot"
            aria-hidden="true"
            :style="{ backgroundColor: colorForType(entity.entity_type) }"
          />
          <div class="panel-heading-copy">
            <h2 class="panel-title">
              {{ entity.canonical_name }}
            </h2>
            <p class="panel-subtitle">
              {{ entity.entity_type }}
            </p>
          </div>
        </div>
        <button
          type="button"
          class="panel-close"
          aria-label="关闭详情"
          @click="emit('close')"
        >
          ✕
        </button>
      </header>

      <dl class="metric-row">
        <div class="metric">
          <dt>相邻实体</dt>
          <dd>{{ entity.degree }}</dd>
        </div>
        <div class="metric">
          <dt>来源证据</dt>
          <dd>{{ entity.evidence_count }}</dd>
        </div>
      </dl>

      <button
        type="button"
        class="panel-focus"
        @click="emit('focus', entity.entity_id)"
      >
        在画布中聚焦该实体
      </button>

      <p
        v-if="pending"
        class="panel-pending"
        role="status"
      >
        正在更新详情…
      </p>

      <section
        v-if="entity.aliases.length > 0"
        class="panel-section"
      >
        <h3 class="section-title">
          别名
        </h3>
        <ul class="alias-list">
          <li
            v-for="alias in entity.aliases"
            :key="alias"
            class="alias-item"
          >
            {{ alias }}
          </li>
        </ul>
      </section>

      <section
        v-if="canonicalEntities.length > 0"
        class="panel-section"
      >
        <h3 class="section-title">
          已归并实体
        </h3>
        <ul class="plain-list">
          <li
            v-for="item in canonicalEntities"
            :key="item.canonical_entity_id"
          >
            {{ item.canonical_name }}<span class="list-hint">{{ item.entity_type }}</span>
          </li>
        </ul>
      </section>

      <section
        v-if="evidenceItems.length > 0"
        class="panel-section"
      >
        <h3 class="section-title">
          来源证据
        </h3>
        <ul class="evidence-list">
          <li
            v-for="item in evidenceItems"
            :key="item.chunk_id"
            class="evidence-item"
          >
            <span class="evidence-document">
              {{ documentLabel(item.document_id) || "来自文档" }}
            </span>
            <span class="evidence-page">{{ pageLabel(item) }}</span>
          </li>
        </ul>
      </section>

      <section
        v-if="uniqueNeighbors.length > 0"
        class="panel-section"
      >
        <h3 class="section-title">
          相邻实体
        </h3>
        <ul class="neighbor-list">
          <li
            v-for="item in uniqueNeighbors"
            :key="item.entity_id"
          >
            <button
              type="button"
              class="neighbor-button"
              @click="emit('select', item.entity_id)"
            >
              <span
                class="type-dot"
                aria-hidden="true"
                :style="{ backgroundColor: colorForType(item.entity_type) }"
              />
              <span class="neighbor-copy">
                <strong>{{ item.canonical_name }}</strong>
                <span class="neighbor-meta">
                  {{ neighborSummary[item.entity_id]?.label }}
                  <template v-if="item.shared_canonical_name">
                    · {{ item.shared_canonical_name }}
                  </template>
                  <template v-else-if="pageLabel(item)"> · {{ pageLabel(item) }}</template>
                </span>
              </span>
            </button>
          </li>
        </ul>
      </section>
    </template>
  </aside>
</template>

<style scoped>
.entity-panel {
  display: flex;
  flex-direction: column;
  gap: var(--ks-space-4);
  padding: var(--ks-space-4);
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-md);
}

.panel-empty {
  padding: var(--ks-space-3) 0;
}

.panel-empty-title {
  margin: 0 0 var(--ks-space-1);
  color: var(--ks-ink);
  font-size: 15px;
  font-weight: 680;
}

.panel-empty-copy {
  margin: 0;
  color: var(--ks-muted);
  font-size: 13px;
  line-height: 1.7;
}

.panel-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--ks-space-2);
}

.panel-heading {
  display: flex;
  align-items: flex-start;
  gap: var(--ks-space-2);
}

.type-dot {
  flex: 0 0 10px;
  width: 10px;
  height: 10px;
  margin-top: 6px;
  border-radius: 999px;
}

.panel-heading-copy {
  min-width: 0;
}

.panel-title {
  margin: 0;
  color: var(--ks-ink);
  font-size: 17px;
  font-weight: 700;
  overflow-wrap: anywhere;
}

.panel-subtitle {
  margin: 2px 0 0;
  color: var(--ks-muted);
  font-size: 13px;
}

.panel-close {
  display: grid;
  width: 28px;
  height: 28px;
  padding: 0;
  place-items: center;
  color: var(--ks-muted);
  font-size: 13px;
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--ks-radius-sm);
  cursor: pointer;
  transition: color var(--ks-duration-fast) var(--ks-ease-out),
    background-color var(--ks-duration-fast) var(--ks-ease-out);
}

.panel-close:hover {
  color: var(--ks-ink);
  background: var(--ks-surface-subtle);
}

.metric-row {
  display: flex;
  gap: var(--ks-space-4);
  margin: 0;
}

.metric dt {
  color: var(--ks-muted);
  font-size: 12px;
}

.metric dd {
  margin: 2px 0 0;
  color: var(--ks-ink);
  font-size: 18px;
  font-weight: 700;
}

.panel-focus {
  min-height: 32px;
  padding: 0 12px;
  color: var(--ks-accent-strong);
  font-size: 13px;
  font-weight: 650;
  background: var(--ks-accent-soft);
  border: 1px solid transparent;
  border-radius: var(--ks-radius-sm);
  cursor: pointer;
  transition: background-color var(--ks-duration-fast) var(--ks-ease-out);
}

.panel-focus:hover {
  background: #d8e8e2;
}

.panel-pending {
  margin: 0;
  color: var(--ks-muted);
  font-size: 12px;
}

.panel-section {
  display: flex;
  flex-direction: column;
  gap: var(--ks-space-2);
  padding-top: var(--ks-space-3);
  border-top: 1px solid var(--ks-border);
}

.section-title {
  margin: 0;
  color: var(--ks-ink);
  font-size: 13px;
  font-weight: 680;
}

.alias-list,
.plain-list,
.evidence-list,
.neighbor-list {
  display: flex;
  flex-direction: column;
  gap: var(--ks-space-1);
  margin: 0;
  padding: 0;
  list-style: none;
}

.alias-item {
  display: inline-flex;
  align-self: flex-start;
  padding: 2px 8px;
  color: var(--ks-text);
  font-size: 12px;
  background: var(--ks-surface-subtle);
  border: 1px solid var(--ks-border);
  border-radius: 999px;
}

.plain-list li,
.evidence-item {
  color: var(--ks-text);
  font-size: 13px;
  line-height: 1.6;
}

.list-hint {
  margin-left: 6px;
  color: var(--ks-faint);
  font-size: 12px;
}

.evidence-item {
  display: flex;
  align-items: baseline;
  gap: var(--ks-space-2);
}

.evidence-document {
  flex: 1;
  min-width: 0;
  overflow-wrap: anywhere;
}

.evidence-page {
  flex: 0 0 auto;
  color: var(--ks-muted);
  font-size: 12px;
}

.neighbor-button {
  display: flex;
  width: 100%;
  align-items: center;
  gap: var(--ks-space-2);
  padding: 6px 8px;
  color: inherit;
  text-align: left;
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--ks-radius-sm);
  cursor: pointer;
  transition: background-color var(--ks-duration-fast) var(--ks-ease-out),
    border-color var(--ks-duration-fast) var(--ks-ease-out);
}

.neighbor-button:hover {
  background: var(--ks-surface-subtle);
  border-color: var(--ks-border);
}

.neighbor-copy {
  display: flex;
  min-width: 0;
  flex-direction: column;
}

.neighbor-copy strong {
  color: var(--ks-ink);
  font-size: 13px;
  font-weight: 650;
  overflow-wrap: anywhere;
}

.neighbor-meta {
  color: var(--ks-muted);
  font-size: 12px;
}
</style>
