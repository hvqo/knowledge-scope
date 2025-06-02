<script setup lang="ts">
import type { RAGCitation } from "../../api/types";
import EvidenceCard from "./EvidenceCard.vue";

defineProps<{
  citations: RAGCitation[];
  selectedMarker: string | null;
  documentTitles: Record<string, string>;
}>();

const emit = defineEmits<{ selectCitation: [marker: string] }>();
</script>

<template>
  <aside class="evidence-panel">
    <div class="evidence-panel__header">
      <div>
        <span class="eyebrow">来源</span>
        <h2>证据</h2>
      </div>
      <span
        v-if="citations.length > 0"
        class="evidence-count"
      >
        {{ citations.length }}
      </span>
    </div>

    <div
      v-if="citations.length === 0"
      class="evidence-empty"
    >
      <div
        class="evidence-empty__mark"
        aria-hidden="true"
      >
        ⌁
      </div>
      <p>回答完成后显示来源</p>
      <span>点击回答中的引用标记，可定位对应证据。</span>
    </div>
    <div
      v-else
      class="evidence-list"
      aria-label="回答来源"
    >
      <EvidenceCard
        v-for="citation in citations"
        :key="citation.marker"
        :citation="citation"
        :selected="citation.marker === selectedMarker"
        :document-title="documentTitles[citation.document_id] ?? null"
        @select="emit('selectCitation', citation.marker)"
      />
    </div>
  </aside>
</template>

<style scoped>
.evidence-panel {
  display: flex;
  min-width: 274px;
  width: 302px;
  flex-direction: column;
  gap: 18px;
  padding: 22px 16px 18px;
  background: var(--ks-surface-subtle);
  border-left: 1px solid var(--ks-border);
}

.evidence-panel__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.eyebrow {
  color: var(--ks-accent);
  font-size: 11px;
  font-weight: 720;
  letter-spacing: 0.08em;
}

h2 {
  margin: 4px 0 0;
  color: var(--ks-ink);
  font-size: 20px;
  letter-spacing: -0.04em;
}

.evidence-count {
  display: grid;
  min-width: 25px;
  height: 25px;
  padding: 0 7px;
  place-items: center;
  color: var(--ks-accent-strong);
  font-size: 11px;
  font-weight: 720;
  background: var(--ks-accent-soft);
  border-radius: 999px;
}

.evidence-list {
  display: flex;
  min-height: 0;
  flex: 1;
  flex-direction: column;
  gap: 10px;
  overflow-y: auto;
}

.evidence-empty {
  display: flex;
  margin: auto 4px;
  align-items: center;
  color: var(--ks-muted);
  font-size: 12px;
  line-height: 1.65;
  text-align: center;
  flex-direction: column;
}

.evidence-empty__mark {
  display: grid;
  width: 42px;
  height: 42px;
  place-items: center;
  color: var(--ks-accent);
  font-size: 25px;
  background: var(--ks-accent-soft);
  border-radius: 50%;
}

.evidence-empty p {
  margin: 12px 0 3px;
  color: var(--ks-text);
  font-weight: 650;
}

.evidence-empty span {
  max-width: 190px;
}

@media (max-width: 1200px) {
  .evidence-panel {
    min-width: 250px;
    width: 270px;
  }
}
</style>
