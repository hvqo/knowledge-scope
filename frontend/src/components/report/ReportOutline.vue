<script setup lang="ts">
import type { ReportSection } from "../../api/types";

defineProps<{
  title: string;
  sections: ReportSection[];
  activeSectionId: string | null;
  disabled: boolean;
}>();

const emit = defineEmits<{
  "update:title": [value: string];
  select: [id: string];
  add: [];
  delete: [];
}>();
</script>

<template>
  <aside class="report-outline">
    <div class="report-outline__eyebrow">
      报告大纲
    </div>
    <label
      class="report-outline__label"
      for="report-title"
    >报告标题</label>
    <input
      id="report-title"
      class="report-outline__title-input"
      :value="title"
      :disabled="disabled"
      aria-label="报告标题"
      @input="emit('update:title', ($event.target as HTMLInputElement).value)"
    >

    <div class="report-outline__section-head">
      <span>章节</span>
      <span class="report-outline__count">{{ sections.length }}</span>
    </div>
    <nav
      v-if="sections.length > 0"
      class="report-outline__sections"
      aria-label="报告章节"
    >
      <button
        v-for="section in sections"
        :key="section.id"
        type="button"
        class="report-outline__section"
        :class="{ 'is-active': section.id === activeSectionId }"
        :disabled="disabled"
        @click="emit('select', section.id)"
      >
        <span class="report-outline__section-number">{{ section.position + 1 }}</span>
        <span class="report-outline__section-title">{{ section.title }}</span>
      </button>
    </nav>
    <p
      v-else
      class="report-outline__empty"
    >
      还没有章节，先添加一个章节。
    </p>

    <div class="report-outline__actions">
      <button
        type="button"
        class="report-outline__add"
        :disabled="disabled"
        @click="emit('add')"
      >
        <span aria-hidden="true">＋</span>
        添加章节
      </button>
      <button
        v-if="activeSectionId"
        type="button"
        class="report-outline__delete"
        :disabled="disabled"
        @click="emit('delete')"
      >
        删除当前章节
      </button>
    </div>
  </aside>
</template>

<style scoped>
.report-outline {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 11px;
  padding: 20px 16px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-lg);
}

.report-outline__eyebrow {
  color: var(--ks-accent);
  font-size: 11px;
  font-weight: 760;
  letter-spacing: 0.1em;
}

.report-outline__label,
.report-outline__section-head {
  color: var(--ks-muted);
  font-size: 12px;
  font-weight: 680;
}

.report-outline__title-input {
  width: 100%;
  min-height: 38px;
  padding: 0 10px;
  color: var(--ks-ink);
  font-size: 14px;
  font-weight: 680;
  background: var(--ks-surface-subtle);
  border: 1px solid var(--ks-border);
  border-radius: 8px;
}

.report-outline__title-input:focus {
  border-color: var(--ks-accent);
  outline: 2px solid rgb(55 116 102 / 12%);
}

.report-outline__section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 15px;
}

.report-outline__count {
  color: var(--ks-faint);
  font-weight: 500;
}

.report-outline__sections {
  display: flex;
  max-height: 46vh;
  flex-direction: column;
  gap: 4px;
  overflow-y: auto;
}

.report-outline__section {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 9px;
  padding: 10px 8px;
  color: var(--ks-text);
  text-align: left;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 8px;
  cursor: pointer;
}

.report-outline__section:hover,
.report-outline__section.is-active {
  color: var(--ks-accent-strong);
  background: var(--ks-accent-soft);
}

.report-outline__section-number {
  display: grid;
  width: 22px;
  height: 22px;
  flex: 0 0 22px;
  place-items: center;
  color: var(--ks-muted);
  font-size: 11px;
  background: var(--ks-surface-muted);
  border-radius: 6px;
}

.report-outline__section.is-active .report-outline__section-number {
  color: var(--ks-surface);
  background: var(--ks-accent);
}

.report-outline__section-title {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.report-outline__empty {
  margin: 0;
  color: var(--ks-muted);
  font-size: 12px;
  line-height: 1.6;
}

.report-outline__actions {
  display: flex;
  flex-direction: column;
  gap: 7px;
  margin-top: auto;
  padding-top: 14px;
  border-top: 1px solid var(--ks-border);
}

.report-outline__add,
.report-outline__delete {
  min-height: 34px;
  padding: 0 9px;
  font-size: 12px;
  font-weight: 680;
  border-radius: 7px;
  cursor: pointer;
}

.report-outline__add {
  color: var(--ks-accent-strong);
  background: var(--ks-accent-soft);
  border: 1px solid transparent;
}

.report-outline__delete {
  color: var(--ks-danger);
  background: transparent;
  border: 1px solid var(--ks-border);
}

.report-outline__add:disabled,
.report-outline__delete:disabled,
.report-outline__section:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}
</style>
