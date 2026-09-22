<script setup lang="ts">
defineProps<{
  sql: string | null;
}>();
</script>

<template>
  <details
    v-if="sql"
    class="sql-disclosure"
  >
    <summary>查看已执行的安全 SQL</summary>
    <pre>{{ sql }}</pre>
    <p>SQL 仅供核对，不能在此页面编辑或执行。</p>
  </details>
</template>

<style scoped>
.sql-disclosure {
  border-top: 1px solid var(--ks-border);
}

.sql-disclosure summary {
  padding: 13px 0;
  color: var(--ks-muted);
  font-size: 12px;
  font-weight: 680;
  cursor: pointer;
}

.sql-disclosure pre {
  max-height: 220px;
  padding: 12px;
  margin: 0;
  overflow: auto;
  color: var(--ks-text);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  line-height: 1.6;
  white-space: pre-wrap;
  background: var(--ks-surface-subtle);
  border: 1px solid var(--ks-border);
  border-radius: 8px;
}

.sql-disclosure p {
  margin: 7px 0 12px;
  color: var(--ks-faint);
  font-size: 11px;
}
</style>
