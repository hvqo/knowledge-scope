<script setup lang="ts">
import type { ChatBIColumnMetadata, ChatBIResultScalar } from "../../api/types";

defineProps<{
  columns: ChatBIColumnMetadata[];
  rows: ChatBIResultScalar[][];
}>();

function formatValue(value: ChatBIResultScalar): string {
  if (value === null) {
    return "NULL";
  }
  if (typeof value === "number") {
    return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 6 }).format(value);
  }
  if (typeof value === "boolean") {
    return value ? "是" : "否";
  }
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value);
}
</script>

<template>
  <div class="result-table-wrap">
    <table
      v-if="rows.length > 0 && columns.length > 0"
      class="result-table"
    >
      <thead>
        <tr>
          <th
            v-for="column in columns"
            :key="column.ordinal"
          >
            {{ column.name }}
            <span>{{ column.data_type }}</span>
          </th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="(row, rowIndex) in rows"
          :key="rowIndex"
        >
          <td
            v-for="(value, columnIndex) in row"
            :key="columns[columnIndex]?.ordinal ?? columnIndex"
          >
            {{ formatValue(value) }}
          </td>
        </tr>
      </tbody>
    </table>
    <div
      v-else
      class="result-table__empty"
    >
      查询成功，但没有返回匹配数据。
    </div>
  </div>
</template>

<style scoped>
.result-table-wrap {
  overflow: auto;
  border: 1px solid var(--ks-border);
  border-radius: 10px;
}

.result-table {
  width: 100%;
  min-width: 580px;
  border-collapse: collapse;
  font-size: 13px;
}

.result-table th,
.result-table td {
  padding: 11px 13px;
  text-align: left;
  vertical-align: top;
  border-bottom: 1px solid var(--ks-border);
}

.result-table th {
  color: var(--ks-muted);
  font-size: 11px;
  font-weight: 720;
  background: var(--ks-surface-subtle);
}

.result-table th span {
  display: block;
  margin-top: 3px;
  color: var(--ks-faint);
  font-size: 10px;
  font-weight: 500;
}

.result-table td {
  max-width: 360px;
  color: var(--ks-text);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.result-table tbody tr:last-child td {
  border-bottom: 0;
}

.result-table__empty {
  padding: 34px 16px;
  color: var(--ks-muted);
  font-size: 13px;
  text-align: center;
}
</style>
