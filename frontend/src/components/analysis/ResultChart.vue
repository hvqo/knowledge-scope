<script setup lang="ts">
import { computed } from "vue";

import type { AnalysisChartSpec } from "./chartInference";

const props = defineProps<{
  spec: AnalysisChartSpec;
}>();

const maxValue = computed(() => Math.max(...props.spec.points.map((point) => point.value), 0));
const minValue = computed(() => Math.min(...props.spec.points.map((point) => point.value), 0));
const range = computed(() => Math.max(maxValue.value - minValue.value, 1));
const linePoints = computed(() =>
  props.spec.points
    .map((point, index) => {
      const x = 24 + (index * 252) / Math.max(props.spec.points.length - 1, 1);
      const y = 172 - ((point.value - minValue.value) / range.value) * 144;
      return `${x},${y}`;
    })
    .join(" "),
);

const pieGradient = computed(() => {
  const total = props.spec.points.reduce((sum, point) => sum + point.value, 0);
  if (total <= 0) {
    return "conic-gradient(#dfe7e1 0 100%)";
  }
  let cursor = 0;
  const colors = ["#377466", "#6b9d91", "#a5c5bc", "#d0e1db", "#8baf82", "#c29a62"];
  const stops = props.spec.points.map((point, index) => {
    const start = cursor;
    cursor += (point.value / total) * 100;
    return `${colors[index % colors.length]} ${start}% ${cursor}%`;
  });
  return `conic-gradient(${stops.join(", ")})`;
});
</script>

<template>
  <div class="result-chart">
    <div class="result-chart__heading">
      <span>{{ spec.kind === "line" ? "趋势" : spec.kind === "pie" ? "分布" : "对比" }}</span>
      <small>{{ spec.categoryColumn }} × {{ spec.valueColumn }}</small>
    </div>
    <div
      v-if="spec.kind === 'pie'"
      class="result-chart__pie-layout"
    >
      <div
        class="result-chart__pie"
        :style="{ background: pieGradient }"
        aria-hidden="true"
      />
      <ul class="result-chart__legend">
        <li
          v-for="(point, index) in spec.points"
          :key="point.label"
        >
          <span
            class="result-chart__legend-dot"
            :style="{ background: ['#377466', '#6b9d91', '#a5c5bc', '#d0e1db', '#8baf82', '#c29a62'][index % 6] }"
            aria-hidden="true"
          />
          <span>{{ point.label }}</span>
          <strong>{{ point.value }}</strong>
        </li>
      </ul>
    </div>
    <div
      v-else
      class="result-chart__plot"
    >
      <svg
        viewBox="0 0 300 200"
        role="img"
        aria-label="分析结果图表"
      >
        <line
          x1="24"
          y1="172"
          x2="276"
          y2="172"
          stroke="#d4d9d2"
          stroke-width="1"
        />
        <polyline
          v-if="spec.kind === 'line'"
          :points="linePoints"
          fill="none"
          stroke="#377466"
          stroke-width="3"
          stroke-linecap="round"
          stroke-linejoin="round"
        />
        <template v-else>
          <rect
            v-for="(point, index) in spec.points"
            :key="point.label"
            :x="28 + (index * 244) / spec.points.length + 4"
            :y="172 - ((point.value - minValue) / range) * 144"
            :width="Math.max(10, 180 / spec.points.length)"
            :height="Math.max(0, ((point.value - minValue) / range) * 144)"
            rx="3"
            fill="#6b9d91"
          />
        </template>
      </svg>
      <div class="result-chart__labels">
        <span
          v-for="point in spec.points"
          :key="point.label"
        >{{ point.label }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.result-chart {
  padding: 16px;
  background: var(--ks-surface-subtle);
  border: 1px solid var(--ks-border);
  border-radius: 10px;
}

.result-chart__heading {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  color: var(--ks-ink);
  font-size: 13px;
  font-weight: 700;
}

.result-chart__heading small {
  color: var(--ks-muted);
  font-size: 11px;
  font-weight: 500;
}

.result-chart__plot {
  margin-top: 10px;
}

.result-chart__plot svg {
  display: block;
  width: 100%;
  height: 220px;
}

.result-chart__labels {
  display: flex;
  justify-content: space-around;
  gap: 8px;
  overflow: hidden;
  color: var(--ks-muted);
  font-size: 10px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.result-chart__pie-layout {
  display: flex;
  align-items: center;
  gap: 22px;
  margin-top: 16px;
}

.result-chart__pie {
  width: 168px;
  height: 168px;
  flex: 0 0 168px;
  border-radius: 50%;
}

.result-chart__legend {
  display: flex;
  min-width: 0;
  flex: 1;
  flex-direction: column;
  gap: 8px;
  padding: 0;
  margin: 0;
  color: var(--ks-text);
  font-size: 12px;
  list-style: none;
}

.result-chart__legend li {
  display: grid;
  grid-template-columns: 8px minmax(0, 1fr) auto;
  align-items: center;
  gap: 7px;
}

.result-chart__legend-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.result-chart__legend strong {
  color: var(--ks-ink);
  font-weight: 700;
}

@media (max-width: 560px) {
  .result-chart__pie-layout {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
