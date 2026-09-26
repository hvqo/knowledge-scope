<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";

import type { GraphEdge, GraphNode } from "../../api/types";
import { colorForType } from "./graphPalette";
import { buildAdjacency, computeForceLayout } from "./graphLayout";
import type { LayoutPoint } from "./graphLayout";

const VIEW_WIDTH = 900;
const VIEW_HEIGHT = 620;
const ZOOM_MIN = 0.4;
const ZOOM_MAX = 3;
const ALWAYS_LABEL_NODE_LIMIT = 26;
const HUB_LABEL_DEGREE = 4;
const LABEL_LENGTH = 10;
const FOCUS_MAX_SCALE = 2.2;
/** View-box space covered by the floating detail panel, used when framing. */
const PANEL_INSET = 150;

const props = defineProps<{
  nodes: GraphNode[];
  edges: GraphEdge[];
  selectedId: string | null;
}>();

const emit = defineEmits<{
  select: [entityId: string | null];
}>();

type PointerSession =
  | { mode: "pan"; startClient: LayoutPoint; startTransform: LayoutPoint }
  | { mode: "node"; entityId: string };

const svgRef = ref<SVGSVGElement | null>(null);
const transform = ref({ x: 0, y: 0, scale: 1 });
const hoveredId = ref<string | null>(null);
const draggedId = ref<string | null>(null);
const overrides = ref<Record<string, LayoutPoint>>({});
const showAllLabels = ref(false);
const hideIsolated = ref(false);
let pointerSession: PointerSession | null = null;

/** Unconnected entities are common after a search; hiding them is opt-in. */
const visibleNodes = computed(() =>
  hideIsolated.value ? props.nodes.filter((node) => node.degree > 0) : props.nodes,
);

const visibleIds = computed(() => new Set(visibleNodes.value.map((node) => node.entity_id)));

const visibleEdges = computed(() =>
  props.edges.filter(
    (edge) => visibleIds.value.has(edge.source_entity_id) && visibleIds.value.has(edge.target_entity_id),
  ),
);

const graphEdges = computed(() =>
  visibleEdges.value.map((edge) => ({
    source: edge.source_entity_id,
    target: edge.target_entity_id,
  })),
);

const adjacency = computed(() => buildAdjacency(graphEdges.value));

const layout = computed(() =>
  computeForceLayout(
    visibleNodes.value.map((node) => ({ id: node.entity_id })),
    graphEdges.value,
  ),
);

const radii = computed<Record<string, number>>(() =>
  Object.fromEntries(
    visibleNodes.value.map((node) => [node.entity_id, 9 + Math.min(node.degree, 12) * 1.1]),
  ),
);

const focusId = computed(() => hoveredId.value ?? props.selectedId);

const highlighted = computed<Set<string> | null>(() => {
  const focus = focusId.value;
  if (focus === null) {
    return null;
  }
  return new Set<string>([focus, ...(adjacency.value.get(focus) ?? [])]);
});

const ariaLabel = computed(
  () =>
    `知识图谱画布，共 ${visibleNodes.value.length} 个实体、${visibleEdges.value.length} 条关系，选中节点后可在右侧查看证据与相邻实体。`,
);

const nodeCount = computed(() => visibleNodes.value.length);

const isolatedCount = computed(
  () => props.nodes.filter((node) => node.degree === 0).length,
);

const renderedNodes = computed(() => {
  const focus = focusId.value;
  const highlight = highlighted.value;
  const focusPoint = focus === null ? null : (overrides.value[focus] ?? layout.value.get(focus) ?? null);
  const graphCenter = { x: VIEW_WIDTH / 2, y: VIEW_HEIGHT / 2 };
  return visibleNodes.value.map((node) => {
    const point = overrides.value[node.entity_id] ?? layout.value.get(node.entity_id) ?? graphCenter;
    const isFocused = focus === node.entity_id;
    const radius = radii.value[node.entity_id] ?? 9;
    // Labels radiate away from the focused node (or the graph centre) so they
    // do not stack on top of each other inside a dense neighbourhood.
    const origin = focusPoint !== null && !isFocused ? focusPoint : graphCenter;
    const offsetX = point.x - origin.x;
    const offsetY = point.y - origin.y;
    const length = Math.hypot(offsetX, offsetY) || 1;
    const ratioX = offsetX / length;
    const ratioY = offsetY / length;
    const distance = radius + 12;
    return {
      node,
      point,
      radius,
      label:
        node.canonical_name.length > LABEL_LENGTH
          ? `${node.canonical_name.slice(0, LABEL_LENGTH)}…`
          : node.canonical_name,
      labelX: ratioX * distance,
      labelY: ratioY * distance + 3,
      labelAnchor: ratioX > 0.35 ? "start" : ratioX < -0.35 ? "end" : "middle",
      showLabel: shouldShowLabel(node, isFocused, highlight),
    };
  });
});

/** Relation labels are useful, but a hub with many edges would just stack them. */
const showEdgeLabels = computed(
  () => visibleEdges.value.filter((edge) => isFocusEdge(edge)).length <= 4,
);

const renderedEdges = computed(() =>
  visibleEdges.value.flatMap((edge) => {
    const source = overrides.value[edge.source_entity_id] ?? layout.value.get(edge.source_entity_id);
    const target = overrides.value[edge.target_entity_id] ?? layout.value.get(edge.target_entity_id);
    if (!source || !target) {
      return [];
    }
    const diffX = target.x - source.x;
    const diffY = target.y - source.y;
    const distance = Math.hypot(diffX, diffY) || 1;
    const unitX = diffX / distance;
    const unitY = diffY / distance;
    const sourceGap = (radii.value[edge.source_entity_id] ?? 9) + 1;
    const targetGap = (radii.value[edge.target_entity_id] ?? 9) + 5;
    const start = { x: source.x + unitX * sourceGap, y: source.y + unitY * sourceGap };
    const end = { x: target.x - unitX * targetGap, y: target.y - unitY * targetGap };
    const label = { x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 - 4 };
    return [{ edge, start, end, label }];
  }),
);

const canFocusSelection = computed(
  () => props.selectedId !== null && layout.value.has(props.selectedId),
);

watch(
  () =>
    `${props.nodes.map((node) => node.entity_id).join("|")}#${props.edges.length}#${hideIsolated.value}`,
  () => {
    overrides.value = {};
    draggedId.value = null;
    transform.value = { x: 0, y: 0, scale: 1 };
  },
);

/**
 * Labels are the first thing that turns a graph into noise.  Only the focused
 * entity, its direct neighbours, and obvious hubs stay labelled once the graph
 * grows past a handful of nodes.
 */
function shouldShowLabel(
  node: GraphNode,
  isFocused: boolean,
  highlight: Set<string> | null,
): boolean {
  if (isFocused || highlight?.has(node.entity_id)) {
    return true;
  }
  if (showAllLabels.value || nodeCount.value <= ALWAYS_LABEL_NODE_LIMIT) {
    return true;
  }
  return node.degree >= HUB_LABEL_DEGREE && nodeCount.value <= 120;
}

function pointOf(entityId: string): LayoutPoint | null {
  return overrides.value[entityId] ?? layout.value.get(entityId) ?? null;
}

function fitToPoints(points: readonly LayoutPoint[], margin: number, maxScale = ZOOM_MAX): void {
  if (points.length === 0) {
    return;
  }
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const point of points) {
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  }
  const spanX = Math.max(maxX - minX, 1);
  const spanY = Math.max(maxY - minY, 1);
  const scale = Math.min(
    maxScale,
    Math.max(
      ZOOM_MIN,
      Math.min((VIEW_WIDTH - margin * 2) / spanX, (VIEW_HEIGHT - margin * 2) / spanY),
    ),
  );
  // The detail panel floats over the right edge, so frame into what is visible.
  const centerX = VIEW_WIDTH / 2 - (props.selectedId === null ? 0 : PANEL_INSET);
  transform.value = {
    scale,
    x: centerX - ((minX + maxX) / 2) * scale,
    y: VIEW_HEIGHT / 2 - ((minY + maxY) / 2) * scale,
  };
}

/** Frame the selected entity together with its neighbours. */
function focusSelected(): void {
  const entityId = props.selectedId;
  if (entityId === null) {
    return;
  }
  const ids = [entityId, ...(adjacency.value.get(entityId) ?? [])];
  const points = ids.flatMap((id) => {
    const point = pointOf(id);
    return point === null ? [] : [point];
  });
  fitToPoints(points, 120, FOCUS_MAX_SCALE);
}

function fitToView(): void {
  fitToPoints([...layout.value.values()], 34);
}

function selectNode(entityId: string | null): void {
  emit("select", entityId);
}

function focusNode(entityId: string): void {
  emit("select", entityId);
  const point = pointOf(entityId);
  if (point !== null) {
    const ids = [entityId, ...(adjacency.value.get(entityId) ?? [])];
    const points = ids.flatMap((id) => {
      const candidate = pointOf(id);
      return candidate === null ? [] : [candidate];
    });
    fitToPoints(points, 80, FOCUS_MAX_SCALE);
  }
}

function isFocusEdge(edge: GraphEdge): boolean {
  const focus = focusId.value;
  if (focus === null) {
    return false;
  }
  return edge.source_entity_id === focus || edge.target_entity_id === focus;
}

function nodeClasses(entityId: string): string[] {
  const classes = ["graph-node"];
  if (props.selectedId === entityId) {
    classes.push("is-selected");
  }
  if (focusId.value === entityId) {
    classes.push("is-focus");
  }
  if (highlighted.value && !highlighted.value.has(entityId)) {
    classes.push("is-dimmed");
  }
  return classes;
}

function edgeClasses(edge: GraphEdge): string[] {
  const classes = ["graph-edge"];
  if (edge.kind === "canonical_bridge") {
    classes.push("is-merge");
  }
  if (isFocusEdge(edge)) {
    classes.push("is-focus");
  } else if (highlighted.value) {
    classes.push("is-dimmed");
  }
  return classes;
}

function toViewPoint(event: { clientX: number; clientY: number }): LayoutPoint {
  const rect = svgRef.value?.getBoundingClientRect();
  if (!rect || rect.width === 0 || rect.height === 0) {
    return { x: VIEW_WIDTH / 2, y: VIEW_HEIGHT / 2 };
  }
  return {
    x: ((event.clientX - rect.left) / rect.width) * VIEW_WIDTH,
    y: ((event.clientY - rect.top) / rect.height) * VIEW_HEIGHT,
  };
}

function toGraphPoint(event: PointerEvent): LayoutPoint {
  const view = toViewPoint(event);
  return {
    x: (view.x - transform.value.x) / transform.value.scale,
    y: (view.y - transform.value.y) / transform.value.scale,
  };
}

function zoomBy(factor: number, anchor?: LayoutPoint): void {
  const scale = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, transform.value.scale * factor));
  const ratio = scale / transform.value.scale;
  const anchorX = anchor?.x ?? VIEW_WIDTH / 2;
  const anchorY = anchor?.y ?? VIEW_HEIGHT / 2;
  transform.value = {
    scale,
    x: anchorX - (anchorX - transform.value.x) * ratio,
    y: anchorY - (anchorY - transform.value.y) * ratio,
  };
}

function onWheel(event: WheelEvent): void {
  zoomBy(event.deltaY < 0 ? 1.12 : 1 / 1.12, toViewPoint(event));
}

function resetView(): void {
  transform.value = { x: 0, y: 0, scale: 1 };
  overrides.value = {};
}

function endPointerSession(): void {
  window.removeEventListener("pointermove", handlePointerMove);
  window.removeEventListener("pointerup", handlePointerUp);
  pointerSession = null;
  draggedId.value = null;
}

function handlePointerMove(event: PointerEvent): void {
  if (pointerSession?.mode === "node") {
    overrides.value = {
      ...overrides.value,
      [pointerSession.entityId]: toGraphPoint(event),
    };
    return;
  }
  if (pointerSession?.mode === "pan") {
    const rect = svgRef.value?.getBoundingClientRect();
    const ratioX = rect && rect.width > 0 ? VIEW_WIDTH / rect.width : 1;
    const ratioY = rect && rect.height > 0 ? VIEW_HEIGHT / rect.height : 1;
    transform.value = {
      ...transform.value,
      x: pointerSession.startTransform.x + (event.clientX - pointerSession.startClient.x) * ratioX,
      y: pointerSession.startTransform.y + (event.clientY - pointerSession.startClient.y) * ratioY,
    };
  }
}

function handlePointerUp(): void {
  endPointerSession();
}

function startPan(event: PointerEvent): void {
  if (event.button !== 0) {
    return;
  }
  pointerSession = {
    mode: "pan",
    startClient: { x: event.clientX, y: event.clientY },
    startTransform: { x: transform.value.x, y: transform.value.y },
  };
  window.addEventListener("pointermove", handlePointerMove);
  window.addEventListener("pointerup", handlePointerUp, { once: true });
}

function startNodeDrag(entityId: string, event: PointerEvent): void {
  if (event.button !== 0) {
    return;
  }
  const session = pointerSession;
  if (session?.mode === "pan") {
    endPointerSession();
  }
  pointerSession = { mode: "node", entityId };
  draggedId.value = entityId;
  window.addEventListener("pointermove", handlePointerMove);
  window.addEventListener("pointerup", handlePointerUp, { once: true });
}

function clearSelection(event: PointerEvent): void {
  emit("select", null);
  startPan(event);
}

function pointValue(point: LayoutPoint): string {
  return `${point.x.toFixed(1)} ${point.y.toFixed(1)}`;
}

defineExpose({ fitToView, focusSelected });

onBeforeUnmount(() => {
  endPointerSession();
});
</script>

<template>
  <div class="graph-canvas-shell">
    <div class="canvas-toolbar">
      <p class="canvas-summary">
        {{ nodeCount }} 个实体 · {{ visibleEdges.length }} 条关系
      </p>
      <div class="canvas-actions">
        <label
          v-if="isolatedCount > 0"
          class="canvas-toggle"
          title="隐藏没有可见关系的实体"
        >
          <input
            v-model="hideIsolated"
            type="checkbox"
          >
          隐藏未连接（{{ isolatedCount }}）
        </label>
        <label
          class="canvas-toggle"
          title="显示所有实体名称"
        >
          <input
            v-model="showAllLabels"
            type="checkbox"
          >
          全部标签
        </label>
        <button
          type="button"
          class="canvas-button is-wide"
          :disabled="!canFocusSelection"
          @click="focusSelected"
        >
          聚焦选中
        </button>
        <button
          type="button"
          class="canvas-button is-wide"
          @click="fitToView"
        >
          适应画布
        </button>
        <button
          type="button"
          class="canvas-button"
          title="放大"
          @click="zoomBy(1.2)"
        >
          +
        </button>
        <button
          type="button"
          class="canvas-button"
          title="缩小"
          @click="zoomBy(1 / 1.2)"
        >
          −
        </button>
        <button
          type="button"
          class="canvas-button is-wide"
          title="清除手动拖动并回到初始视图"
          @click="resetView"
        >
          重置
        </button>
      </div>
    </div>

    <svg
      ref="svgRef"
      class="graph-canvas"
      :viewBox="`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`"
      role="img"
      :aria-label="ariaLabel"
      @wheel.prevent="onWheel"
      @pointerdown.self="clearSelection"
    >
      <defs>
        <marker
          id="graph-arrow"
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="7"
          markerHeight="7"
          orient="auto-start-reverse"
        >
          <path
            d="M 0 0 L 10 5 L 0 10 z"
            fill="var(--ks-border-strong)"
          />
        </marker>
        <marker
          id="graph-arrow-focus"
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="7"
          markerHeight="7"
          orient="auto-start-reverse"
        >
          <path
            d="M 0 0 L 10 5 L 0 10 z"
            fill="var(--ks-accent-strong)"
          />
        </marker>
      </defs>

      <g :transform="`translate(${transform.x} ${transform.y}) scale(${transform.scale})`">
        <g class="edge-layer">
          <g
            v-for="item in renderedEdges"
            :key="item.edge.relation_id"
            :class="edgeClasses(item.edge)"
          >
            <line
              :x1="item.start.x"
              :y1="item.start.y"
              :x2="item.end.x"
              :y2="item.end.y"
              :marker-end="
                item.edge.kind === 'canonical_bridge'
                  ? ''
                  : isFocusEdge(item.edge)
                    ? 'url(#graph-arrow-focus)'
                    : 'url(#graph-arrow)'
              "
            />
            <text
              v-if="showEdgeLabels && isFocusEdge(item.edge)"
              class="edge-label"
              :x="item.label.x"
              :y="item.label.y"
              text-anchor="middle"
            >
              {{ item.edge.relation_type }}
            </text>
          </g>
        </g>

        <g class="node-layer">
          <g
            v-for="item in renderedNodes"
            :key="item.node.entity_id"
            :class="nodeClasses(item.node.entity_id)"
            :transform="`translate(${pointValue(item.point)})`"
            @pointerdown.stop="startNodeDrag(item.node.entity_id, $event)"
            @pointerenter="hoveredId = item.node.entity_id"
            @pointerleave="hoveredId = null"
            @click.stop="selectNode(item.node.entity_id)"
            @dblclick.stop="focusNode(item.node.entity_id)"
          >
            <circle
              :r="item.radius"
              :fill="colorForType(item.node.entity_type)"
            />
            <text
              v-if="item.showLabel"
              class="node-label"
              :x="item.labelX"
              :y="item.labelY"
              :text-anchor="item.labelAnchor"
            >
              {{ item.label }}
            </text>
            <title>{{ item.node.canonical_name }}（{{ item.node.entity_type }}）</title>
          </g>
        </g>
      </g>
    </svg>

    <p class="canvas-hint">
      点击节点查看来源证据，双击节点或使用“聚焦选中”放大它和相邻实体；拖动可微调位置，滚轮缩放。
    </p>
  </div>
</template>

<style scoped>
.graph-canvas-shell {
  display: flex;
  flex-direction: column;
  gap: var(--ks-space-2);
}

.canvas-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: var(--ks-space-2) var(--ks-space-3);
}

.canvas-summary {
  margin: 0;
  color: var(--ks-muted);
  font-size: 13px;
}

.canvas-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--ks-space-1);
}

.canvas-button {
  display: grid;
  height: 30px;
  min-width: 30px;
  padding: 0;
  place-items: center;
  color: var(--ks-text);
  font-size: 15px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-sm);
  cursor: pointer;
  transition: color var(--ks-duration-fast) var(--ks-ease-out),
    border-color var(--ks-duration-fast) var(--ks-ease-out);
}

.canvas-button:hover:not(:disabled) {
  color: var(--ks-accent-strong);
  border-color: var(--ks-border-strong);
}

.canvas-button:disabled {
  color: var(--ks-faint);
  cursor: not-allowed;
}

.canvas-button.is-wide {
  width: auto;
  padding: 0 10px;
  font-size: 12px;
  font-weight: 620;
}

.canvas-select-field,
.canvas-toggle {
  display: inline-flex;
  height: 30px;
  align-items: center;
  gap: 5px;
  padding: 0 8px;
  color: var(--ks-muted);
  font-size: 12px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-sm);
}

.canvas-select {
  padding: 2px 0;
  color: var(--ks-text);
  font-size: 12px;
  background: transparent;
  border: 0;
  cursor: pointer;
}

.canvas-toggle {
  cursor: pointer;
}

.canvas-toggle input {
  margin: 0;
  accent-color: var(--ks-accent);
  cursor: pointer;
}

.graph-canvas {
  width: 100%;
  height: auto;
  aspect-ratio: 900 / 620;
  background: var(--ks-surface-subtle);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-md);
  cursor: grab;
  touch-action: none;
}

.graph-canvas:active {
  cursor: grabbing;
}

.graph-edge line {
  stroke: #b9c1b9;
  stroke-width: 1.3;
  vector-effect: non-scaling-stroke;
}

.graph-edge.is-merge line {
  stroke: var(--ks-accent);
  stroke-dasharray: 5 4;
}

.graph-edge.is-focus line {
  stroke: var(--ks-accent-strong);
  stroke-width: 2;
}

.graph-edge.is-dimmed line {
  opacity: 0.22;
}

.edge-label {
  fill: var(--ks-accent-strong);
  font-size: 11px;
  font-weight: 620;
  paint-order: stroke;
  stroke: var(--ks-surface-subtle);
  stroke-width: 3px;
  user-select: none;
}

.graph-node {
  cursor: pointer;
}

.graph-node circle {
  stroke: var(--ks-surface);
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
  transition: opacity var(--ks-duration-fast) var(--ks-ease-out);
}

.graph-node.is-focus circle {
  stroke: var(--ks-accent-strong);
  stroke-width: 3;
}

.graph-node.is-selected circle {
  stroke: var(--ks-ink);
  stroke-width: 3;
}

.graph-node.is-dimmed circle,
.graph-node.is-dimmed .node-label {
  opacity: 0.32;
}

.node-label {
  fill: var(--ks-text);
  font-size: 11px;
  font-weight: 600;
  paint-order: stroke;
  stroke: var(--ks-surface-subtle);
  stroke-width: 3px;
  user-select: none;
}

.canvas-hint {
  margin: 0;
  color: var(--ks-faint);
  font-size: 12px;
}

@media (max-width: 720px) {
  .canvas-toolbar {
    flex-direction: column;
    align-items: flex-start;
    gap: var(--ks-space-2);
  }
}
</style>
