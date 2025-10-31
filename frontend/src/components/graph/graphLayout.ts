export interface LayoutNode {
  id: string;
}

export interface LayoutEdge {
  source: string;
  target: string;
}

export interface LayoutPoint {
  x: number;
  y: number;
}

export interface LayoutOptions {
  width?: number;
  height?: number;
  padding?: number;
  iterations?: number;
}

const DEFAULT_WIDTH = 900;
const DEFAULT_HEIGHT = 620;
const DEFAULT_PADDING = 34;
const MIN_CELL = 32;
const MAX_CELL = 128;
const COMPONENT_GAP_RATIO = 0.5;
const AREA_FILL = 0.75;
const GROW_STEP = 1.12;
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

interface ComponentPlacement {
  nodeIds: string[];
  x: number;
  y: number;
  width: number;
  height: number;
}

interface Packing {
  placements: ComponentPlacement[];
  width: number;
  height: number;
  scale: number;
}

/** Stable pseudo-random value in [0, 1) derived from one identifier. */
export function hashUnit(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return ((hash >>> 0) % 100_000) / 100_000;
}

export function buildAdjacency(edges: readonly LayoutEdge[]): Map<string, Set<string>> {
  const adjacency = new Map<string, Set<string>>();
  for (const edge of edges) {
    if (edge.source === edge.target) {
      continue;
    }
    const source = adjacency.get(edge.source) ?? new Set<string>();
    source.add(edge.target);
    adjacency.set(edge.source, source);
    const target = adjacency.get(edge.target) ?? new Set<string>();
    target.add(edge.source);
    adjacency.set(edge.target, target);
  }
  return adjacency;
}

/**
 * Split nodes into connected components in stable input order.
 *
 * Neighbours are visited in the order the nodes were handed in, so the same
 * graph always yields the same groups.
 */
export function findComponents(
  nodes: readonly LayoutNode[],
  edges: readonly LayoutEdge[],
): string[][] {
  const ordered = nodes.map((node) => node.id);
  const known = new Set(ordered);
  const adjacency = new Map<string, string[]>(ordered.map((id) => [id, []]));
  for (const edge of edges) {
    if (edge.source === edge.target || !known.has(edge.source) || !known.has(edge.target)) {
      continue;
    }
    adjacency.get(edge.source)?.push(edge.target);
    adjacency.get(edge.target)?.push(edge.source);
  }

  const visited = new Set<string>();
  const components: string[][] = [];
  for (const id of ordered) {
    if (visited.has(id)) {
      continue;
    }
    const group: string[] = [];
    const pending = [id];
    visited.add(id);
    while (pending.length > 0) {
      const current = pending.pop();
      if (current === undefined) {
        continue;
      }
      group.push(current);
      for (const neighbour of adjacency.get(current) ?? []) {
        if (!visited.has(neighbour)) {
          visited.add(neighbour);
          pending.push(neighbour);
        }
      }
    }
    components.push(group);
  }
  return components;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function resolveIterations(nodeCount: number): number {
  if (nodeCount <= 12) {
    return 280;
  }
  if (nodeCount <= 120) {
    return 340;
  }
  if (nodeCount <= 320) {
    return 240;
  }
  return 180;
}

/**
 * Area reserved for one node, including its label and the gap to its neighbours.
 *
 * Demand is summed per component because a component of n nodes needs a box of
 * about ``cell * (sqrt(n) + 0.8)``; sizing by node count alone would hand small
 * graphs oversized boxes that no longer fit together.
 */
function resolveCell(
  components: readonly string[][],
  width: number,
  height: number,
  padding: number,
): number {
  const available = Math.max((width - padding * 2) * (height - padding * 2), 1);
  const demand = components.reduce(
    (total, nodeIds) => total + (Math.sqrt(nodeIds.length) + 0.8) ** 2,
    0,
  );
  return clamp(Math.sqrt((available * AREA_FILL) / Math.max(demand, 1)), MIN_CELL, MAX_CELL);
}

function componentSide(nodeCount: number, cell: number): number {
  return cell * (Math.sqrt(nodeCount) + 0.8);
}

/**
 * Relax one connected component inside an elliptical wall.
 *
 * The wall replaces the "unbounded" part of a classic force layout: repulsion
 * presses the community outwards until it fills its box instead of collapsing
 * into a dot or unfolding into a thin line, which is what makes a cluster
 * readable without dragging its nodes apart.
 */
function relaxComponent(
  nodeIds: readonly string[],
  internalEdges: readonly LayoutEdge[],
  boxWidth: number,
  boxHeight: number,
  iterations: number,
): Map<string, LayoutPoint> {
  const points = new Map<string, LayoutPoint>();
  const count = nodeIds.length;
  const centerX = boxWidth / 2;
  const centerY = boxHeight / 2;
  if (count === 1) {
    points.set(nodeIds[0], { x: centerX, y: centerY });
    return points;
  }

  const inset = Math.min(Math.min(boxWidth, boxHeight) * 0.1, 26);
  const wallX = Math.max(centerX - inset, 1);
  const wallY = Math.max(centerY - inset, 1);
  const spacing = (1.25 * Math.sqrt(wallX * wallY)) / Math.sqrt(count);
  nodeIds.forEach((id, position) => {
    const angle = position * GOLDEN_ANGLE + hashUnit(id) * 0.35;
    const distance = 0.7 * Math.sqrt((position + 1) / count);
    points.set(id, {
      x: centerX + Math.cos(angle) * distance * wallX,
      y: centerY + Math.sin(angle) * distance * wallY,
    });
  });

  const index = new Map(nodeIds.map((id, position) => [id, position]));
  const xs = new Float64Array(count);
  const ys = new Float64Array(count);
  nodeIds.forEach((id, position) => {
    const point = points.get(id) ?? { x: centerX, y: centerY };
    xs[position] = point.x;
    ys[position] = point.y;
  });

  const forcesX = new Float64Array(count);
  const forcesY = new Float64Array(count);
  let temperature = Math.min(wallX, wallY) * 0.5;
  const cooling = temperature / Math.max(iterations, 1);

  for (let step = 0; step < iterations; step += 1) {
    forcesX.fill(0);
    forcesY.fill(0);
    for (let i = 0; i < count; i += 1) {
      for (let j = i + 1; j < count; j += 1) {
        let diffX = xs[i] - xs[j];
        let diffY = ys[i] - ys[j];
        let distance = Math.hypot(diffX, diffY);
        if (distance < 0.01) {
          diffX = (hashUnit(`${nodeIds[i]}:${nodeIds[j]}`) - 0.5) * 0.2;
          diffY = (hashUnit(`${nodeIds[j]}:${nodeIds[i]}`) - 0.5) * 0.2;
          distance = Math.hypot(diffX, diffY) || 0.01;
        }
        const repulsion = (spacing * spacing) / distance;
        const forceX = (diffX / distance) * repulsion;
        const forceY = (diffY / distance) * repulsion;
        forcesX[i] += forceX;
        forcesY[i] += forceY;
        forcesX[j] -= forceX;
        forcesY[j] -= forceY;
      }
    }
    for (const edge of internalEdges) {
      const source = index.get(edge.source);
      const target = index.get(edge.target);
      if (source === undefined || target === undefined || source === target) {
        continue;
      }
      const diffX = xs[source] - xs[target];
      const diffY = ys[source] - ys[target];
      const distance = Math.hypot(diffX, diffY) || 0.01;
      const attraction = (distance * distance) / spacing;
      const forceX = (diffX / distance) * attraction;
      const forceY = (diffY / distance) * attraction;
      forcesX[source] -= forceX;
      forcesY[source] -= forceY;
      forcesX[target] += forceX;
      forcesY[target] += forceY;
    }
    for (let i = 0; i < count; i += 1) {
      forcesX[i] += (centerX - xs[i]) * 0.05;
      forcesY[i] += (centerY - ys[i]) * 0.05;
      const force = Math.hypot(forcesX[i], forcesY[i]);
      if (force < 1e-6) {
        continue;
      }
      const displacement = Math.min(force, temperature);
      xs[i] += (forcesX[i] / force) * displacement;
      ys[i] += (forcesY[i] / force) * displacement;
      const offsetX = (xs[i] - centerX) / wallX;
      const offsetY = (ys[i] - centerY) / wallY;
      const radius = Math.hypot(offsetX, offsetY);
      if (radius > 1) {
        xs[i] = centerX + (offsetX / radius) * wallX;
        ys[i] = centerY + (offsetY / radius) * wallY;
      }
    }
    temperature = Math.max(temperature - cooling, 0);
  }

  nodeIds.forEach((id, position) => {
    points.set(id, { x: xs[position], y: ys[position] });
  });
  return points;
}

/**
 * Ensure one relaxed component fills the box it was given.
 *
 * The circular wall already keeps the shape compact, so scaling it up is
 * predictable: the community uses the space instead of leaving a margin that
 * separates it from its neighbours more than from its own nodes.
 */
function fillBox(
  points: Map<string, LayoutPoint>,
  nodeIds: readonly string[],
  boxWidth: number,
  boxHeight: number,
): void {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const id of nodeIds) {
    const point = points.get(id);
    if (point === undefined) {
      continue;
    }
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  }
  if (!Number.isFinite(minX)) {
    return;
  }
  const spanX = Math.max(maxX - minX, 0.001);
  const spanY = Math.max(maxY - minY, 0.001);
  const inset = Math.min(Math.min(boxWidth, boxHeight) * 0.08, 20);
  const scale = Math.min(
    Math.max(boxWidth - inset * 2, 1) / spanX,
    Math.max(boxHeight - inset * 2, 1) / spanY,
  );
  const offsetX = (boxWidth - spanX * scale) / 2;
  const offsetY = (boxHeight - spanY * scale) / 2;
  for (const id of nodeIds) {
    const point = points.get(id);
    if (point === undefined) {
      continue;
    }
    points.set(id, {
      x: offsetX + (point.x - minX) * scale,
      y: offsetY + (point.y - minY) * scale,
    });
  }
}

/** Shelf-pack the component boxes, wrapping at the usable width. */
function packBoxes(
  components: readonly string[][],
  cell: number,
  usableWidth: number,
): Packing {
  const ordered = [...components].sort((left, right) => {
    if (left.length !== right.length) {
      return right.length - left.length;
    }
    return (left[0] ?? "") < (right[0] ?? "") ? -1 : 1;
  });
  const gap = cell * COMPONENT_GAP_RATIO;
  const placements: ComponentPlacement[] = [];
  let cursorX = 0;
  let cursorY = 0;
  let rowHeight = 0;
  let contentWidth = 0;

  for (const nodeIds of ordered) {
    const side = componentSide(nodeIds.length, cell);
    if (cursorX > 0 && cursorX + side > usableWidth) {
      cursorX = 0;
      cursorY += rowHeight + gap;
      rowHeight = 0;
    }
    placements.push({ nodeIds, x: cursorX, y: cursorY, width: side, height: side });
    cursorX += side + gap;
    rowHeight = Math.max(rowHeight, side);
    contentWidth = Math.max(contentWidth, cursorX - gap);
  }
  return {
    placements,
    width: Math.max(contentWidth, 1),
    height: Math.max(cursorY + rowHeight, 1),
    scale: 1,
  };
}

/**
 * A single community gets the whole canvas instead of a square box, so a
 * document-scoped graph fills the viewport rather than sitting in the middle.
 */
function fillCanvas(
  components: readonly string[][],
  usableWidth: number,
  usableHeight: number,
): Packing | null {
  if (components.length !== 1) {
    return null;
  }
  return {
    placements: [
      {
        nodeIds: [...components[0]],
        x: 0,
        y: 0,
        width: usableWidth,
        height: usableHeight,
      },
    ],
    width: usableWidth,
    height: usableHeight,
    scale: 1,
  };
}

function fits(packing: Packing, usableWidth: number, usableHeight: number): boolean {
  return packing.width <= usableWidth && packing.height <= usableHeight;
}

/** Centre the packed boxes inside the canvas. */
function centrePacking(
  packing: Packing,
  width: number,
  height: number,
): void {
  const offsetX = (width - packing.width) / 2;
  const offsetY = (height - packing.height) / 2;
  for (const placement of packing.placements) {
    placement.x += offsetX;
    placement.y += offsetY;
  }
}

/**
 * Deterministic cluster-aware force layout.
 *
 * Connected components are relaxed inside their own circular wall and packed
 * as boxes across the canvas.  Communities therefore fill the space they are
 * given, so both their internal structure and the links between them stay
 * readable without dragging nodes apart by hand.
 */
export function computeForceLayout(
  nodes: readonly LayoutNode[],
  edges: readonly LayoutEdge[],
  options: LayoutOptions = {},
): Map<string, LayoutPoint> {
  const positions = new Map<string, LayoutPoint>();
  const nodeCount = nodes.length;
  if (nodeCount === 0) {
    return positions;
  }
  const width = options.width ?? DEFAULT_WIDTH;
  const height = options.height ?? DEFAULT_HEIGHT;
  const padding = options.padding ?? DEFAULT_PADDING;
  if (nodeCount === 1) {
    positions.set(nodes[0].id, { x: width / 2, y: height / 2 });
    return positions;
  }

  const components = findComponents(nodes, edges);
  const usableWidth = Math.max(width - padding * 2, 1);
  const usableHeight = Math.max(height - padding * 2, 1);
  let packing = fillCanvas(components, usableWidth, usableHeight);
  if (packing === null) {
    let cell = resolveCell(components, width, height, padding);
    let packed = packBoxes(components, cell, usableWidth);
    for (let attempt = 0; attempt < 8 && !fits(packed, usableWidth, usableHeight); attempt += 1) {
      cell *= 0.85;
      packed = packBoxes(components, cell, usableWidth);
    }
    for (let attempt = 0; attempt < 6; attempt += 1) {
      const grown = packBoxes(components, cell * GROW_STEP, usableWidth);
      if (!fits(grown, usableWidth, usableHeight)) {
        break;
      }
      cell *= GROW_STEP;
      packed = grown;
    }
    packing = packed;
  }
  centrePacking(packing, width, height);

  const known = new Set(nodes.map((node) => node.id));
  for (const placement of packing.placements) {
    const members = new Set(placement.nodeIds);
    const internalEdges = edges.filter(
      (edge) => members.has(edge.source) && members.has(edge.target) && known.has(edge.source),
    );
    const local = relaxComponent(
      placement.nodeIds,
      internalEdges,
      placement.width,
      placement.height,
      options.iterations ?? resolveIterations(placement.nodeIds.length),
    );
    fillBox(local, placement.nodeIds, placement.width, placement.height);
    for (const id of placement.nodeIds) {
      const point = local.get(id);
      if (point === undefined) {
        continue;
      }
      positions.set(id, { x: placement.x + point.x, y: placement.y + point.y });
    }
  }
  return positions;
}
