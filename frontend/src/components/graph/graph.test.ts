import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchGraphEntityDetail, fetchGraphOverview } from "../../api/client";
import { buildAdjacency, computeForceLayout, findComponents, hashUnit } from "./graphLayout";
import { colorForType } from "./graphPalette";

function starGraph(leafCount: number) {
  const nodes = [{ id: "hub" }, ...Array.from({ length: leafCount }, (_, index) => ({ id: `leaf-${index}` }))];
  const edges = Array.from({ length: leafCount }, (_, index) => ({
    source: "hub",
    target: `leaf-${index}`,
  }));
  return { nodes, edges };
}

describe("computeForceLayout", () => {
  it("returns one finite position inside the canvas for every node", () => {
    const { nodes, edges } = starGraph(8);

    const positions = computeForceLayout(nodes, edges);

    expect(positions.size).toBe(9);
    for (const node of nodes) {
      const point = positions.get(node.id);
      expect(point).toBeDefined();
      expect(Number.isFinite(point?.x)).toBe(true);
      expect(Number.isFinite(point?.y)).toBe(true);
      expect(point?.x ?? -1).toBeGreaterThanOrEqual(0);
      expect(point?.y ?? -1).toBeGreaterThanOrEqual(0);
      expect(point?.x ?? -1).toBeLessThanOrEqual(900);
      expect(point?.y ?? -1).toBeLessThanOrEqual(620);
    }
  });

  it("places connected nodes closer than the canvas diagonal", () => {
    const { nodes, edges } = starGraph(6);

    const positions = computeForceLayout(nodes, edges);

    const distances = edges.map((edge) => {
      const source = positions.get(edge.source);
      const target = positions.get(edge.target);
      return Math.hypot((source?.x ?? 0) - (target?.x ?? 0), (source?.y ?? 0) - (target?.y ?? 0));
    });
    for (const distance of distances) {
      expect(distance).toBeLessThan(Math.hypot(900, 620));
      expect(distance).toBeGreaterThan(0);
    }
  });

  it("is deterministic for one graph", () => {
    const { nodes, edges } = starGraph(5);

    const first = computeForceLayout(nodes, edges);
    const second = computeForceLayout(nodes, edges);

    expect([...first.entries()]).toEqual([...second.entries()]);
  });

  it("handles empty and single-node graphs", () => {
    expect(computeForceLayout([], []).size).toBe(0);

    const single = computeForceLayout([{ id: "only" }], []);

    expect(single.get("only")).toEqual({ x: 450, y: 310 });
  });
});

describe("buildAdjacency", () => {
  it("links both directions and ignores self references", () => {
    const adjacency = buildAdjacency([
      { source: "a", target: "b" },
      { source: "a", target: "a" },
    ]);

    expect([...(adjacency.get("a") ?? [])]).toEqual(["b"]);
    expect([...(adjacency.get("b") ?? [])]).toEqual(["a"]);
  });
});

describe("findComponents", () => {
  it("groups connected nodes and keeps isolated ones apart", () => {
    const nodes = ["a", "b", "c", "d", "solo"].map((id) => ({ id }));
    const edges = [
      { source: "a", target: "b" },
      { source: "b", target: "c" },
      { source: "c", target: "d" },
    ];

    expect(findComponents(nodes, edges)).toEqual([["a", "b", "c", "d"], ["solo"]]);
  });
});

describe("cluster packing", () => {
  function chainNodes(prefix: string, size: number) {
    return {
      nodes: Array.from({ length: size }, (_, index) => ({ id: `${prefix}${index}` })),
      edges: Array.from({ length: size - 1 }, (_, index) => ({
        source: `${prefix}${index}`,
        target: `${prefix}${index + 1}`,
      })),
    };
  }

  function starNodes(prefix: string, leaves: number) {
    const hub = `${prefix}hub`;
    return {
      nodes: [hub, ...Array.from({ length: leaves }, (_, index) => `${prefix}${index}`)].map(
        (id) => ({ id }),
      ),
      edges: Array.from({ length: leaves }, (_, index) => ({
        source: hub,
        target: `${prefix}${index}`,
      })),
    };
  }

  function centroid(
    points: Map<string, { x: number; y: number }>,
    ids: string[],
  ): { x: number; y: number } {
    const values = ids.map((id) => points.get(id) ?? { x: 0, y: 0 });
    return {
      x: values.reduce((total, point) => total + point.x, 0) / values.length,
      y: values.reduce((total, point) => total + point.y, 0) / values.length,
    };
  }

  it("keeps every community closer to itself than to another community", () => {
    const first = starNodes("a", 6);
    const second = starNodes("b", 6);
    const points = computeForceLayout(
      [...first.nodes, ...second.nodes],
      [...first.edges, ...second.edges],
    );
    const firstCenter = centroid(points, first.nodes.map((node) => node.id));
    const secondCenter = centroid(points, second.nodes.map((node) => node.id));
    const centerDistance = Math.hypot(
      firstCenter.x - secondCenter.x,
      firstCenter.y - secondCenter.y,
    );

    for (const node of first.nodes) {
      const point = points.get(node.id) ?? { x: 0, y: 0 };
      expect(Math.hypot(point.x - firstCenter.x, point.y - firstCenter.y)).toBeLessThan(
        centerDistance / 2,
      );
    }
  });

  it("keeps communities at least half a node spacing apart", () => {
    const first = starNodes("a", 7);
    const second = starNodes("b", 7);
    const points = computeForceLayout(
      [...first.nodes, ...second.nodes],
      [...first.edges, ...second.edges],
    );
    const distance = (left: { id: string }, right: { id: string }) => {
      const source = points.get(left.id) ?? { x: 0, y: 0 };
      const target = points.get(right.id) ?? { x: 0, y: 0 };
      return Math.hypot(source.x - target.x, source.y - target.y);
    };
    let smallestCross = Infinity;
    for (const left of first.nodes) {
      for (const right of second.nodes) {
        smallestCross = Math.min(smallestCross, distance(left, right));
      }
    }
    const insideEdges = [...first.edges];
    const meanInside =
      insideEdges.reduce(
        (total, edge) =>
          total +
          distance(
            { id: edge.source },
            { id: edge.target },
          ),
        0,
      ) / insideEdges.length;

    expect(smallestCross).toBeGreaterThan(meanInside * 0.5);
  });

  it("uses most of the canvas for a single community", () => {
    const chain = chainNodes("a", 14);
    const points = computeForceLayout(chain.nodes, chain.edges);
    const xs = [...points.values()].map((point) => point.x);
    const ys = [...points.values()].map((point) => point.y);

    expect(Math.max(...xs) - Math.min(...xs)).toBeGreaterThan(450);
    expect(Math.max(...ys) - Math.min(...ys)).toBeGreaterThan(150);
  });
});

describe("colorForType", () => {
  it("keeps one colour per entity type", () => {
    expect(colorForType("概念")).toBe(colorForType("概念"));
    expect(colorForType("概念")).toMatch(/^#[0-9a-f]{6}$/);
    expect(colorForType("机构")).not.toBe("");
  });

  it("spreads stable values across the palette", () => {
    const hashed = hashUnit("概念");
    expect(hashed).toBeGreaterThanOrEqual(0);
    expect(hashed).toBeLessThan(1);
  });
});

describe("graph api client", () => {
  afterEach(() => vi.restoreAllMocks());

  it("sends the bounded overview filters", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } }),
    );

    await fetchGraphOverview("kb-1", { search: "离心泵", entityType: "机构", limit: 100 });

    const url = String(fetchMock.mock.calls[0]?.[0]);
    expect(url).toContain("/api/v1/knowledge-bases/kb-1/graph?");
    expect(url).toContain("limit=100");
    expect(url).toContain("entity_type=%E6%9C%BA%E6%9E%84");
    expect(url).toContain("search=");
  });

  it("omits empty filters and keeps the default limit", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } }),
    );

    await fetchGraphOverview("kb-1");

    const url = String(fetchMock.mock.calls[0]?.[0]);
    expect(url).toContain("limit=200");
    expect(url).not.toContain("search=");
    expect(url).not.toContain("entity_type=");
  });

  it("requests one entity detail with its neighbour budget", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } }),
    );

    await fetchGraphEntityDetail("kb-1", "entity-a", { maxNeighbors: 12 });

    const url = String(fetchMock.mock.calls[0]?.[0]);
    expect(url).toContain("/api/v1/knowledge-bases/kb-1/graph/entities/entity-a?");
    expect(url).toContain("max_neighbors=12");
  });
});
