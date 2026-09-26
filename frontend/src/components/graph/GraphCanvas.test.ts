import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import type { GraphEdge, GraphNode } from "../../api/types";
import GraphCanvas from "./GraphCanvas.vue";

function makeNode(id: string, degree: number, name = id): GraphNode {
  return {
    entity_id: id,
    canonical_name: name,
    entity_type: "概念",
    aliases: [],
    document_id: "doc-1",
    canonical_entity_ids: [],
    evidence_count: 1,
    degree,
  };
}

function makeEdge(relationId: string, source: string, target: string): GraphEdge {
  return {
    relation_id: relationId,
    source_entity_id: source,
    target_entity_id: target,
    relation_type: "包含",
    kind: "relation",
    document_id: "doc-1",
  };
}

function chainGraph(size: number, isolated = 0) {
  const nodes = Array.from({ length: size }, (_, index) => makeNode(`e${index}`, index === 0 ? 6 : 1));
  const edges = Array.from({ length: size - 1 }, (_, index) =>
    makeEdge(`r${index}`, `e${index}`, `e${index + 1}`),
  );
  for (let index = 0; index < isolated; index += 1) {
    nodes.push(makeNode(`i${index}`, 0, `孤立${index}`));
  }
  return { nodes, edges };
}

function toggleByText(wrapper: ReturnType<typeof mount>, text: string) {
  const toggle = wrapper
    .findAll(".canvas-toggle")
    .find((candidate) => candidate.text().includes(text));
  expect(toggle).toBeDefined();
  return toggle?.get("input");
}

describe("GraphCanvas", () => {
  it("labels only hubs until the user asks for every label", async () => {
    const { nodes, edges } = chainGraph(40);
    const wrapper = mount(GraphCanvas, {
      props: { nodes, edges, selectedId: null },
    });

    const labelled = wrapper.findAll(".node-label").length;
    expect(labelled).toBeGreaterThan(0);
    expect(labelled).toBeLessThan(6);

    await toggleByText(wrapper, "全部标签")?.setValue(true);

    expect(wrapper.findAll(".node-label")).toHaveLength(40);
  });

  it("can hide unconnected entities", async () => {
    const { nodes, edges } = chainGraph(6, 3);
    const wrapper = mount(GraphCanvas, {
      props: { nodes, edges, selectedId: null },
    });

    expect(wrapper.findAll(".graph-node")).toHaveLength(9);

    await toggleByText(wrapper, "隐藏未连接")?.setValue(true);

    expect(wrapper.findAll(".graph-node")).toHaveLength(6);
    expect(wrapper.text()).toContain("6 个实体 · 5 条关系");
  });

  it("zooms to the focused entity instead of requiring manual dragging", async () => {
    const { nodes, edges } = chainGraph(40);
    const wrapper = mount(GraphCanvas, {
      props: { nodes, edges, selectedId: null },
    });
    const viewport = wrapper.get(".graph-canvas > g");
    expect(viewport.attributes("transform")).toContain("scale(1)");

    await wrapper.findAll(".graph-node")[5].trigger("click");
    expect(wrapper.emitted("select")?.at(-1)).toEqual(["e5"]);
    await wrapper.setProps({ selectedId: "e5" });

    const focusButton = wrapper
      .findAll(".canvas-button")
      .find((button) => button.text().includes("聚焦选中"));
    expect(focusButton?.attributes("disabled")).toBeUndefined();
    await focusButton?.trigger("click");

    const transform = wrapper.get(".graph-canvas > g").attributes("transform") ?? "";
    expect(transform).not.toContain("scale(1)");
    expect(transform).toMatch(/scale\(([2-9]|1\.[0-9])/);
  });

  it("keeps focus available for the selected entity only", () => {
    const { nodes, edges } = chainGraph(6);
    const wrapper = mount(GraphCanvas, {
      props: { nodes, edges, selectedId: null },
    });

    const focusButton = wrapper
      .findAll(".canvas-button")
      .find((button) => button.text().includes("聚焦选中"));

    expect(focusButton?.attributes("disabled")).toBeDefined();
  });
});
