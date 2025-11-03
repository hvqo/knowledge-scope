import { describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";

import AnswerText from "./AnswerText.vue";
import type { RAGCitation } from "../../api/types";

const citations = [{ marker: "C1" }, { marker: "C2" }] as unknown as RAGCitation[];

function mountAnswer(content: string) {
  return mount(AnswerText, {
    props: { content, citations },
  });
}

describe("AnswerText", () => {
  it("renders headings, paragraphs and lists as real elements", () => {
    const wrapper = mountAnswer(
      ["总述一句。", "", "## 要点", "- 第一条 [C1]", "- 第二条 [C2]", "", "1. 步骤一", "2. 步骤二"].join("\n"),
    );

    expect(wrapper.findAll("p.answer-text__paragraph")).toHaveLength(1);
    expect(wrapper.get("h3.answer-text__heading").text()).toBe("要点");
    expect(wrapper.findAll("ul.answer-text__list li")).toHaveLength(2);
    expect(wrapper.findAll("ol.answer-text__list li")).toHaveLength(2);
    expect(wrapper.findAll("li")[0]?.text()).toContain("第一条");
  });

  it("renders inline formatting without leaking markdown syntax", () => {
    const wrapper = mountAnswer("先看 **重点** 与 `NO3`，见 [C1]。");

    expect(wrapper.get("strong").text()).toBe("重点");
    expect(wrapper.get("code").text()).toBe("NO3");
    expect(wrapper.text()).not.toContain("**");
    expect(wrapper.text()).not.toContain("`");
  });

  it("turns markers into chips and reports which one was clicked", async () => {
    const wrapper = mountAnswer("结论在这里[C1]。");

    const chip = wrapper.get("button.citation-marker");
    expect(chip.text()).toBe("[C1]");
    await chip.trigger("click");

    expect(wrapper.emitted("selectCitation")?.[0]).toEqual(["C1"]);
  });

  it("renders inline formulas with KaTeX", async () => {
    const wrapper = mountAnswer("当 $a>0$ 且 $a\\ne1$ 时，$x=\\log_a N$。");
    // KaTeX is loaded on demand, so the formulas appear one tick later.
    await vi.waitFor(() => expect(wrapper.findAll(".answer-math")).toHaveLength(3));

    const math = wrapper.findAll(".answer-math");
    expect(math).toHaveLength(3);
    // KaTeX replaces the source with its own markup instead of leaking LaTeX.
    expect(wrapper.text()).not.toContain("$a>0$");
    expect(math[0]?.html()).toContain("katex");
    // MathML keeps the source available for screen readers and copying.
    expect(math[1]?.html()).toContain("annotation");
    expect(math[0]?.attributes("title")).toBe("a>0");
  });

  it("renders a display formula as its own block", async () => {
    const wrapper = mountAnswer("推导：\n\n$$\\log_a 1 = 0$$\n\n结束。");
    await vi.waitFor(() => expect(wrapper.find(".answer-math.is-display").exists()).toBe(true));

    const display = wrapper.get(".answer-math.is-display");
    expect(display.html()).toContain("katex-display");
    expect(wrapper.findAll("p.answer-text__paragraph")).toHaveLength(2);
  });

  it("falls back to the raw formula when KaTeX cannot parse it", async () => {
    const wrapper = mountAnswer("公式 $\\frac{1}{$ 有问题");
    await vi.waitFor(() => expect(wrapper.find(".answer-math.is-raw").exists()).toBe(true));

    // KaTeX would throw; the source is shown so nothing is lost.
    expect(wrapper.text()).toContain("\\frac{1}{");
    expect(wrapper.findAll(".katex")).toHaveLength(0);
  });

  it("does not treat prices as formulas", () => {
    const wrapper = mountAnswer("预算 $5 到 $10 之间。");

    expect(wrapper.findAll(".answer-math")).toHaveLength(0);
    expect(wrapper.text()).toContain("$5 到 $10");
  });

  it("disables a marker the answer never retrieved", async () => {
    const wrapper = mountAnswer("结论在这里[C9]。");

    const chip = wrapper.get("button.citation-marker");
    expect(chip.attributes("disabled")).toBeDefined();
    await chip.trigger("click");

    expect(wrapper.emitted("selectCitation")).toBeUndefined();
  });
});
