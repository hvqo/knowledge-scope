import { describe, expect, it } from "vitest";

import { normalizeCitationLayout, referencedCitations } from "./citations";
import type { RAGCitation } from "../../api/types";

describe("normalizeCitationLayout", () => {
  it("keeps well placed markers untouched", () => {
    expect(normalizeCitationLayout("机械增氧利用增氧机搅动水体[C1]。")).toBe(
      "机械增氧利用增氧机搅动水体[C1]。",
    );
  });

  it("glues a marker that the model left on its own line", () => {
    expect(normalizeCitationLayout("原形与过去式对照如下：\n\n[C1]\n\n- am / is")).toBe(
      "原形与过去式对照如下：[C1]\n\n- am / is",
    );
  });

  it("never rewrites the answer's own spacing or punctuation", () => {
    expect(normalizeCitationLayout("回答 [C1]")).toBe("回答 [C1]");
    expect(normalizeCitationLayout("资料中给出了不规则变化[C1]。")).toBe(
      "资料中给出了不规则变化[C1]。",
    );
    expect(normalizeCitationLayout("资料中给出了不规则变化。[C1]")).toBe(
      "资料中给出了不规则变化。[C1]",
    );
  });

  it("does not treat paragraph breaks outside markers as citation layout", () => {
    const text = "第一段。\n\n第二段另起一行。";
    expect(normalizeCitationLayout(text)).toBe(text);
  });
});

describe("referencedCitations", () => {
  const citations: RAGCitation[] = [
    { marker: "C1" },
    { marker: "C2" },
  ] as unknown as RAGCitation[];

  it("keeps only the markers the answer used", () => {
    expect(referencedCitations("答案[C2]", citations)).toEqual([{ marker: "C2" }]);
    expect(referencedCitations("没有引用", citations)).toEqual([]);
  });
});
