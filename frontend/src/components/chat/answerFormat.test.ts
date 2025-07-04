import { describe, expect, it } from "vitest";

import { parseAnswer, parseInline } from "./answerFormat";

describe("parseInline", () => {
  it("keeps text, bold, code and citation markers apart", () => {
    expect(parseInline("先看 **重点** 与 `NO3`，见 [C2]。")).toEqual([
      { kind: "text", value: "先看 " },
      { kind: "bold", value: "重点" },
      { kind: "text", value: " 与 " },
      { kind: "code", value: "NO3" },
      { kind: "text", value: "，见 " },
      { kind: "marker", value: "[C2]" },
      { kind: "text", value: "。" },
    ]);
  });

  it("leaves unknown syntax as plain text", () => {
    expect(parseInline("a ~b~ *c*")).toEqual([{ kind: "text", value: "a ~b~ *c*" }]);
  });

  it("recognizes inline formulas", () => {
    expect(parseInline("当 $a>0$ 且 $a\\ne1$ 时，$a^x=N$ 成立。")).toEqual([
      { kind: "text", value: "当 " },
      { kind: "math", value: "a>0" },
      { kind: "text", value: " 且 " },
      { kind: "math", value: "a\\ne1" },
      { kind: "text", value: " 时，" },
      { kind: "math", value: "a^x=N" },
      { kind: "text", value: " 成立。" },
    ]);
  });

  it("recognizes single-variable math like $x$ or $N$", () => {
    expect(parseInline("数 $x$ 叫做以 $a$ 为底 $N$ 的对数")).toEqual([
      { kind: "text", value: "数 " },
      { kind: "math", value: "x" },
      { kind: "text", value: " 叫做以 " },
      { kind: "math", value: "a" },
      { kind: "text", value: " 为底 " },
      { kind: "math", value: "N" },
      { kind: "text", value: " 的对数" },
    ]);
  });

  it("keeps dollar amounts as text", () => {
    expect(parseInline("预算 $5 到 $10 之间。")).toEqual([
      { kind: "text", value: "预算 $5 到 $10 之间。" },
    ]);
    expect(parseInline("单价 $120 元")).toEqual([{ kind: "text", value: "单价 $120 元" }]);
  });

  it("keeps an escaped dollar literal", () => {
    expect(parseInline("成本是 \\$5 与 $x=1$")).toEqual([
      { kind: "text", value: "成本是 $5 与 " },
      { kind: "math", value: "x=1" },
    ]);
  });
});

describe("parseAnswer", () => {
  it("splits paragraphs, headings and lists", () => {
    const blocks = parseAnswer(
      [
        "## 结论",
        "硝酸具有强氧化性。",
        "",
        "要点：",
        "- 与指示剂反应 [C1]",
        "- 与金属反应 [C2]",
        "",
        "1. 第一步",
        "2. 第二步",
      ].join("\n"),
    );

    expect(blocks.map((block) => block.kind)).toEqual([
      "heading",
      "paragraph",
      "paragraph",
      "list",
      "list",
    ]);
    const [heading, , , bulletList, orderedList] = blocks;
    expect(heading).toEqual({
      kind: "heading",
      level: 2,
      content: [{ kind: "text", value: "结论" }],
    });
    expect(bulletList).toMatchObject({ kind: "list", ordered: false });
    expect(orderedList).toMatchObject({ kind: "list", ordered: true });
  });

  it("keeps an enumeration inside one list", () => {
    const blocks = parseAnswer("- 甲\n- 乙\n- 丙");

    expect(blocks).toHaveLength(1);
    expect(blocks[0]).toMatchObject({ kind: "list", ordered: false });
    expect((blocks[0] as { items: unknown[] }).items).toHaveLength(3);
  });

  it("keeps single line breaks inside one paragraph", () => {
    const blocks = parseAnswer("第一行\n第二行");

    expect(blocks).toHaveLength(1);
    expect(blocks[0]).toEqual({
      kind: "paragraph",
      content: [{ kind: "text", value: "第一行\n第二行" }],
    });
  });

  it("pulls a marker off its own line onto the sentence above", () => {
    const blocks = parseAnswer("结论如下：\n\n[C1]\n\n下一步。");

    expect(blocks[0]).toMatchObject({
      kind: "paragraph",
      content: [{ kind: "text", value: "结论如下：" }, { kind: "marker", value: "[C1]" }],
    });
  });

  it("handles an empty answer", () => {
    expect(parseAnswer("")).toEqual([]);
  });

  it("parses display formulas on their own line", () => {
    const blocks = parseAnswer("公式如下：\n\n$$x=\\log_a N$$\n\n下一段。");

    expect(blocks.map((block) => block.kind)).toEqual(["paragraph", "math", "paragraph"]);
    expect(blocks[1]).toEqual({ kind: "math", value: "x=\\log_a N" });
  });

  it("parses a fenced display formula across lines", () => {
    const blocks = parseAnswer("推导：\n\n$$\n\\log_a 1 = 0\n$$\n");

    expect(blocks[1]).toEqual({ kind: "math", value: "\\log_a 1 = 0" });
    expect(blocks).toHaveLength(2);
  });

  it("keeps an unterminated formula instead of dropping it", () => {
    const blocks = parseAnswer("$$\n\\log_a a = 1");

    expect(blocks[0]).toEqual({ kind: "math", value: "\\log_a a = 1" });
  });
});
