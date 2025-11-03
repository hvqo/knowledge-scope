/**
 * Turn one answer into blocks the chat can lay out properly.
 *
 * The model writes short markdown — `##` headings, `-`/`1.` lists, **bold**,
 * `code`, `$inline math$`, `$$display math$$` — plus `[Cn]` citation markers.
 * A full markdown pipeline is overkill here: unknown syntax stays literal text
 * instead of disappearing.
 */

import { normalizeCitationLayout } from "./citations";

export type AnswerInline =
  | { kind: "text"; value: string }
  | { kind: "marker"; value: string }
  | { kind: "bold"; value: string }
  | { kind: "code"; value: string }
  | { kind: "math"; value: string };

export type AnswerBlock =
  | { kind: "heading"; level: 2 | 3; content: AnswerInline[] }
  | { kind: "list"; ordered: boolean; items: AnswerInline[][] }
  | { kind: "paragraph"; content: AnswerInline[] }
  | { kind: "math"; value: string };

const HEADING_PATTERN = /^#{2,4}\s+(.+)$/;
const BULLET_PATTERN = /^\s*[-*•]\s+(.+)$/;
const ORDERED_PATTERN = /^\s*\d+[.、)]\s+(.+)$/;
const DISPLAY_MATH_LINE_PATTERN = /^\s*\$\$(.+?)\$\$\s*$/;
const DISPLAY_MATH_FENCE_PATTERN = /^\s*\$\$\s*$/;
const ESCAPED_DOLLAR_PLACEHOLDER = "\u0000";
const INLINE_PATTERN = /(\$\$[^$]+\$\$|\$[^$\n]+\$|\*\*[^*]+\*\*|`[^`]+`|\[C[1-9][0-9]*\])/g;
const MARKER_PATTERN = /^\[C[1-9][0-9]*\]$/;

/**
 * Only treat `$…$` as math when it actually looks like math.
 *
 * Industry documents are full of prices ("预算 $5 到 $10"), so a bare pair of
 * dollar signs is not enough.  Formulas never contain CJK, and they always
 * carry a letter (even a lone `$x$`) or a LaTeX command — anything else is
 * money, a list number, or plain text.
 */
const MATH_LETTER_PATTERN = /[a-zA-Z]/;
const MATH_CJK_PATTERN = /[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]/;
const MATH_MAX_CHARACTERS = 400;

export function looksLikeMath(body: string): boolean {
  const trimmed = body.trim();
  if (trimmed === "" || trimmed.length > MATH_MAX_CHARACTERS) {
    return false;
  }
  // Chinese text between two dollar signs is a price or an aside, not a formula.
  if (MATH_CJK_PATTERN.test(trimmed)) {
    return false;
  }
  // A lone number or a money range is not a formula.
  if (!MATH_LETTER_PATTERN.test(trimmed)) {
    return false;
  }
  return true;
}


export function parseInline(text: string): AnswerInline[] {
  const prepared = text.replaceAll("\\$", ESCAPED_DOLLAR_PLACEHOLDER);
  const tokens: AnswerInline[] = [];
  let cursor = 0;
  for (const match of prepared.matchAll(INLINE_PATTERN)) {
    const index = match.index ?? 0;
    const value = match[0];
    const mathBody = value.startsWith("$$")
      ? value.slice(2, -2)
      : value.startsWith("$")
        ? value.slice(1, -1)
        : null;
    if (mathBody !== null && !looksLikeMath(mathBody)) {
      // Not a formula (a price, for example): keep scanning after it.
      continue;
    }
    if (index > cursor) {
      tokens.push({ kind: "text", value: prepared.slice(cursor, index) });
    }
    if (MARKER_PATTERN.test(value)) {
      tokens.push({ kind: "marker", value });
    } else if (mathBody !== null) {
      tokens.push({ kind: "math", value: mathBody.trim() });
    } else if (value.startsWith("**")) {
      tokens.push({ kind: "bold", value: value.slice(2, -2) });
    } else {
      tokens.push({ kind: "code", value: value.slice(1, -1) });
    }
    cursor = index + value.length;
  }
  if (cursor < prepared.length) {
    tokens.push({ kind: "text", value: prepared.slice(cursor) });
  }
  return tokens.map((token) =>
    token.kind === "text"
      ? { kind: "text", value: token.value.replaceAll(ESCAPED_DOLLAR_PLACEHOLDER, "$") }
      : token,
  );
}

export function parseAnswer(content: string): AnswerBlock[] {
  const blocks: AnswerBlock[] = [];
  let paragraph: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;

  const flushParagraph = (): void => {
    if (paragraph.length === 0) {
      return;
    }
    blocks.push({ kind: "paragraph", content: parseInline(paragraph.join("\n")) });
    paragraph = [];
  };
  const flushList = (): void => {
    if (list === null) {
      return;
    }
    blocks.push({
      kind: "list",
      ordered: list.ordered,
      items: list.items.map((item) => parseInline(item)),
    });
    list = null;
  };

  let mathFence: string[] | null = null;
  for (const rawLine of normalizeCitationLayout(content).split("\n")) {
    const line = rawLine.trimEnd();
    if (mathFence !== null) {
      if (DISPLAY_MATH_FENCE_PATTERN.test(line)) {
        blocks.push({ kind: "math", value: mathFence.join("\n").trim() });
        mathFence = null;
      } else {
        mathFence.push(line);
      }
      continue;
    }
    if (line.trim() === "") {
      flushParagraph();
      flushList();
      continue;
    }
    if (DISPLAY_MATH_FENCE_PATTERN.test(line)) {
      flushParagraph();
      flushList();
      mathFence = [];
      continue;
    }
    const displayMath = DISPLAY_MATH_LINE_PATTERN.exec(line);
    if (displayMath !== null) {
      flushParagraph();
      flushList();
      blocks.push({ kind: "math", value: (displayMath[1] ?? "").trim() });
      continue;
    }
    const heading = HEADING_PATTERN.exec(line);
    if (heading !== null) {
      flushParagraph();
      flushList();
      blocks.push({
        kind: "heading",
        level: line.startsWith("###") ? 3 : 2,
        content: parseInline(heading[1] ?? ""),
      });
      continue;
    }
    const bullet = BULLET_PATTERN.exec(line);
    const ordered = bullet === null ? ORDERED_PATTERN.exec(line) : null;
    const item = bullet?.[1] ?? ordered?.[1];
    if (item !== undefined) {
      flushParagraph();
      const wantsOrdered = ordered !== null;
      if (list === null || list.ordered !== wantsOrdered) {
        flushList();
        list = { ordered: wantsOrdered, items: [] };
      }
      list.items.push(item);
      continue;
    }
    flushList();
    paragraph.push(line);
  }
  flushParagraph();
  flushList();
  if (mathFence !== null && mathFence.length > 0) {
    // An unterminated fence still renders: better than dropping the formula.
    blocks.push({ kind: "math", value: mathFence.join("\n").trim() });
  }
  return blocks;
}
