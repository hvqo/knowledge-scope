# ruff: noqa: RUF001
"""Application-level trust rules for vision-derived facts.

The vision model is treated as an untrusted reading aid, never as an
authoritative data source.  This module classifies question risk, extracts
numeric claims from model output, and produces conservative deterministic
answers for high-risk visual numeric questions.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

# Questions that ask for a precise number, growth rate, ranking, or aggregate.
_ANALYTIC_NUMBER_PATTERNS = re.compile(
    "多少|具体|精确|准确|百分|百分之|增长|下降|降幅|增幅|最低|最高|最大|最小|"
    "排名|合计|总和|总计|平均|占比|比例|差值|相差|金额|总量|总额"
)
# Questions that explicitly target a visual artifact.
_VISUAL_INTENT_PATTERNS = re.compile("图|图表|图片|示意图|曲线|柱状|折线|饼图|扫描|照片|图形|海报")
# A conservative numeric token matcher: integers, decimals, thousands separators,
# percentages, and the common Chinese magnitude suffixes used in captions.
_NUMERIC_TOKEN_PATTERN = re.compile(
    r"(?<![0-9A-Za-z_.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"
    r"|(?<![0-9A-Za-z_.])[-+]?\d+(?:\.\d+)?%?"
)
_YEAR_PATTERN = re.compile(r"^(19|20)\d{2}$")
_QUARTER_PATTERN = re.compile(r"^[Qq][1-4]$")
_CITATION_PATTERN = re.compile(r"\[C[1-9][0-9]*\]")


@dataclass(frozen=True, slots=True)
class NumericClaim:
    """One number found in untrusted (model) output with a nearby context."""

    value: str
    context: str


@dataclass(frozen=True, slots=True)
class VisualRiskAssessment:
    """How strongly a question depends on precise numbers from an image."""

    high_risk: bool
    visual_intent: bool
    reasons: list[str] = field(default_factory=list)


def assess_visual_risk(query: str) -> VisualRiskAssessment:
    """Classify whether the question asks for precise numbers from a visual."""

    text = _CITATION_PATTERN.sub(" ", query).strip()
    reasons: list[str] = []
    analytic_match = _ANALYTIC_NUMBER_PATTERNS.search(text)
    if analytic_match is not None:
        reasons.append(f"问题涉及精确数值/计算（{analytic_match.group(0)}）")
    visual_match = _VISUAL_INTENT_PATTERNS.search(text)
    if visual_match is not None:
        reasons.append(f"问题指向图片类素材（{visual_match.group(0)}）")
    return VisualRiskAssessment(
        high_risk=analytic_match is not None,
        visual_intent=visual_match is not None,
        reasons=reasons,
    )


def is_ignorable_numeric_token(token: str) -> bool:
    """Return True for years, quarter labels, and citation-shaped fragments."""

    stripped = token.strip().strip("[]")
    return bool(_YEAR_PATTERN.match(stripped) or _QUARTER_PATTERN.match(stripped))


def extract_numeric_claims(text: str) -> list[NumericClaim]:
    """Extract numeric claims from untrusted model text, preserving context."""

    claims: list[NumericClaim] = []
    seen: set[tuple[str, int]] = set()
    for match in _NUMERIC_TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        if is_ignorable_numeric_token(token):
            continue
        start = max(match.start() - 24, 0)
        end = min(match.end() + 24, len(text))
        context = " ".join(text[start:end].split())
        key = (token, match.start())
        if key in seen:
            continue
        seen.add(key)
        claims.append(NumericClaim(value=token, context=context))
    return claims


def answer_has_numeric_claims(text: str) -> bool:
    """Return True when untrusted text contains at least one numeric claim."""

    return bool(extract_numeric_claims(text))


def strip_numeric_claims(text: str) -> str:
    """Replace every non-label number in untrusted text with a review marker."""

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if is_ignorable_numeric_token(token):
            return token
        return "〔数值未经核验〕"

    return _NUMERIC_TOKEN_PATTERN.sub(_replace, text)


_GATED_ANSWER_HEADER = (
    "当前问题需要从图片中读取精确数值，但系统无法核验视觉模型给出的数字，"
    "因此不会把具体数值作为结论输出。"
)
_GATED_ANSWER_FOOTER = (
    "如需精确数值，请使用包含结构化表格的文档获取确定性计算结果，"
    "或人工核对原始图片。本次实际分析的图片已在下方展示，可点击查看原图。"
)
_GATED_ANSWER_UNAVAILABLE = (
    "本次视觉分析未能完成，系统不会根据图片猜测精确数值。"
    "建议改用包含结构化表格的文档，或人工核对原始图片后重试。"
)


def build_gated_visual_answer(
    *,
    description: str | None,
    vision_completed: bool,
    verified_facts: Sequence[str] | None = None,
) -> tuple[str, str]:
    """Build a deterministic, number-free visual answer plus a safety notice.

    Returns ``(answer, notice)``.  The answer never contains a precise number
    from the vision model, so it is safe to stream once built.  ``verified_facts``
    may carry deterministic table-tool results, which are trusted application
    output (not model text) and therefore allowed to contain numbers.
    """

    facts = [item for item in (verified_facts or []) if item]
    fact_block = "\n".join(facts)
    if not vision_completed:
        if facts:
            answer = (
                _GATED_ANSWER_HEADER
                + "\n\n视觉分析未能完成，因此没有给出任何图片中的精确数值。\n\n"
                + "以下为表格工具给出的确定性计算结果（已核验）：\n"
                + fact_block
                + "\n\n"
                + _GATED_ANSWER_FOOTER
            )
            return answer, "视觉分析未完成，已阻止未核验的数值输出；表格计算结果已核验"
        return _GATED_ANSWER_UNAVAILABLE, "视觉分析未完成，已阻止未核验的数值输出"
    sections = [_GATED_ANSWER_HEADER]
    if facts:
        sections.append("以下为表格工具给出的确定性计算结果（已核验）：\n" + fact_block)
    if description:
        safe_description = strip_numeric_claims(description).strip()
        if safe_description:
            sections.append(
                f"视觉模型给出的未经核验的描述如下（具体数值已隐去）：\n{safe_description[:500]}"
            )
    sections.append(_GATED_ANSWER_FOOTER)
    return "\n\n".join(sections), "视觉数值未经核验，已省略精确数字并标注为视觉估计"


__all__ = [
    "NumericClaim",
    "VisualRiskAssessment",
    "answer_has_numeric_claims",
    "assess_visual_risk",
    "build_gated_visual_answer",
    "extract_numeric_claims",
    "is_ignorable_numeric_token",
    "strip_numeric_claims",
]
