# ruff: noqa: RUF001
"""Explainable image-index policy for selective visual vectorization.

The policy estimates whether the textual representation (caption/context) is
likely sufficient for retrieval.  It is deliberately conservative: a missing
or low-confidence signal selects ``text_and_visual`` so visual recall is never
lost silently.  It is not a model; every decision carries a readable reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

VISUAL_INDEX_POLICY_VERSION = "visual-index-policy-v1"

VisualIndexMode = Literal["text_only", "text_and_visual"]

_VISUAL_DETAIL_PATTERN = re.compile(
    r"图|表|示意|流程|趋势|统计|柱状|折线|曲线|饼图|散点|地图|结构|示意|关系|分布|对比|比较"
)
_GENERIC_CAPTION_PATTERN = re.compile(r"^(图|表)\s*[\d一二三四五六七八九十]+[\s.、:：-]*$")
_MIN_CAPTION_CHARS = 15
_MIN_CONTEXT_CHARS = 80
_LARGE_IMAGE_PIXELS = 600_000


@dataclass(frozen=True)
class VisualPolicyDecision:
    """One deterministic index decision with human-readable evidence."""

    mode: VisualIndexMode
    reason: str
    policy_version: str
    signals: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "policy_version": self.policy_version,
            "signals": self.signals,
        }


def decide_image_index(
    *,
    caption: str | None,
    context: str | None,
    image_width: int | None,
    image_height: int | None,
    page_image_count: int | None,
    page_has_table: bool = False,
    asset_size_bytes: int | None = None,
) -> VisualPolicyDecision:
    """Return an explainable index decision for one image asset."""

    caption_text = (caption or "").strip()
    context_text = (context or "").strip()
    caption_chars = len(caption_text)
    context_chars = len(context_text)
    image_pixels = None
    if image_width and image_height:
        image_pixels = int(image_width) * int(image_height)
    aspect_ratio = None
    if image_width and image_height and image_height > 0:
        aspect_ratio = round(float(image_width) / float(image_height), 3)
    signals: dict[str, Any] = {
        "caption_chars": caption_chars,
        "context_chars": context_chars,
        "image_pixels": image_pixels,
        "aspect_ratio": aspect_ratio,
        "page_image_count": page_image_count,
        "page_has_table": page_has_table,
        "asset_size_bytes": asset_size_bytes,
    }

    def visual(reason: str) -> VisualPolicyDecision:
        return VisualPolicyDecision(
            mode="text_and_visual",
            reason=reason,
            policy_version=VISUAL_INDEX_POLICY_VERSION,
            signals=signals,
        )

    if not caption_text:
        context_sufficient = (
            context_chars >= 100
            and page_image_count == 1
            and not page_has_table
            and (image_pixels is None or image_pixels <= _LARGE_IMAGE_PIXELS)
            and not _VISUAL_DETAIL_PATTERN.search(context_text)
        )
        if context_sufficient:
            return VisualPolicyDecision(
                mode="text_only",
                reason="缺少图片标题，但页面上下文描述较完整且无视觉细节信号；先依赖文本索引",
                policy_version=VISUAL_INDEX_POLICY_VERSION,
                signals=signals,
            )
        return visual("缺少图片标题，纯文本索引无法覆盖视觉内容")
    if _GENERIC_CAPTION_PATTERN.match(caption_text):
        return visual("图片标题只有编号、没有描述性文字")
    if caption_chars < _MIN_CAPTION_CHARS:
        return visual(f"图片标题过短（{caption_chars} 字），信息量不足")
    if context_chars < _MIN_CONTEXT_CHARS:
        return visual(f"图片上下文过短（{context_chars} 字），文本表示可能不完整")
    if page_has_table:
        return visual("同页包含表格，版面较复杂；保守建立视觉索引")
    if page_image_count is None or page_image_count > 1:
        return visual(f"同页包含多张图片（{page_image_count}），需要视觉区分")
    if image_pixels is None or image_pixels > _LARGE_IMAGE_PIXELS:
        return visual("图片像素面积较大或未知，视觉细节可能超出标题描述")
    if _VISUAL_DETAIL_PATTERN.search(caption_text):
        return visual("图片标题包含图表/流程/结构等视觉描述词，需要原生视觉表示")
    if context_text and _VISUAL_DETAIL_PATTERN.search(context_text):
        return visual("图片上下文包含视觉描述词，需要原生视觉表示")
    return VisualPolicyDecision(
        mode="text_only",
        reason="图片标题与上下文信息较完整，且无视觉细节信号；先依赖文本索引",
        policy_version=VISUAL_INDEX_POLICY_VERSION,
        signals=signals,
    )


__all__ = [
    "VISUAL_INDEX_POLICY_VERSION",
    "VisualIndexMode",
    "VisualPolicyDecision",
    "decide_image_index",
]
