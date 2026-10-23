# ruff: noqa: RUF001
"""Request, provider-payload and public result contracts for vision analysis."""

from __future__ import annotations

import json
import re
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import VisionResponseFormatError

VisionTaskType = Literal["describe_image", "analyze_chart", "extract_table"]
VisionStatus = Literal["ok", "needs_review", "not_configured", "failed"]
# Application-assigned provenance, never a model self-assessment.
VisualFactStatus = Literal[
    "source_verified",  # deterministic structured table evidence / calculation
    "visual_estimate",  # read from an image by the VLM; estimate only
    "unverified",  # extracted but not validated (e.g. image-table extraction)
    "not_available",  # no trustworthy visual data is available
]

EVIDENCE_ID_PATTERN: str = r"^evidence_v1_[0-9a-f]{64}$"
MAX_QUESTION_CHARS: int = 2_000
MAX_TABLE_ROWS: int = 100
MAX_TABLE_COLUMNS: int = 30
MAX_CELL_CHARS: int = 500
MAX_LIST_ITEMS: int = 20
MAX_TEXT_CHARS: int = 4_000
MAX_WARNINGS: int = 20

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_REQUIRED_TABLE_WARNING = "视觉模型提取，数值需核对，不能作为精确统计数据使用"


class VisionAnalysisRequest(BaseModel):
    """One explicit, read-only vision analysis request scoped to one evidence asset."""

    model_config = ConfigDict(extra="forbid")

    knowledge_base_id: UUID
    document_id: UUID
    evidence_id: str = Field(pattern=EVIDENCE_ID_PATTERN)
    asset_index: int = Field(default=0, ge=0, le=255)
    task_type: VisionTaskType
    question: str | None = Field(default=None, max_length=MAX_QUESTION_CHARS)


class VisionSource(BaseModel):
    """Traceable origin of the analysed image."""

    knowledge_base_id: UUID
    document_id: UUID
    evidence_id: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    source_block_ids: list[str]


class ChartAnalysis(BaseModel):
    """Normalized chart reading returned by the vision model."""

    chart_type: str | None = None
    axes: list[str] = Field(default_factory=list)
    series: list[str] = Field(default_factory=list)
    trends: list[str] = Field(default_factory=list)
    peaks: list[str] = Field(default_factory=list)
    anomalies: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ExtractedTable(BaseModel):
    """Read-only table extraction; values are untrusted suggestions, never facts."""

    title: str | None = None
    unit: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str | None]] = Field(default_factory=list)
    merged_cell_info: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    extraction_method: Literal["vlm"] = "vlm"


class VisionNumericClaim(BaseModel):
    """One numeric claim extracted from untrusted visual-model output."""

    value: str = Field(min_length=1, max_length=64)
    context: str | None = Field(default=None, max_length=200)
    fact_status: VisualFactStatus = "visual_estimate"


class VisionAnalysisResult(BaseModel):
    """Uniform, JSON-serializable result for every vision task outcome."""

    status: VisionStatus
    task_type: VisionTaskType
    # Application-assigned provenance of the result; never trust model claims.
    fact_status: VisualFactStatus = "not_available"
    numeric_claims: list[VisionNumericClaim] = Field(default_factory=list)
    answer: str | None = None
    chart_analysis: ChartAnalysis | None = None
    extracted_table: ExtractedTable | None = None
    extraction_method: Literal["vlm"] | None = None
    provider: str | None = None
    model: str | None = None
    latency_ms: float | None = None
    warnings: list[str] = Field(default_factory=list)
    sources: list[VisionSource] = Field(default_factory=list)
    error_code: str | None = None
    message: str | None = None


class _DescriptionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    subject: str = ""
    key_content: list[Any] = Field(default_factory=list)
    visible_text: list[Any] = Field(default_factory=list)
    answer: str = ""
    uncertain: list[Any] = Field(default_factory=list)


class _ChartPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chart_type: str | None = None
    axes: list[Any] = Field(default_factory=list)
    series: list[Any] = Field(default_factory=list)
    trends: list[Any] = Field(default_factory=list)
    peaks: list[Any] = Field(default_factory=list)
    anomalies: list[Any] = Field(default_factory=list)
    limitations: list[Any] = Field(default_factory=list)


class _TablePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    unit: str | None = None
    columns: list[Any] = Field(default_factory=list)
    rows: list[Any] = Field(default_factory=list)
    merged_cell_info: list[Any] = Field(default_factory=list)
    warnings: list[Any] = Field(default_factory=list)


def extract_json_object(content: str) -> dict[str, Any]:
    """Extract one JSON object from fenced or lightly decorated model output."""
    cleaned = content.strip()
    fenced = _JSON_FENCE_RE.search(cleaned)
    if fenced is not None:
        cleaned = fenced.group(1).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise VisionResponseFormatError("视觉模型未返回可解析的 JSON") from None
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as error:
            raise VisionResponseFormatError("视觉模型未返回可解析的 JSON") from error
    if not isinstance(payload, dict):
        raise VisionResponseFormatError("视觉模型返回的 JSON 不是对象")
    return payload


def _text(value: Any, *, limit: int = MAX_TEXT_CHARS) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text[:limit]


def _string_items(values: list[Any], *, limit: int = MAX_LIST_ITEMS) -> list[str]:
    items: list[str] = []
    for value in values:
        text = _text(value, limit=MAX_CELL_CHARS)
        if text:
            items.append(text)
        if len(items) >= limit:
            break
    return items


def _unknown_field_warnings(payload: dict[str, Any], allowed: frozenset[str]) -> list[str]:
    unknown = sorted(set(payload) - allowed)
    if not unknown:
        return []
    return ["视觉模型返回的未知字段已忽略：" + ", ".join(unknown[:5])]


def normalize_description(content: str) -> tuple[str, list[str]]:
    """Normalize a describe_image answer into readable text plus warnings."""
    raw = extract_json_object(content)
    try:
        parsed = _DescriptionPayload.model_validate(raw)
    except ValidationError as error:
        raise VisionResponseFormatError("视觉模型输出不符合图片理解契约") from error
    warnings = _unknown_field_warnings(
        raw,
        frozenset({"subject", "key_content", "visible_text", "answer", "uncertain"}),
    )
    sections: list[str] = []
    subject = _text(parsed.subject, limit=1_000)
    answer = _text(parsed.answer)
    if answer:
        sections.append(answer)
    elif subject:
        sections.append(f"图片主体：{subject}")
    key_content = _string_items(parsed.key_content)
    if key_content:
        sections.append("关键内容：\n" + "\n".join(f"- {item}" for item in key_content))
    visible_text = _string_items(parsed.visible_text)
    if visible_text:
        sections.append("图中文字：\n" + "\n".join(f"- {item}" for item in visible_text))
    uncertain = _string_items(parsed.uncertain)
    if uncertain:
        warnings.append("以下内容无法确认：" + "；".join(uncertain))
    if not sections:
        raise VisionResponseFormatError("视觉模型未给出可用的图片描述")
    return "\n\n".join(sections), warnings


def normalize_chart(content: str) -> tuple[str, ChartAnalysis, list[str]]:
    """Normalize an analyze_chart answer into a structured chart reading."""
    raw = extract_json_object(content)
    try:
        parsed = _ChartPayload.model_validate(raw)
    except ValidationError as error:
        raise VisionResponseFormatError("视觉模型输出不符合图表分析契约") from error
    warnings = _unknown_field_warnings(
        raw,
        frozenset({"chart_type", "axes", "series", "trends", "peaks", "anomalies", "limitations"}),
    )
    chart = ChartAnalysis(
        chart_type=_text(parsed.chart_type, limit=200) or None,
        axes=_string_items(parsed.axes),
        series=_string_items(parsed.series),
        trends=_string_items(parsed.trends),
        peaks=_string_items(parsed.peaks),
        anomalies=_string_items(parsed.anomalies),
        limitations=_string_items(parsed.limitations),
    )
    sections: list[str] = []
    if chart.chart_type:
        sections.append(f"图表类型：{chart.chart_type}")
    if chart.trends:
        sections.append("主要趋势：\n" + "\n".join(f"- {item}" for item in chart.trends))
    if chart.peaks:
        sections.append("峰值与低谷：\n" + "\n".join(f"- {item}" for item in chart.peaks))
    if chart.anomalies:
        sections.append("显著异常：\n" + "\n".join(f"- {item}" for item in chart.anomalies))
    if chart.limitations:
        sections.append("图表限制：\n" + "\n".join(f"- {item}" for item in chart.limitations))
    if not sections:
        raise VisionResponseFormatError("视觉模型未给出可用的图表分析")
    return "\n\n".join(sections), chart, warnings


def normalize_table(content: str) -> tuple[ExtractedTable, list[str]]:
    """Validate and clamp an image-table extraction; never trust raw model values."""
    raw = extract_json_object(content)
    try:
        parsed = _TablePayload.model_validate(raw)
    except ValidationError as error:
        raise VisionResponseFormatError("视觉模型输出不符合表格提取契约") from error
    warnings = _unknown_field_warnings(
        raw,
        frozenset({"title", "unit", "columns", "rows", "merged_cell_info", "warnings"}),
    )
    warnings.extend(_string_items(parsed.warnings, limit=MAX_WARNINGS // 2))

    columns = [_text(item, limit=MAX_CELL_CHARS) for item in parsed.columns[:MAX_TABLE_COLUMNS]]
    if not columns:
        raise VisionResponseFormatError("视觉模型未给出表格列")
    if len(parsed.columns) > MAX_TABLE_COLUMNS:
        warnings.append(f"表格列数超过上限，仅保留前 {MAX_TABLE_COLUMNS} 列")

    rows: list[list[str | None]] = []
    truncated_rows = 0
    for raw_row in parsed.rows:
        if not isinstance(raw_row, list):
            warnings.append("存在无法识别的表格行，已忽略")
            continue
        if len(rows) >= MAX_TABLE_ROWS:
            truncated_rows += 1
            continue
        cells: list[str | None] = []
        for cell in raw_row[: len(columns)]:
            text = _text(cell, limit=MAX_CELL_CHARS)
            cells.append(text or None)
        cells.extend([None] * (len(columns) - len(cells)))
        if len(raw_row) > len(columns):
            warnings.append("存在超出列数的单元格，已忽略")
        rows.append(cells)
    if truncated_rows:
        warnings.append(f"表格行数超过上限，仅保留前 {MAX_TABLE_ROWS} 行")
    if not rows:
        raise VisionResponseFormatError("视觉模型未提取到表格行")

    empty_cells = sum(1 for row in rows for cell in row if cell is None)
    if empty_cells:
        warnings.append(f"存在 {empty_cells} 个无法确认的单元格，已保留空值")
    merged = _string_items(parsed.merged_cell_info)
    if merged:
        warnings.append("合并单元格为视觉推断结果，请对照原图核对")

    warnings.append(_REQUIRED_TABLE_WARNING)
    return (
        ExtractedTable(
            title=_text(parsed.title, limit=MAX_CELL_CHARS) or None,
            unit=_text(parsed.unit, limit=200) or None,
            columns=[item or f"column_{index + 1}" for index, item in enumerate(columns)],
            rows=rows,
            merged_cell_info=merged,
            warnings=list(dict.fromkeys(warnings))[:MAX_WARNINGS],
        ),
        [],
    )


__all__ = [
    "EVIDENCE_ID_PATTERN",
    "MAX_CELL_CHARS",
    "MAX_LIST_ITEMS",
    "MAX_QUESTION_CHARS",
    "MAX_TABLE_COLUMNS",
    "MAX_TABLE_ROWS",
    "MAX_TEXT_CHARS",
    "MAX_WARNINGS",
    "ChartAnalysis",
    "ExtractedTable",
    "VisionAnalysisRequest",
    "VisionAnalysisResult",
    "VisionSource",
    "VisionStatus",
    "VisionTaskType",
    "extract_json_object",
    "normalize_chart",
    "normalize_description",
    "normalize_table",
]
