"""Provider-backed report generation previews.

This module deliberately stops at an unpersisted preview.  The report API owns
source selection and maps model-visible ``R1``-style markers back to persisted
``ReportSourceReference`` rows; model output never supplies database IDs.
"""

# Chinese prompt text is intentionally part of the provider contract.
# ruff: noqa: RUF001

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError

from knowledge_scope.llm.errors import LLMError
from knowledge_scope.llm.schemas import LLMMessage, LLMRequest, LLMResponseFormat, LLMResult
from knowledge_scope.reports.models import Report, ReportSection, ReportSourceReference
from knowledge_scope.reports.schemas import (
    ReportAICitation,
    ReportAIDraftResponse,
    ReportAIOutlineItem,
    ReportAIOutlineResponse,
)
from knowledge_scope.shared.config import Settings

REPORT_GENERATION_PROMPT_VERSION = "report-generation-v1"
REPORT_GENERATION_MAX_TOKENS = 2_048
REPORT_GENERATION_MAX_CONTEXT_CHARS = 24_000
REPORT_GENERATION_MAX_SECTION_CHARS = 12_000
REPORT_GENERATION_MAX_INSTRUCTION_CHARS = 2_000
_CITATION_MARKER = re.compile(r"\[R([0-9]+)\]")
_CITATION_PREFIX = re.compile(r"\[R")


def stable_source_marker(source_id: UUID | str) -> str:
    """Return a bounded marker that survives preview acceptance and export."""
    compact_id = str(source_id).replace("-", "").upper()
    return f"S-{compact_id[:12]}"


class ReportGenerationGateway(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResult:
        """Complete one request through the application LLM gateway."""


class ReportGenerationError(Exception):
    """Safe, bounded failure category for a report-generation preview."""

    def __init__(self, category: Literal["provider", "invalid_output", "context_too_large"]):
        super().__init__(category)
        self.category = category


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class _OutlineWireItem(_WireModel):
    title: StrictStr = Field(min_length=1, max_length=200)
    summary: StrictStr = Field(min_length=1, max_length=1_000)


class _OutlineWire(_WireModel):
    items: list[_OutlineWireItem] = Field(min_length=1, max_length=12)


class _DraftWire(_WireModel):
    content: StrictStr = Field(min_length=1, max_length=100_000)
    citations: list[StrictStr] = Field(default_factory=list, max_length=12)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _decode_wire(text: str, model: type[_WireModel]) -> _WireModel:
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        if not isinstance(value, dict):
            raise ValueError("top-level JSON value must be an object")
        return model.model_validate(value)
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as error:
        raise ReportGenerationError("invalid_output") from error


def _citation_keys(value: str) -> list[str]:
    """Parse every citation-looking marker before resolving it."""
    matches = list(_CITATION_MARKER.finditer(value))
    valid_starts = {match.start() for match in matches}
    if any(match.start() not in valid_starts for match in _CITATION_PREFIX.finditer(value)):
        raise ReportGenerationError("invalid_output")
    return [f"R{match.group(1)}" for match in matches]


def _clip(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized if len(normalized) <= limit else normalized[:limit] + "…"


def _json_safe(value: object) -> object:
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def _source_context(reference: ReportSourceReference, key: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "key": key,
        "kind": reference.source_kind,
        "title": _clip(reference.title, 255),
        "document_title": _clip(reference.document_title, 512),
        "evidence_type": reference.evidence_type,
        "page_start": reference.page_start,
        "page_end": reference.page_end,
    }
    if reference.source_kind == "rag_evidence":
        payload["snippet"] = _clip(reference.snippet, 1_800)
        payload["section_path"] = list(reference.section_path or [])[:16]
    else:
        payload.update(
            {
                "question": _clip(reference.question, 1_800),
                "answer": _clip(reference.answer, 2_400),
                "columns": [
                    {
                        "name": _clip(str(column.get("name", "")), 255),
                        "data_type": _clip(str(column.get("data_type", "")), 255),
                    }
                    for column in (reference.columns or [])[:24]
                    if isinstance(column, dict)
                ],
                "rows": [
                    [_json_safe(cell) for cell in row[:24]] for row in (reference.rows or [])[:8]
                ],
                "row_count": reference.row_count,
            }
        )
    return payload


def _prompt_context(
    report: Report,
    section: ReportSection | None,
    references: Sequence[ReportSourceReference],
    *,
    operation: str,
    instruction: str | None,
    topic: str | None = None,
) -> str:
    context: dict[str, object] = {
        "operation": operation,
        "report": {"title": report.title, "topic": _clip(topic, 200)},
        "section": (
            None
            if section is None
            else {
                "title": section.title,
                "content": _clip(section.content, REPORT_GENERATION_MAX_SECTION_CHARS),
            }
        ),
        "instruction": _clip(instruction, REPORT_GENERATION_MAX_INSTRUCTION_CHARS),
        "sources": [
            _source_context(reference, f"R{index}") for index, reference in enumerate(references, 1)
        ],
    }
    serialized = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(serialized.encode("utf-8")) > REPORT_GENERATION_MAX_CONTEXT_CHARS:
        raise ReportGenerationError("context_too_large")
    return serialized


_OUTLINE_SYSTEM_PROMPT = f"""你是报告大纲助手。提示版本：{REPORT_GENERATION_PROMPT_VERSION}。
只根据用户提供的报告信息和来源生成大纲，不补造事实。必须只输出一个 JSON 对象，形状严格为
{{"items":[{{"title":"章节标题","summary":"章节目的"}}]}}。
不得输出 Markdown、代码围栏、解释、引用标记、SQL 或其他字段。章节数量为 1 到 12。
"""

_DRAFT_SYSTEM_PROMPT = f"""你是报告写作助手。提示版本：{REPORT_GENERATION_PROMPT_VERSION}。
只根据输入的报告、章节和来源写作；没有来源支持的事实不要加入。必须只输出一个 JSON 对象，形状严格为
{{"content":"正文，可在有依据处使用 [R1] 等标记","citations":["R1"]}}。
citations 中只能使用输入来源的 key，正文中的每个引用标记都必须列在 citations 中。
不得输出 Markdown 代码围栏、解释、SQL、provider 信息或其他字段。不要输出来源 key 以外的内部 ID。
"""


class ReportGenerationService:
    """Generate bounded previews from persisted report state without mutation."""

    def __init__(self, gateway: ReportGenerationGateway, settings: Settings) -> None:
        self._gateway = gateway
        self._settings = settings

    async def _complete(self, system_prompt: str, context: str) -> LLMResult:
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=context),
            ],
            task_type="report_generation",
            temperature=0.2,
            max_tokens=REPORT_GENERATION_MAX_TOKENS,
            model=self._settings.llm_model,
            response_format=LLMResponseFormat(type="json_object"),
            reasoning="disabled",
        )
        try:
            return await self._gateway.complete(request)
        except LLMError as error:
            raise ReportGenerationError("provider") from error

    async def generate_outline(
        self,
        report: Report,
        references: Sequence[ReportSourceReference],
        *,
        topic: str | None,
        instruction: str | None,
    ) -> ReportAIOutlineResponse:
        context = _prompt_context(
            report,
            None,
            references,
            operation="outline",
            instruction=instruction,
            topic=topic,
        )
        result = await self._complete(_OUTLINE_SYSTEM_PROMPT, context)
        wire = _decode_wire(result.text, _OutlineWire)
        assert isinstance(wire, _OutlineWire)
        try:
            items = [ReportAIOutlineItem.model_validate(item.model_dump()) for item in wire.items]
        except ValidationError as error:
            raise ReportGenerationError("invalid_output") from error
        if any(_citation_keys(item.title) or _citation_keys(item.summary) for item in items):
            raise ReportGenerationError("invalid_output")
        return ReportAIOutlineResponse(items=items)

    async def _draft(
        self,
        report: Report,
        section: ReportSection,
        references: Sequence[ReportSourceReference],
        *,
        operation: Literal["generate", "rewrite", "expand", "summarize"],
        instruction: str | None,
    ) -> ReportAIDraftResponse:
        context = _prompt_context(
            report,
            section,
            references,
            operation=operation,
            instruction=instruction,
        )
        result = await self._complete(_DRAFT_SYSTEM_PROMPT, context)
        wire = _decode_wire(result.text, _DraftWire)
        assert isinstance(wire, _DraftWire)
        reference_by_key = {f"R{index}": reference for index, reference in enumerate(references, 1)}
        markers = _citation_keys(wire.content)
        keys = list(wire.citations)
        if len(set(keys)) != len(keys) or set(keys) != set(markers):
            raise ReportGenerationError("invalid_output")
        if any(key not in reference_by_key for key in keys):
            raise ReportGenerationError("invalid_output")
        stable_content = wire.content.strip()
        for key, reference in reference_by_key.items():
            stable_content = stable_content.replace(
                f"[{key}]",
                f"[{stable_source_marker(reference.id)}]",
            )
        citations = [
            ReportAICitation(
                marker=stable_source_marker(reference_by_key[key].id),
                title=reference_by_key[key].title,
                page_start=reference_by_key[key].page_start,
                page_end=reference_by_key[key].page_end,
            )
            for key in keys
        ]
        return ReportAIDraftResponse(
            operation=operation,
            content=stable_content,
            citations=citations,
        )

    async def generate_section(
        self,
        report: Report,
        section: ReportSection,
        references: Sequence[ReportSourceReference],
        *,
        instruction: str | None,
    ) -> ReportAIDraftResponse:
        return await self._draft(
            report,
            section,
            references,
            operation="generate",
            instruction=instruction,
        )

    async def edit_section(
        self,
        report: Report,
        section: ReportSection,
        references: Sequence[ReportSourceReference],
        *,
        operation: Literal["rewrite", "expand", "summarize"],
        instruction: str | None,
    ) -> ReportAIDraftResponse:
        return await self._draft(
            report,
            section,
            references,
            operation=operation,
            instruction=instruction,
        )


__all__ = [
    "REPORT_GENERATION_MAX_CONTEXT_CHARS",
    "REPORT_GENERATION_MAX_TOKENS",
    "REPORT_GENERATION_PROMPT_VERSION",
    "ReportGenerationError",
    "ReportGenerationService",
    "stable_source_marker",
]
