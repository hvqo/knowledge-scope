# ruff: noqa: RUF001
"""Deterministic retrieval and tool execution for the multimodal agent router.

The agent never receives raw filesystem access, arbitrary code execution, or a
database session.  Retrieval candidates are produced by the existing unified
retrieval service, and every tool call is routed through the already hardened
M2/M3 services that validate the knowledge base, document, Evidence, and asset.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from knowledge_scope.agent.schemas import (
    AgentCandidate,
    AgentPlan,
    AgentToolOutcome,
)
from knowledge_scope.documents.assets import AssetAccessError, get_evidence
from knowledge_scope.documents.models import Document
from knowledge_scope.evidence.models import MultimodalEvidence
from knowledge_scope.rag.context import citation_for_unified
from knowledge_scope.retrieval.unified import UnifiedRetrievalService
from knowledge_scope.shared.config import Settings
from knowledge_scope.table_analysis.models import (
    TableAccessError,
    TableAnalysisError,
    TableAnalysisRequest,
)
from knowledge_scope.table_analysis.parsing import (
    ParsedTableData,
    detect_unit_hint,
    parse_table_content,
)
from knowledge_scope.table_analysis.service import TableAnalysisService
from knowledge_scope.vision.errors import VisionError
from knowledge_scope.vision.service import VisionAnalysisService

DocumentLoader = Callable[[UUID, UUID], Awaitable[Document | None]]

_TABLE_REPRESENTATION_TYPES = ("html", "markdown")
_TABLE_CAPTION_TYPES = ("caption",)
_TABLE_CONTEXT_TYPES = ("context",)


@dataclass(frozen=True, slots=True)
class AgentRetrievalResult:
    """One bounded retrieval pass exposed to the agent planner and answer prompt."""

    candidates: tuple[AgentCandidate, ...]
    context_text: str
    retrieval_latency_ms: float
    degraded: bool
    branch_statuses: dict[str, str]


def _first_content(evidence: MultimodalEvidence, representation_types: Sequence[str]) -> str | None:
    for representation in evidence.representations:
        if (
            representation.representation_type in representation_types
            and representation.content is not None
            and representation.searchable
        ):
            return representation.content
    return None


def parse_evidence_table(evidence: MultimodalEvidence) -> ParsedTableData:
    """Parse one table Evidence deterministically using its stored representation."""
    content: str | None = None
    representation_type: str | None = None
    for content_type in _TABLE_REPRESENTATION_TYPES:
        candidate = _first_content(evidence, (content_type,))
        if candidate:
            content = candidate
            representation_type = content_type
            break
    if content is None or representation_type is None:
        raise TableAccessError("table_representation_missing", "表格证据缺少可计算的结构化内容")
    caption = _first_content(evidence, _TABLE_CAPTION_TYPES)
    context = _first_content(evidence, _TABLE_CONTEXT_TYPES)
    return parse_table_content(
        content,
        representation_type,  # type: ignore[arg-type]
        caption=caption,
        unit_hint=detect_unit_hint(caption, context),
    )


def _table_column_headers(parsed: ParsedTableData) -> tuple[str, ...]:
    headers: list[str] = []
    for column in parsed.columns:
        header = column.header.strip()
        if header and header not in headers:
            headers.append(header)
    return tuple(headers)


def _clip(text: str, limit: int) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: max(limit - 1, 0)] + "…"


class AgentRetrievalTool:
    """Wrap the existing unified retrieval service for one agent turn."""

    def __init__(
        self,
        settings: Settings,
        unified_retrieval: UnifiedRetrievalService,
        *,
        document_loader: DocumentLoader,
        result_limit: int,
    ) -> None:
        self._settings = settings
        self._unified = unified_retrieval
        self._document_loader = document_loader
        self._result_limit = max(1, min(result_limit, 20))

    async def retrieve(self, query: str, knowledge_base_id: UUID) -> AgentRetrievalResult:
        result = await self._unified.search(
            query,
            knowledge_base_id,
            failure_mode="degraded",
        )
        items = list(result.items[: self._result_limit])
        titles: dict[UUID, str | None] = {}
        documents: dict[UUID, Document | None] = {}
        for item in items:
            if item.document_id not in documents:
                document = await self._document_loader(knowledge_base_id, item.document_id)
                documents[item.document_id] = document
                titles[item.document_id] = (
                    document.original_filename if document is not None else None
                )

        candidates: list[AgentCandidate] = []
        context_parts: list[str] = []
        remaining = self._settings.agent_context_budget_chars
        for index, item in enumerate(items, start=1):
            citation = citation_for_unified(item, f"C{index}")
            columns: tuple[str, ...] = ()
            row_labels: tuple[str, ...] = ()
            issue: str | None = None
            if item.candidate_kind == "evidence" and item.modality == "table":
                document = documents.get(item.document_id)
                evidence_id = item.evidence_id
                if document is None or evidence_id is None:
                    issue = "document_missing"
                else:
                    try:
                        snapshot = get_evidence(self._settings, document, evidence_id)
                        parsed = parse_evidence_table(snapshot.evidence)
                        columns = _table_column_headers(parsed)
                        row_labels = tuple(label for label in parsed.row_labels if label)[:8]
                    except AssetAccessError as error:
                        issue = error.code
                    except TableAnalysisError as error:
                        issue = error.code
                    except Exception:
                        issue = "table_unavailable"
            candidates.append(
                AgentCandidate(
                    index=index,
                    marker=citation.marker,
                    citation=citation,
                    document_title=titles.get(item.document_id),
                    table_columns=columns,
                    table_rows=row_labels,
                    table_issue=issue,
                )
            )
            if remaining <= 0:
                continue
            source_text = (item.source_text or item.rerank_text or "").strip()
            if not source_text:
                continue
            page = str(item.page_start)
            if item.page_end != item.page_start:
                page = f"{page}-{item.page_end}"
            title = titles.get(item.document_id) or str(item.document_id)
            block = (
                f"[{citation.marker}] {title} 第{page}页 "
                f"modality={item.modality or item.candidate_kind}\n{source_text}"
            )
            block = _clip(block, remaining)
            context_parts.append(block)
            remaining -= len(block)

        return AgentRetrievalResult(
            candidates=tuple(candidates),
            context_text="\n\n".join(context_parts),
            retrieval_latency_ms=result.total_latency_ms,
            degraded=result.degraded,
            branch_statuses={branch.branch: branch.status for branch in result.branches},
        )


class AgentToolExecutor:
    """Execute one planner tool call through the existing M2/M3 services."""

    def __init__(
        self,
        settings: Settings,
        *,
        vision_service: VisionAnalysisService,
        table_service: TableAnalysisService,
        document_loader: DocumentLoader,
    ) -> None:
        self._settings = settings
        self._vision_service = vision_service
        self._table_service = table_service
        self._document_loader = document_loader

    async def _load_document(self, candidate: AgentCandidate) -> Document:
        document = await self._document_loader(
            candidate.citation.knowledge_base_id or UUID(int=0),
            candidate.citation.document_id,
        )
        if document is None:
            raise _ToolFailure("document_missing", "候选来源所属文档不可用")
        return document

    async def analyze_image(
        self,
        candidate: AgentCandidate,
        plan: AgentPlan,
    ) -> AgentToolOutcome:
        evidence_id = candidate.evidence_id
        if evidence_id is None:
            return _failure("analyze_image", "evidence_missing", "候选来源缺少 Evidence 标识")
        try:
            document = await self._load_document(candidate)
            result = await self._vision_service.analyze(
                document,
                evidence_id=evidence_id,
                asset_index=plan.asset_index,
                task_type=plan.vision_task or "describe_image",
                question=(plan.question or "").strip() or None,
            )
        except _ToolFailure as error:
            return _failure("analyze_image", error.code, error.message)
        except VisionError as error:
            return _failure("analyze_image", error.code, error.message)
        except AssetAccessError as error:
            return _failure("analyze_image", error.code, error.message)
        except Exception:
            return _failure("analyze_image", "vision_tool_failed", "视觉工具执行失败")
        payload = result.model_dump(mode="json")
        if result.status in {"ok", "needs_review"}:
            label = (
                "图片理解"
                if result.task_type == "describe_image"
                else ("图表分析" if result.task_type == "analyze_chart" else "图片表格提取")
            )
            summary = f"{label}完成（{result.model or '视觉模型'}）"
            if result.status == "needs_review":
                summary += "，结果需人工核对"
            return AgentToolOutcome(
                tool="analyze_image",
                status="completed",
                summary=summary,
                payload=payload,
                fact_status=result.fact_status,
                numeric_claims=[claim.model_dump(mode="json") for claim in result.numeric_claims],
            )
        return AgentToolOutcome(
            tool="analyze_image",
            status="failed",
            summary=result.message or "视觉工具未返回可用结果",
            payload=payload,
            error_code=result.error_code or "vision_tool_failed",
            error_message=result.message,
            fact_status=result.fact_status,
            numeric_claims=[claim.model_dump(mode="json") for claim in result.numeric_claims],
        )

    async def analyze_table(
        self,
        candidate: AgentCandidate,
        plan: AgentPlan,
        second_candidate: AgentCandidate | None = None,
    ) -> AgentToolOutcome:
        evidence_id = candidate.evidence_id
        if evidence_id is None or plan.operation is None:
            return _failure("analyze_table", "invalid_plan", "表格工具缺少必要的证据或操作参数")
        knowledge_base_id = candidate.citation.knowledge_base_id
        if knowledge_base_id is None:
            return _failure("analyze_table", "invalid_plan", "候选来源缺少知识库标识")
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "knowledge_base_id": str(knowledge_base_id),
            "operation": plan.operation,
            "table": {
                "document_id": str(candidate.citation.document_id),
                "evidence_id": evidence_id,
            },
            "parameters": plan.parameters or {},
        }
        second_document: Document | None = None
        if plan.operation == "compare":
            if second_candidate is None or second_candidate.evidence_id is None:
                return _failure("analyze_table", "invalid_plan", "跨表对比缺少第二条表格证据")
            payload["compare_table"] = {
                "document_id": str(second_candidate.citation.document_id),
                "evidence_id": second_candidate.evidence_id,
            }
        try:
            request = TableAnalysisRequest.model_validate(payload)
        except ValidationError:
            return _failure(
                "analyze_table",
                "invalid_parameters",
                "表格工具参数不合法，未执行计算",
            )
        try:
            document = await self._load_document(candidate)
            if plan.operation == "compare":
                assert second_candidate is not None  # validated above
                second_document = await self._load_document(second_candidate)
            result = self._table_service.analyze(
                request,
                document=document,
                compare_document=second_document,
            )
        except _ToolFailure as error:
            return _failure("analyze_table", error.code, error.message)
        except TableAnalysisError as error:
            return _failure("analyze_table", error.code, error.message)
        except Exception:
            return _failure("analyze_table", "table_tool_failed", "表格工具执行失败")
        result_payload = result.model_dump(mode="json")
        if result.status == "ok":
            return AgentToolOutcome(
                tool="analyze_table",
                status="completed",
                summary=f"表格计算完成（{plan.operation}）",
                payload=result_payload,
                fact_status="source_verified",
            )
        return AgentToolOutcome(
            tool="analyze_table",
            status="failed",
            summary=result.message or "表格数据不可计算",
            payload=result_payload,
            error_code=result.error_code or "table_not_computable",
            error_message=result.message,
        )


class _ToolFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _failure(tool: str, code: str, message: str) -> AgentToolOutcome:
    return AgentToolOutcome(
        tool=tool,
        status="failed",
        summary=message,
        payload={"status": "failed", "error_code": code, "message": message},
        error_code=code,
        error_message=message,
    )


__all__ = [
    "AgentRetrievalResult",
    "AgentRetrievalTool",
    "AgentToolExecutor",
    "DocumentLoader",
    "parse_evidence_table",
]
