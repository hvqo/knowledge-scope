# ruff: noqa: RUF001
"""Application-controlled agent loop for multimodal RAG questions.

The service never executes model-provided code.  The planner may only choose the
bounded retrieval, image-understanding, or table-computation tools, and all
citation markers are assigned by application code rather than by the model.

Reliability rules enforced here:

* Only markers that belong to a retrieval candidate may reach the client; any
  other ``[C...]`` token produced by the model is stripped before streaming.
* Citations are filtered to the candidates that were actually cited or used by
  a tool, instead of echoing every retrieval candidate.
* A degraded turn keeps the real number of executed tool calls and tells the
  user that the tool step did not complete.
* Conversation history is reused through the existing history compressor and
  query-rewrite service instead of a second memory implementation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from knowledge_scope.agent.evidence import (
    EvidenceAssessment,
    EvidenceRequirement,
    assess_evidence,
    build_supplementary_query,
    extract_requirement,
    visual_feature_requested,
)
from knowledge_scope.agent.planner import AgentPlanner
from knowledge_scope.agent.schemas import (
    AGENT_PROMPT_VERSION,
    AgentCandidate,
    AgentPlan,
    AgentPlanningError,
    AgentToolObservation,
    AgentToolOutcome,
)
from knowledge_scope.agent.tools import (
    AgentRetrievalTool,
    AgentToolExecutor,
    DocumentLoader,
)
from knowledge_scope.llm.errors import LLMError
from knowledge_scope.llm.gateway import LLMGateway
from knowledge_scope.llm.schemas import LLMMessage, LLMRequest
from knowledge_scope.rag.history import HistoryCompressor, HistoryPlan
from knowledge_scope.rag.rewrite import QueryRewriteService
from knowledge_scope.rag.schemas import (
    RAGAgentCandidateData,
    RAGAgentToolTraceData,
    RAGAnswerDeltaData,
    RAGCitation,
    RAGCitationsData,
    RAGCompleteData,
    RAGErrorData,
    RAGQueryRequest,
    RAGStreamEvent,
    RAGToolStatusData,
    RAGVisualSource,
)
from knowledge_scope.retrieval.unified import UnifiedRetrievalService
from knowledge_scope.shared.config import Settings
from knowledge_scope.table_analysis.service import TableAnalysisService
from knowledge_scope.vision.safety import (
    assess_visual_risk,
    build_gated_visual_answer,
    extract_numeric_claims,
)
from knowledge_scope.vision.service import VisionAnalysisService

logger = logging.getLogger(__name__)

_CITATION_TOKEN_PATTERN = re.compile(r"^C[1-9][0-9]*$")
_TRAILING_PARTIAL_MARKER_PATTERN = re.compile(r"\[C[0-9]*$")

_ANSWER_SYSTEM_PROMPT = (
    "你是 KnowledgeScope 的多模态知识库问答助手。你只能依据用户消息中给出的"
    "候选证据和工具结果回答，不得使用模型记忆补充事实。\n"
    "引用规则：\n"
    "1. 候选证据前的 [C1]、[C2] 等标记由系统分配，回答中提到来源时必须使用这些标记。\n"
    "2. 不得编造标记，也不得编造 Evidence ID、文档名、页码或来源块编号。\n"
    "3. 如果证据不足，请明确说明当前知识库中没有找到足够依据，不要猜测。\n"
    "4. 只引用真正支持当前论断的候选证据；与问题无关的候选不要引用。\n"
    "工具结果规则：\n"
    "5. 表格工具给出的是确定性计算结果；引用数值时保留原单位，并说明使用的公式或口径。\n"
    "   引用工具数值时必须保留工具输出的原始精度（例如 10.6977%），不要四舍五入或改写。\n"
    "6. 视觉模型结果仅供参考，须说明其来自视觉模型、可能需要人工核对；不要把它当作精确统计数据。\n"
    "7. 不要自行推算或计算数值。涉及增长率、增量、求和、平均等计算时，"
    "只能引用表格工具给出的结果；如果表格工具未成功执行，请明确说明无法计算，"
    "可以引用证据中的原始数值，但不得给出自行计算的结果。\n"
    "8. 不要输出代码块、SQL、脚本或任何可执行内容，只输出面向用户的简洁中文回答。"
)

_FALLBACK_NOTICE = (
    "（智能体未能完成必要的工具步骤：{reason}。以下为常规检索回答，"
    "未执行视觉分析或确定性计算；回答中若出现汇总、增长率等计算数值，"
    "均不是确定性工具结果，请以证据原文为准。）\n\n"
)


def _format_table_facts(outcome: AgentToolOutcome | None) -> list[str]:
    """Render deterministic table-tool results as trusted application text."""

    if outcome is None or outcome.status != "completed":
        return []
    payload = outcome.payload or {}
    if payload.get("status") != "ok":
        return []
    parts: list[str] = ["表格工具确定性计算结果（已核验）："]
    columns = [
        str(item.get("name", "")) for item in payload.get("columns") or [] if isinstance(item, dict)
    ]
    if columns:
        parts.append("列：" + " | ".join(name for name in columns if name))
    for row in (payload.get("rows") or [])[:6]:
        cells = row.get("cells") if isinstance(row, dict) else row
        if not isinstance(cells, list):
            continue
        parts.append("行：" + " | ".join("" if cell is None else str(cell) for cell in cells))
    if payload.get("formula"):
        parts.append("公式：" + str(payload["formula"]))
    units = [str(unit) for unit in payload.get("units") or [] if unit]
    if units:
        parts.append("单位：" + " | ".join(units))
    return ["\n".join(parts)] if len(parts) > 1 else []


def _verified_table_answer(outcome: AgentToolOutcome | None) -> str:
    facts = _format_table_facts(outcome)
    if not facts:
        return (
            "表格工具没有返回可用于回答的确定性结果，因此本轮不输出精确数值，请核对原始表格证据。"
        )
    return (
        "本轮回答中的部分数值无法由表格工具结果核验，为避免把未核验数字当作事实，"
        "下面只给出表格工具的确定性计算结果：\n\n" + "\n".join(facts)
    )


def _table_cell_label(value: Any) -> bool:
    """True when a cell is a textual row/indicator label, not a number."""

    text = "" if value is None else str(value).strip()
    if not text:
        return False
    return not extract_numeric_claims(text)


def _structured_column_unit(column: dict[str, Any]) -> str:
    """Return the unit that belongs to one concrete result column.

    Derived percentage columns must not inherit the input metric's unit (for
    example "10.6977 亿美元"); a relative change is a percentage while the
    absolute change of a percentage metric is expressed in percentage points.
    """

    name = str(column.get("name") or "").strip()
    kind = str(column.get("kind") or "").strip()
    unit = str(column.get("unit") or "").strip()
    if kind == "percent":
        if name.startswith("绝对"):
            return "百分点"
        return unit or "%"
    return unit


def _structured_table_facts(outcome: AgentToolOutcome | None) -> list[str]:
    """Render M2 structured results with explicit row, column, and unit binding.

    The application — not the model — decides how each value is presented, so a
    wrong field/value correspondence can never be published as verified text.
    """

    if outcome is None or outcome.status != "completed":
        return []
    payload = outcome.payload or {}
    if payload.get("status") != "ok":
        return []
    column_specs = [item for item in payload.get("columns") or [] if isinstance(item, dict)]
    legacy_units = [str(unit).strip() for unit in payload.get("units") or [] if str(unit).strip()]
    has_column_units = any(str(item.get("unit") or "").strip() for item in column_specs)
    facts: list[str] = []
    for row in (payload.get("rows") or [])[:10]:
        cells = row.get("cells") if isinstance(row, dict) else row
        if not isinstance(cells, list):
            continue
        labels = [str(cell).strip() for cell in cells if _table_cell_label(cell)]
        pairs: list[str] = []
        for index, cell in enumerate(cells):
            text = "" if cell is None else str(cell).strip()
            if not text or _table_cell_label(cell):
                continue
            column_spec = column_specs[index] if index < len(column_specs) else {}
            column = str(column_spec.get("name", "")).strip() or f"列 {index + 1}"
            unit = _structured_column_unit(column_spec)
            suffix = "%" if unit == "%" else f" {unit}" if unit else ""
            pairs.append(f"{column} = {text}{suffix}")
        if not pairs:
            continue
        prefix = " / ".join(labels) if labels else "结果"
        facts.append(f"{prefix}：" + "；".join(pairs) + "。")
    formula = str(payload.get("formula") or "").strip()
    if formula:
        facts.append(f"公式：{formula}")
    if legacy_units and not has_column_units:
        facts.append("单位：" + " | ".join(legacy_units))
    return facts


def _structured_table_answer(outcome: AgentToolOutcome | None) -> str:
    facts = _structured_table_facts(outcome)
    if not facts:
        return (
            "表格工具没有返回可用于回答的确定性结果，因此本轮不输出精确数值，请核对原始表格证据。"
        )
    return "根据表格工具的确定性计算结果（已核验，数值与行列、单位一一对应）：\n\n" + "\n".join(
        facts
    )


@dataclass(frozen=True)
class _VisualExecution:
    """One actual ``analyze_image`` attempt with its application parameters."""

    candidate: AgentCandidate
    outcome: AgentToolOutcome
    asset_index: int
    vision_task: str


def _last_completed_visual(
    executions: Sequence[_VisualExecution],
) -> _VisualExecution | None:
    for execution in reversed(executions):
        if execution.outcome.status == "completed":
            return execution
    return None


class _RAGStreamService(Protocol):
    def stream(self, request: RAGQueryRequest) -> AsyncIterator[RAGStreamEvent]: ...


class _AnswerMarkerFilter:
    """Stream-safe citation marker validator.

    Text is forwarded as soon as it is unambiguous.  A completed ``[C...]``
    token is forwarded only when it belongs to a retrieval candidate; every
    other citation-shaped token is dropped.  A short incomplete tail is held
    back so markers split across model deltas are still validated.
    """

    def __init__(self, allowed_markers: set[str]) -> None:
        self._allowed = allowed_markers
        self._buffer = ""
        self.used_markers: set[str] = set()
        self.stripped_markers = 0

    def feed(self, text: str) -> str:
        self._buffer += text
        output: list[str] = []
        while True:
            start = self._buffer.find("[")
            if start == -1:
                output.append(self._buffer)
                self._buffer = ""
                break
            output.append(self._buffer[:start])
            rest = self._buffer[start:]
            end = rest.find("]")
            if end == -1:
                if len(rest) > 48:
                    # A long fragment without a closing bracket is ordinary text,
                    # not a citation marker; forward it once.
                    output.append(rest[1:])
                    self._buffer = ""
                    continue
                self._buffer = rest
                break
            token = rest[1:end]
            if _CITATION_TOKEN_PATTERN.match(token):
                if token in self._allowed:
                    self.used_markers.add(token)
                    output.append(f"[{token}]")
                else:
                    self.stripped_markers += 1
                self._buffer = rest[end + 1 :]
                continue
            output.append(rest[: end + 1])
            self._buffer = rest[end + 1 :]
        return "".join(output)

    def flush(self) -> str:
        rest = self._buffer
        self._buffer = ""
        return _TRAILING_PARTIAL_MARKER_PATTERN.sub("", rest)


class AgentChatService:
    """Run one agentic turn: retrieve, optionally call one tool, then answer."""

    def __init__(
        self,
        settings: Settings,
        *,
        gateway: LLMGateway,
        unified_retrieval: UnifiedRetrievalService,
        vision_service: VisionAnalysisService,
        table_service: TableAnalysisService,
        rag_service: _RAGStreamService | None = None,
        planner: AgentPlanner | None = None,
        history_compressor: HistoryCompressor | None = None,
        query_rewrite: QueryRewriteService | None = None,
    ) -> None:
        self._settings = settings
        self._gateway = gateway
        self._unified_retrieval = unified_retrieval
        self._vision_service = vision_service
        self._table_service = table_service
        self._rag_service = rag_service
        self._planner = planner if planner is not None else AgentPlanner(gateway, settings)
        self._history = history_compressor or HistoryCompressor(gateway, settings)
        self._rewrite = query_rewrite or QueryRewriteService(gateway, settings)

    # -- overridable seams used by tests to avoid live model calls -------------
    def _build_retrieval_tool(self, document_loader: DocumentLoader) -> AgentRetrievalTool:
        return AgentRetrievalTool(
            self._settings,
            self._unified_retrieval,
            document_loader=document_loader,
            result_limit=self._settings.agent_retrieval_limit,
        )

    def _build_executor(self, document_loader: DocumentLoader) -> AgentToolExecutor:
        return AgentToolExecutor(
            self._settings,
            vision_service=self._vision_service,
            table_service=self._table_service,
            document_loader=document_loader,
        )

    async def stream(
        self,
        request: RAGQueryRequest,
        *,
        document_loader: DocumentLoader,
    ) -> AsyncIterator[RAGStreamEvent]:
        started = time.perf_counter()
        step = 0
        tool_calls = 0
        retrieval_latency_ms = 0.0
        retrieval_degraded = False
        branch_statuses: dict[str, str] | None = None
        observations: list[AgentToolObservation] = []
        trace: list[RAGAgentToolTraceData] = []
        candidates: tuple[AgentCandidate, ...] = ()
        used_candidates: set[int] = set()
        visual_executions: list[_VisualExecution] = []
        visual_outcome: AgentToolOutcome | None = None
        table_outcome: AgentToolOutcome | None = None
        table_candidates: set[int] = set()
        table_verified = False
        risk = assess_visual_risk(request.query)
        requirement = extract_requirement(request.query)
        assessment_before: EvidenceAssessment | None = None
        assessment_after: EvidenceAssessment | None = None
        supplementary_retrieval = False
        supplementary_query: str | None = None
        supplementary_added = 0
        supplementary_latency_ms: float | None = None
        supplementary_error: str | None = None
        auto_visual = False

        if request.knowledge_base_id is None:  # validated by the request schema
            yield _error_event("agent", "智能体问答需要指定知识库")
            yield _complete_error(started, tool_calls=0)
            return

        history_context, history_plan = await self._prepare_history(request)
        rewritten, rewrite_latency_ms = await self._rewrite_for_retrieval(request)
        retrieval_query = rewritten or request.query

        step += 1
        yield _tool_status("retrieval", "running", "正在检索知识库并整理候选来源", step)
        retrieval_tool = self._build_retrieval_tool(document_loader)
        try:
            retrieval = await retrieval_tool.retrieve(
                retrieval_query,
                request.knowledge_base_id,
                document_id=request.document_id,
            )
        except Exception as error:
            logger.warning(
                "agent retrieval failed (%s): %s",
                type(error).__name__,
                error,
            )
            trace.append(
                RAGAgentToolTraceData(
                    tool="retrieval",
                    status="failed",
                    summary="检索阶段失败，已降级为常规检索问答",
                    error_code="retrieval_failed",
                )
            )
            yield _tool_status("retrieval", "failed", "检索阶段失败，已降级为常规检索问答", step)
            if self._needs_visual_number_gate(risk, visual_outcome):
                async for event in self._safety_gated_events(
                    request,
                    risk=risk,
                    candidates=candidates,
                    trace=trace,
                    tool_calls=tool_calls,
                    visual_executions=[],
                    table_outcome=table_outcome,
                    table_candidates=table_candidates,
                    retrieval_latency_ms=retrieval_latency_ms,
                    retrieval_degraded=retrieval_degraded,
                    branch_statuses=branch_statuses,
                    started=started,
                ):
                    yield event
                return
            async for event in self._fallback_events(
                request,
                "检索阶段失败",
                started,
                tool_calls=tool_calls,
                trace=trace,
                retrieval_latency_ms=retrieval_latency_ms,
                history_plan=history_plan,
            ):
                yield event
            return

        candidates = retrieval.candidates
        retrieval_latency_ms = retrieval.retrieval_latency_ms
        retrieval_degraded = retrieval.degraded
        branch_statuses = retrieval.branch_statuses
        if not candidates:
            yield _tool_status("retrieval", "skipped", "未检索到可引用来源", step)
            trace.append(
                RAGAgentToolTraceData(
                    tool="retrieval",
                    status="skipped",
                    summary="未检索到可引用来源，已降级为常规检索问答",
                )
            )
            if self._needs_visual_number_gate(risk, visual_outcome):
                async for event in self._safety_gated_events(
                    request,
                    risk=risk,
                    candidates=candidates,
                    trace=trace,
                    tool_calls=tool_calls,
                    visual_executions=[],
                    table_outcome=table_outcome,
                    table_candidates=table_candidates,
                    retrieval_latency_ms=retrieval_latency_ms,
                    retrieval_degraded=retrieval_degraded,
                    branch_statuses=branch_statuses,
                    started=started,
                ):
                    yield event
                return
            async for event in self._fallback_events(
                request,
                "未检索到可引用来源",
                started,
                tool_calls=tool_calls,
                trace=trace,
                retrieval_latency_ms=retrieval_latency_ms,
                history_plan=history_plan,
            ):
                yield event
            return
        yield _tool_status(
            "retrieval",
            "completed",
            f"已检索到 {len(candidates)} 条候选来源",
            step,
        )
        trace.append(
            RAGAgentToolTraceData(
                tool="retrieval",
                status="completed",
                summary=f"已检索到 {len(candidates)} 条候选来源",
                parameters={
                    "candidates": len(candidates),
                    "rewritten_query": rewritten,
                    "degraded": retrieval_degraded,
                },
            )
        )

        if self._settings.agent_evidence_sufficiency_enabled:
            assessment_before = assess_evidence(requirement, candidates)
            if (
                not assessment_before.satisfied
                and self._settings.agent_supplementary_retrieval_max >= 1
            ):
                suppl_query = build_supplementary_query(requirement, assessment_before)
                if suppl_query.strip() and suppl_query.strip() != request.query.strip():
                    supplementary_query = suppl_query
                    step += 1
                    yield _tool_status(
                        "retrieval",
                        "running",
                        f"首次检索证据不足，执行补充检索：{suppl_query}",
                        step,
                    )
                    suppl_started = time.perf_counter()
                    try:
                        extra = await retrieval_tool.retrieve(
                            suppl_query,
                            request.knowledge_base_id,
                            document_id=request.document_id,
                        )
                        supplementary_latency_ms = (time.perf_counter() - suppl_started) * 1000.0
                        merged, added = _merge_candidates(
                            candidates,
                            extra.candidates,
                            limit=(
                                self._settings.agent_retrieval_limit
                                + self._settings.agent_supplementary_candidate_limit
                            ),
                        )
                        candidates = merged
                        supplementary_added = added
                        supplementary_retrieval = True
                        retrieval_latency_ms += extra.retrieval_latency_ms
                        assessment_after = assess_evidence(requirement, candidates)
                        trace.append(
                            RAGAgentToolTraceData(
                                tool="retrieval",
                                status="completed",
                                summary=f"补充检索新增 {added} 条候选来源",
                                parameters={
                                    "supplementary": True,
                                    "query": suppl_query,
                                    "added": added,
                                    "missing_before": list(assessment_before.missing)[:8],
                                },
                            )
                        )
                        yield _tool_status(
                            "retrieval",
                            "completed",
                            f"补充检索新增 {added} 条候选来源",
                            step,
                        )
                    except Exception as error:
                        supplementary_latency_ms = (time.perf_counter() - suppl_started) * 1000.0
                        supplementary_error = type(error).__name__
                        logger.warning(
                            "agent supplementary retrieval failed (%s)",
                            supplementary_error,
                        )
                        trace.append(
                            RAGAgentToolTraceData(
                                tool="retrieval",
                                status="failed",
                                summary="补充检索失败，保留初次检索候选",
                                error_code="supplementary_retrieval_failed",
                            )
                        )
                        yield _tool_status(
                            "retrieval",
                            "failed",
                            "补充检索失败，保留初次检索候选继续回答",
                            step,
                        )
        if assessment_after is None:
            assessment_after = assess_evidence(requirement, candidates)

        executor = self._build_executor(document_loader)
        used_tool_keys: set[tuple[object, ...]] = set()
        try:
            for _round in range(self._settings.agent_max_rounds):
                plan = await self._planner.plan(
                    request.query,
                    candidates,
                    observations,
                    round_number=tool_calls + 1,
                    history_context=history_context or None,
                )
                if not auto_visual and tool_calls < self._settings.agent_max_tool_calls:
                    wants_visual = requirement is not None and requirement.needs_image_feature
                    if plan.action == "final" and (tool_calls == 0 or wants_visual):
                        forced = _auto_visual_plan(
                            request.query,
                            candidates,
                            risk,
                            requirement=requirement,
                            tool_calls=tool_calls,
                        )
                        if forced is not None:
                            plan = forced
                            auto_visual = True
                if plan.action == "final":
                    break
                tool = plan.tool
                if tool is None:  # validator guarantees a tool; defensive only
                    break
                if (
                    tool == "analyze_table"
                    and not auto_visual
                    and requirement is not None
                    and requirement.needs_image_feature
                    and (requirement.comparison or _image_table_intent(request.query))
                    and tool_calls > 0
                ):
                    forced = _auto_visual_plan(
                        request.query,
                        candidates,
                        risk,
                        requirement=requirement,
                        tool_calls=tool_calls,
                    )
                    if forced is not None:
                        plan = forced
                        tool = plan.tool
                        auto_visual = True
                key = (
                    tool,
                    plan.evidence_index,
                    _effective_asset_index(candidates[plan.evidence_index - 1], plan)
                    if tool == "analyze_image"
                    else 0,
                    plan.vision_task if tool == "analyze_image" else None,
                    plan.question if tool == "analyze_image" else None,
                    plan.operation,
                    json.dumps(plan.parameters or {}, sort_keys=True, ensure_ascii=False)
                    if plan.parameters
                    else "",
                )
                if key in used_tool_keys or tool_calls >= self._settings.agent_max_tool_calls:
                    yield _tool_status(
                        tool,
                        "skipped",
                        "已达到工具调用预算，直接基于现有证据回答",
                        step,
                    )
                    trace.append(
                        RAGAgentToolTraceData(
                            tool=tool,
                            status="skipped",
                            summary="已达到工具调用预算，直接基于现有证据回答",
                        )
                    )
                    break
                used_tool_keys.add(key)
                candidate = candidates[plan.evidence_index - 1]
                second_candidate = (
                    candidates[plan.second_evidence_index - 1]
                    if tool == "analyze_table" and plan.second_evidence_index is not None
                    else None
                )
                step += 1
                yield _tool_status(
                    tool,
                    "running",
                    _tool_running_summary(tool, candidate),
                    step,
                )
                tool_calls += 1
                used_candidates.add(candidate.index)
                if second_candidate is not None:
                    used_candidates.add(second_candidate.index)
                outcome = await self._run_tool(
                    executor,
                    candidate,
                    plan,
                    second_candidate,
                )
                yield _tool_status(tool, outcome.status, outcome.summary, step)
                if outcome.tool == "analyze_image":
                    visual_executions.append(
                        _VisualExecution(
                            candidate=candidate,
                            outcome=outcome,
                            asset_index=_effective_asset_index(candidate, plan),
                            vision_task=plan.vision_task or "describe_image",
                        )
                    )
                    visual_outcome = outcome
                elif outcome.tool == "analyze_table" and outcome.status == "completed":
                    table_verified = True
                    table_outcome = outcome
                    table_candidates.add(candidate.index)
                    if second_candidate is not None:
                        table_candidates.add(second_candidate.index)
                tool_parameters = _trace_parameters(plan, candidate, second_candidate) or {}
                if auto_visual and outcome.tool == "analyze_image" and tool_calls == 1:
                    tool_parameters = {
                        **tool_parameters,
                        "auto_visual": True,
                    }
                trace.append(
                    RAGAgentToolTraceData(
                        tool=outcome.tool,
                        status=outcome.status,
                        summary=outcome.summary,
                        evidence_id=candidate.evidence_id,
                        document_id=candidate.citation.document_id,
                        parameters=tool_parameters or None,
                        fact_status=outcome.fact_status,
                        error_code=outcome.error_code,
                    )
                )
                observations.append(
                    AgentToolObservation(
                        tool=outcome.tool,
                        summary=outcome.summary,
                        payload=outcome.payload,
                    )
                )
        except AgentPlanningError:
            logger.warning("agent planning failed after %s tool calls", tool_calls)
            if self._needs_visual_number_gate(risk, visual_outcome):
                async for event in self._safety_gated_events(
                    request,
                    risk=risk,
                    candidates=candidates,
                    trace=trace,
                    tool_calls=tool_calls,
                    visual_executions=visual_executions,
                    table_outcome=table_outcome,
                    table_candidates=table_candidates,
                    retrieval_latency_ms=retrieval_latency_ms,
                    retrieval_degraded=retrieval_degraded,
                    branch_statuses=branch_statuses,
                    started=started,
                ):
                    yield event
                return
            async for event in self._fallback_events(
                request,
                "工具规划失败",
                started,
                tool_calls=tool_calls,
                trace=trace,
                retrieval_latency_ms=retrieval_latency_ms,
                history_plan=history_plan,
            ):
                yield event
            return

        if self._needs_visual_number_gate(risk, visual_outcome):
            async for event in self._safety_gated_events(
                request,
                risk=risk,
                candidates=candidates,
                trace=trace,
                tool_calls=tool_calls,
                visual_executions=visual_executions,
                table_outcome=table_outcome,
                table_candidates=table_candidates,
                retrieval_latency_ms=retrieval_latency_ms,
                retrieval_degraded=retrieval_degraded,
                branch_statuses=branch_statuses,
                started=started,
            ):
                yield event
            return

        async for event in self._answer_events(
            request,
            candidates=candidates,
            observations=observations,
            tool_calls=tool_calls,
            trace=trace,
            used_candidates=used_candidates,
            visual_executions=visual_executions,
            table_verified=table_verified,
            risk=risk,
            table_outcome=table_outcome,
            table_candidates=table_candidates,
            history_context=history_context,
            history_plan=history_plan,
            rewritten=rewritten,
            rewrite_latency_ms=rewrite_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            retrieval_degraded=retrieval_degraded,
            branch_statuses=branch_statuses,
            requirement=requirement,
            assessment_before=assessment_before,
            assessment_after=assessment_after,
            supplementary_retrieval=supplementary_retrieval,
            supplementary_query=supplementary_query,
            supplementary_added=supplementary_added,
            supplementary_latency_ms=supplementary_latency_ms,
            supplementary_error=supplementary_error,
            auto_visual=auto_visual,
            started=started,
        ):
            yield event

    @staticmethod
    def _needs_visual_number_gate(
        risk: object,
        visual_outcome: AgentToolOutcome | None,
    ) -> bool:
        """Decide whether the turn must not trust visual numbers.

        A successful table tool proves only its own deterministic calculation.
        It never upgrades an unrelated visual-number claim, so the gate cannot
        be skipped merely because a table tool succeeded.
        """

        if not bool(getattr(risk, "high_risk", False)):
            return False
        if visual_outcome is not None:
            return True
        return bool(getattr(risk, "visual_intent", False))

    async def _safety_gated_events(
        self,
        request: RAGQueryRequest,
        *,
        risk: object,
        candidates: tuple[AgentCandidate, ...],
        trace: list[RAGAgentToolTraceData],
        tool_calls: int,
        visual_executions: Sequence[_VisualExecution],
        retrieval_latency_ms: float,
        retrieval_degraded: bool,
        branch_statuses: dict[str, str] | None,
        started: float,
        table_outcome: AgentToolOutcome | None = None,
        table_candidates: set[int] | None = None,
    ) -> AsyncIterator[RAGStreamEvent]:
        """Return a deterministic, number-free answer instead of trusting the VLM.

        The normal streaming answer is never started for this turn, so an
        unverified visual number cannot leak through ``answer_delta``.
        """

        completed_visual = _last_completed_visual(visual_executions)
        vision_completed = completed_visual is not None
        description: str | None = None
        if completed_visual is not None:
            payload = completed_visual.outcome.payload or {}
            answer = payload.get("answer")
            if isinstance(answer, str):
                description = answer
        verified_facts = _structured_table_facts(table_outcome)
        answer, notice = build_gated_visual_answer(
            description=description,
            vision_completed=vision_completed,
            verified_facts=verified_facts,
        )
        used = set(table_candidates or ())
        for execution in visual_executions:
            used.add(execution.candidate.index)
        citations = _select_citations(
            candidates,
            used_markers=set(),
            used_candidates=used,
        )
        if citations:
            yield RAGStreamEvent(
                event="citations",
                data=RAGCitationsData(
                    prompt_version=AGENT_PROMPT_VERSION,
                    items=citations,
                ).model_dump(mode="json"),
            )
        yield _delta_event(answer)
        has_table_facts = bool(verified_facts)
        if vision_completed and has_table_facts:
            fact_status = "unverified"
            gate = "visual_numeric_withheld"
            notice = "表格部分为确定性计算结果（已核验）；图片中的数值未经核验，已省略精确数字。"
        elif completed_visual is not None:
            fact_status = completed_visual.outcome.fact_status or "not_available"
            gate = "visual_numeric_withheld"
        elif has_table_facts:
            fact_status = "source_verified"
            gate = "vision_unavailable"
            notice = "视觉分析未能完成，已阻止未核验的数值输出；表格计算结果已核验"
        else:
            fact_status = "not_available"
            gate = "vision_unavailable"
        yield RAGStreamEvent(
            event="complete",
            data=RAGCompleteData(
                status="completed",
                prompt_version=AGENT_PROMPT_VERSION,
                provider=None,
                model=None,
                retrieval_mode="unified",
                route="unified",
                answer_mode="agent",
                tool_calls=tool_calls,
                agent_candidates=_candidate_records(candidates),
                agent_tool_trace=trace or None,
                agent_degraded=False,
                fact_status=fact_status,
                safety_gate=gate,
                safety_notice=notice,
                visual_sources=_visual_sources(visual_executions),
                rewritten_query=None,
                retrieval_cached=False,
                retrieval_escalated=False,
                memories_used=len(request.memories) if request.memories else 0,
                retrieval_degraded=retrieval_degraded,
                retrieval_branch_statuses=branch_statuses,
                finish_reason="safety_gate",
                retrieval_latency_ms=retrieval_latency_ms,
                llm_latency_ms=0.0,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            ).model_dump(mode="json"),
        )

    async def _prepare_history(
        self,
        request: RAGQueryRequest,
    ) -> tuple[str, HistoryPlan | None]:
        history = tuple(request.history)
        if not history:
            return "", None
        try:
            plan = await self._history.prepare(history)
        except Exception as error:  # compression is an optimization
            logger.debug("agent history preparation skipped (%s)", type(error).__name__)
            return "", None
        return _render_history(plan), plan

    async def _rewrite_for_retrieval(
        self,
        request: RAGQueryRequest,
    ) -> tuple[str | None, float | None]:
        history = tuple(request.history)
        if not history:
            return None, None
        started = time.perf_counter()
        try:
            rewritten = await self._rewrite.rewrite(request.query, history)
        except Exception as error:  # rewrite is an optimization
            logger.debug("agent query rewrite skipped (%s)", type(error).__name__)
            return None, None
        latency = (time.perf_counter() - started) * 1000.0
        return rewritten, latency

    async def _run_tool(
        self,
        executor: AgentToolExecutor,
        candidate: AgentCandidate,
        plan: AgentPlan,
        second_candidate: AgentCandidate | None,
    ) -> AgentToolOutcome:
        if plan.tool == "analyze_image":
            effective_asset_index = _effective_asset_index(candidate, plan)
            if plan.asset_index != effective_asset_index:
                plan = plan.model_copy(update={"asset_index": effective_asset_index})
            return await executor.analyze_image(candidate, plan)
        if plan.tool == "analyze_table":
            return await executor.analyze_table(candidate, plan, second_candidate)
        return AgentToolOutcome(
            tool="retrieval",
            status="skipped",
            summary="不支持的智能体工具",
            payload={"status": "skipped"},
            error_code="unsupported_tool",
            error_message="不支持的智能体工具",
        )

    async def _deterministic_table_events(
        self,
        request: RAGQueryRequest,
        *,
        candidates: tuple[AgentCandidate, ...],
        tool_calls: int,
        trace: list[RAGAgentToolTraceData],
        used_candidates: set[int],
        table_outcome: AgentToolOutcome,
        table_candidates: set[int],
        history_plan: HistoryPlan | None,
        rewritten: str | None,
        rewrite_latency_ms: float | None,
        retrieval_latency_ms: float,
        retrieval_degraded: bool,
        branch_statuses: dict[str, str] | None,
        started: float,
    ) -> AsyncIterator[RAGStreamEvent]:
        """Publish the M2 structured result directly, without model paraphrasing.

        High-risk statistical questions are answered from the table tool output
        itself. The model never gets a chance to attach the right number to the
        wrong country/metric/unit, so no numeric-field correspondence has to be
        inferred from free-form text.
        """

        answer = _structured_table_answer(table_outcome)
        yield _delta_event(answer)
        citations = _select_citations(
            candidates,
            used_markers=set(),
            used_candidates=used_candidates | table_candidates,
        )
        yield RAGStreamEvent(
            event="citations",
            data=RAGCitationsData(
                prompt_version=AGENT_PROMPT_VERSION,
                items=citations,
            ).model_dump(mode="json"),
        )
        history_fields = _history_fields(history_plan)
        llm_latency_ms = 0.0
        yield RAGStreamEvent(
            event="complete",
            data=RAGCompleteData(
                status="completed",
                prompt_version=AGENT_PROMPT_VERSION,
                provider=None,
                model=None,
                retrieval_mode="unified",
                route="unified",
                answer_mode="agent",
                tool_calls=tool_calls,
                agent_candidates=_candidate_records(candidates),
                agent_tool_trace=trace or None,
                agent_degraded=False,
                fact_status="source_verified",
                safety_gate="none",
                safety_notice="回答中的数值来自表格工具的确定性计算结果，未经模型改写。",
                visual_sources=[],
                rewritten_query=rewritten,
                retrieval_cached=False,
                retrieval_escalated=False,
                rewrite_latency_ms=rewrite_latency_ms,
                history_compressed_turns=history_fields[0],
                history_cache_hit=history_fields[1],
                history_summary_used=history_fields[2],
                memories_used=len(request.memories) if request.memories else 0,
                retrieval_degraded=retrieval_degraded,
                retrieval_branch_statuses=branch_statuses,
                finish_reason="deterministic_table",
                retrieval_latency_ms=retrieval_latency_ms,
                llm_latency_ms=llm_latency_ms,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            ).model_dump(mode="json"),
        )

    async def _answer_events(
        self,
        request: RAGQueryRequest,
        *,
        candidates: tuple[AgentCandidate, ...],
        observations: list[AgentToolObservation],
        tool_calls: int,
        trace: list[RAGAgentToolTraceData],
        used_candidates: set[int],
        visual_executions: Sequence[_VisualExecution],
        table_verified: bool,
        risk: object,
        table_outcome: AgentToolOutcome | None,
        table_candidates: set[int],
        history_context: str,
        history_plan: HistoryPlan | None,
        rewritten: str | None,
        rewrite_latency_ms: float | None,
        retrieval_latency_ms: float,
        retrieval_degraded: bool,
        branch_statuses: dict[str, str] | None,
        requirement: EvidenceRequirement | None = None,
        assessment_before: EvidenceAssessment | None = None,
        assessment_after: EvidenceAssessment | None = None,
        supplementary_retrieval: bool = False,
        supplementary_query: str | None = None,
        supplementary_added: int = 0,
        supplementary_latency_ms: float | None = None,
        supplementary_error: str | None = None,
        auto_visual: bool = False,
        started: float,
    ) -> AsyncIterator[RAGStreamEvent]:
        if (
            bool(getattr(risk, "high_risk", False))
            and table_outcome is not None
            and not visual_executions
        ):
            async for event in self._deterministic_table_events(
                request,
                candidates=candidates,
                tool_calls=tool_calls,
                trace=trace,
                used_candidates=used_candidates,
                table_outcome=table_outcome,
                table_candidates=table_candidates,
                history_plan=history_plan,
                rewritten=rewritten,
                rewrite_latency_ms=rewrite_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                retrieval_degraded=retrieval_degraded,
                branch_statuses=branch_statuses,
                started=started,
            ):
                yield event
            return
        messages = [
            LLMMessage(role="system", content=_ANSWER_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=_answer_prompt(
                    request.query,
                    candidates,
                    observations,
                    history_context,
                    evidence_notes=_evidence_answer_notes(
                        assessment_before,
                        assessment_after,
                        supplementary_retrieval=supplementary_retrieval,
                        supplementary_error=supplementary_error,
                    ),
                ),
            ),
        ]
        allowed_markers = {candidate.marker for candidate in candidates}
        marker_filter = _AnswerMarkerFilter(allowed_markers)
        provider: str | None = None
        model: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        finish_reason: str | None = None
        emitted = False
        llm_started = time.perf_counter()
        try:
            streaming = self._gateway.stream(
                LLMRequest(
                    messages=messages,
                    task_type="agent",
                    temperature=0.0,
                    max_tokens=self._settings.rag_max_tokens,
                    reasoning="disabled",
                )
            )
            async with aclosing(streaming) as gateway_stream:
                async for event in gateway_stream:
                    provider = event.provider
                    model = event.model
                    input_tokens = event.input_tokens
                    output_tokens = event.output_tokens
                    finish_reason = event.finish_reason
                    if not event.delta:
                        continue
                    safe_delta = marker_filter.feed(event.delta)
                    if safe_delta:
                        emitted = True
                        yield _delta_event(safe_delta)
        except LLMError:
            yield _error_event("llm", "回答生成失败，请稍后重试")
            yield _complete_error(
                started,
                tool_calls=tool_calls,
                retrieval_latency_ms=retrieval_latency_ms,
                trace=trace,
                candidates=candidates,
            )
            return
        tail = marker_filter.flush()
        if tail:
            emitted = True
            yield _delta_event(tail)
        llm_latency_ms = (time.perf_counter() - llm_started) * 1000.0
        if not emitted:
            yield _error_event("llm", "回答生成结果为空，请稍后重试")
            yield _complete_error(
                started,
                tool_calls=tool_calls,
                retrieval_latency_ms=retrieval_latency_ms,
                trace=trace,
                candidates=candidates,
            )
            return
        if marker_filter.stripped_markers:
            logger.info(
                "agent stripped %s invalid citation markers",
                marker_filter.stripped_markers,
            )
        citations = _select_citations(
            candidates,
            used_markers=marker_filter.used_markers,
            used_candidates=used_candidates,
        )
        yield RAGStreamEvent(
            event="citations",
            data=RAGCitationsData(
                prompt_version=AGENT_PROMPT_VERSION,
                items=citations,
            ).model_dump(mode="json"),
        )
        history_fields = _history_fields(history_plan)
        if table_verified and not visual_executions:
            fact_status = "source_verified"
            safety_gate = "none"
            safety_notice = None
        elif visual_executions:
            completed_visual = _last_completed_visual(visual_executions)
            fact_status = (
                completed_visual.outcome.fact_status
                if completed_visual is not None and completed_visual.outcome.fact_status
                else "not_available"
            )
            safety_gate = "visual_unverified"
            safety_notice = (
                "回答包含视觉分析或模型归纳的数值，需人工核对，不能作为精确统计数据使用。"
            )
        else:
            fact_status = "not_available"
            safety_gate = "none"
            safety_notice = None
        yield RAGStreamEvent(
            event="complete",
            data=RAGCompleteData(
                status="completed",
                prompt_version=AGENT_PROMPT_VERSION,
                provider=provider,
                model=model,
                retrieval_mode="unified",
                route="unified",
                answer_mode="agent",
                tool_calls=tool_calls,
                agent_candidates=_candidate_records(candidates),
                agent_tool_trace=trace or None,
                agent_degraded=False,
                fact_status=fact_status,
                safety_gate=safety_gate,
                safety_notice=safety_notice,
                visual_sources=_visual_sources(visual_executions),
                rewritten_query=rewritten,
                retrieval_cached=False,
                retrieval_escalated=False,
                rewrite_latency_ms=rewrite_latency_ms,
                history_compressed_turns=history_fields[0],
                history_cache_hit=history_fields[1],
                history_summary_used=history_fields[2],
                memories_used=len(request.memories) if request.memories else 0,
                retrieval_degraded=retrieval_degraded,
                retrieval_branch_statuses=branch_statuses,
                agent_evidence_required=_requirement_labels(requirement),
                agent_evidence_missing=(
                    list(assessment_after.missing) if assessment_after is not None else None
                ),
                agent_evidence_satisfied=(
                    assessment_after.satisfied if assessment_after is not None else None
                ),
                agent_evidence_coverage_before=(
                    assessment_before.coverage_summary() if assessment_before is not None else None
                ),
                agent_evidence_coverage_after=(
                    assessment_after.coverage_summary() if assessment_after is not None else None
                ),
                agent_evidence_conflicts=(
                    list(assessment_after.conflicts)
                    if assessment_after is not None and assessment_after.conflicts
                    else None
                ),
                agent_supplementary_retrieval=supplementary_retrieval,
                agent_supplementary_query=supplementary_query,
                agent_supplementary_added=supplementary_added,
                agent_supplementary_latency_ms=supplementary_latency_ms,
                agent_supplementary_error=supplementary_error,
                agent_visual_auto_called=auto_visual,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                retrieval_latency_ms=retrieval_latency_ms,
                llm_latency_ms=llm_latency_ms,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            ).model_dump(mode="json"),
        )

    async def _fallback_events(
        self,
        request: RAGQueryRequest,
        reason: str,
        started: float,
        *,
        tool_calls: int,
        trace: list[RAGAgentToolTraceData],
        retrieval_latency_ms: float,
        history_plan: HistoryPlan | None,
    ) -> AsyncIterator[RAGStreamEvent]:
        yield _delta_event(
            _FALLBACK_NOTICE.format(reason=reason),
        )
        if self._rag_service is None:
            yield _error_event("agent", "智能体问答暂不可用，请切换标准问答")
            yield _complete_error(
                started,
                tool_calls=tool_calls,
                retrieval_latency_ms=retrieval_latency_ms,
                trace=trace,
            )
            return
        async for event in self._rag_service.stream(request):
            if event.event == "complete":
                data = dict(event.data)
                data["answer_mode"] = "agent"
                # Never reset the real tool count: a degraded turn must keep the
                # number of tools that actually ran.
                data["tool_calls"] = tool_calls
                data["agent_degraded"] = True
                data["agent_degraded_reason"] = reason[:200]
                data["agent_tool_trace"] = [item.model_dump(mode="json") for item in trace] or None
                data["fact_status"] = data.get("fact_status") or "not_available"
                data.setdefault("safety_gate", "none")
                yield RAGStreamEvent(event="complete", data=data)
            else:
                yield event


def _trace_parameters(
    plan: AgentPlan,
    candidate: AgentCandidate | None = None,
    second_candidate: AgentCandidate | None = None,
) -> dict[str, Any] | None:
    parameters: dict[str, Any] = {}
    if plan.operation is not None:
        parameters["operation"] = plan.operation
    if plan.tool == "analyze_image":
        # Record the exact asset the executor will read, including index 0.
        parameters["vision_task"] = plan.vision_task or "describe_image"
        parameters["asset_index"] = (
            _effective_asset_index(candidate, plan) if candidate is not None else plan.asset_index
        )
    elif plan.vision_task is not None:
        parameters["vision_task"] = plan.vision_task
    if plan.question is not None:
        parameters["question"] = plan.question
    if plan.second_evidence_index is not None:
        parameters["second_evidence_index"] = plan.second_evidence_index
    if plan.parameters is not None:
        parameters["parameters"] = plan.parameters
    if candidate is not None:
        parameters["document_id"] = str(candidate.citation.document_id)
        parameters["evidence_id"] = candidate.evidence_id
    if second_candidate is not None:
        parameters["second_document_id"] = str(second_candidate.citation.document_id)
        parameters["second_evidence_id"] = second_candidate.evidence_id
    return parameters or None


def _visual_sources(
    executions: Sequence[_VisualExecution],
) -> list[RAGVisualSource]:
    """Expose every image the analyze_image tool actually attempted.

    The identifiers and the asset index come from the executed plan and M1
    validated lineage, never from model output. Duplicate assets are collapsed,
    but distinct assets of the same Evidence stay separate and keep execution
    order. Failed attempts are listed with ``execution_status="failed"`` so the
    UI never presents them as successful analyses.
    """

    sources: list[RAGVisualSource] = []
    seen: set[tuple[str, str, int]] = set()
    for execution in executions:
        candidate = execution.candidate
        outcome = execution.outcome
        knowledge_base_id = candidate.citation.knowledge_base_id
        evidence_id = candidate.evidence_id
        if knowledge_base_id is None or evidence_id is None:
            continue
        raw_sources = (outcome.payload or {}).get("sources")
        page_start = candidate.citation.page_start
        page_end = candidate.citation.page_end
        source_block_ids = list(candidate.citation.source_block_ids)[:64]
        for raw in raw_sources if isinstance(raw_sources, list) else []:
            if not isinstance(raw, dict):
                continue
            try:
                raw_kb = UUID(str(raw["knowledge_base_id"]))
                raw_document = UUID(str(raw["document_id"]))
                raw_evidence = str(raw["evidence_id"])
                page_start = int(raw["page_start"])
                page_end = int(raw["page_end"])
                source_block_ids = [str(item) for item in raw.get("source_block_ids") or []][:64]
            except (KeyError, TypeError, ValueError):
                continue
            knowledge_base_id = raw_kb
            document_id = raw_document
            evidence_id = raw_evidence
            break
        else:
            document_id = candidate.citation.document_id
        key = (str(document_id), evidence_id, execution.asset_index)
        if key in seen:
            continue
        seen.add(key)
        completed = outcome.status == "completed"
        try:
            sources.append(
                RAGVisualSource(
                    knowledge_base_id=knowledge_base_id,
                    document_id=document_id,
                    evidence_id=evidence_id,
                    page_start=page_start,
                    page_end=page_end,
                    source_block_ids=source_block_ids,
                    asset_index=execution.asset_index,
                    label=f"第 {page_start} 页图片",
                    fact_status=(
                        outcome.fact_status
                        if completed and outcome.fact_status
                        else "not_available"
                    ),
                    execution_status="completed" if completed else "failed",
                    vision_task=execution.vision_task,
                )
            )
        except ValueError:
            continue
    return sources


def _candidate_records(
    candidates: tuple[AgentCandidate, ...],
) -> list[RAGAgentCandidateData] | None:
    if not candidates:
        return None
    return [
        RAGAgentCandidateData(
            marker=candidate.marker,
            document_id=candidate.citation.document_id,
            evidence_id=candidate.evidence_id,
            modality=candidate.modality,
            page_start=candidate.citation.page_start,
            page_end=candidate.citation.page_end,
        )
        for candidate in candidates
    ]


def _select_citations(
    candidates: tuple[AgentCandidate, ...],
    *,
    used_markers: set[str],
    used_candidates: set[int],
) -> list[RAGCitation]:
    """Return only sources that were cited or actually consumed by a tool."""

    selected = [
        candidate.citation
        for candidate in candidates
        if candidate.marker in used_markers or candidate.index in used_candidates
    ]
    return selected


def _history_fields(
    history_plan: HistoryPlan | None,
) -> tuple[int | None, bool | None, bool | None]:
    if history_plan is None:
        return None, None, None
    return (
        history_plan.compressed_turns,
        history_plan.cache_hit,
        history_plan.summary is not None,
    )


def _render_history(history_plan: HistoryPlan) -> str:
    lines: list[str] = []
    if history_plan.summary:
        lines.append(f"较早对话摘要：{history_plan.summary}")
    for turn in history_plan.recent_turns:
        speaker = "用户" if turn.role == "user" else "助手"
        content = " ".join(turn.content.split())
        lines.append(f"{speaker}：{content[:500]}")
    return "\n".join(lines)


def _tool_running_summary(tool: str, candidate: AgentCandidate) -> str:
    if tool == "analyze_image":
        return f"正在调用视觉模型解读 {candidate.marker} 的图片素材"
    return f"正在对 {candidate.marker} 执行确定性表格计算"


def _answer_prompt(
    question: str,
    candidates: tuple[AgentCandidate, ...],
    observations: list[AgentToolObservation],
    history_context: str,
    *,
    evidence_notes: Sequence[str] = (),
) -> str:
    observation_text = _observations_text(observations) if observations else "（本轮未调用工具。）"
    lines: list[str] = []
    if history_context:
        lines.extend(
            [
                "对话历史（用于理解指代，不代表新的证据）：",
                history_context,
                "",
            ]
        )
    lines.extend([f"用户问题：{question}", "", "候选证据（标记由系统分配）："])
    for candidate in candidates:
        citation = candidate.citation
        section = " / ".join(citation.section_path) if citation.section_path else ""
        line = (
            f"[{citation.marker}] {candidate.document_title or citation.document_id}"
            f" 第{citation.page_start}页"
        )
        if citation.page_end != citation.page_start:
            line += f"-{citation.page_end}页"
        if citation.modality:
            line += f" modality={citation.modality}"
        if section:
            line += f" 章节={section}"
        lines.append(line)
        source_text = (citation.snippet or "").strip()
        if source_text:
            lines.append(source_text)
        lines.append("")
    if evidence_notes:
        lines.extend(
            [
                "证据需求与缺口（应用层检查结果，必须遵守）：",
                *[f"- {note}" for note in evidence_notes],
                "",
            ]
        )
    lines.extend(
        [
            "工具结果：",
            observation_text,
            "",
            "请只引用真正支持回答的候选证据标记；如果某个工具未成功执行，"
            "请明确说明该能力不可用，不要给出自行计算的结果。关键证据仍然不足时，"
            "请明确说明缺少哪个对象、年份或指标，不要用部分证据拼凑完整比较结论。",
        ]
    )
    return "\n".join(lines)


def _observations_text(observations: list[AgentToolObservation]) -> str:
    parts: list[str] = []
    per_observation = 2_000
    for observation in observations:
        payload = json.dumps(
            observation.payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        if len(payload) > per_observation:
            payload = payload[: per_observation - 1] + "…"
        parts.append(f"- {observation.tool}: {observation.summary}\n  {payload}")
    return "\n".join(parts)


def _delta_event(text: str) -> RAGStreamEvent:
    return RAGStreamEvent(
        event="answer_delta",
        data=RAGAnswerDeltaData(text=text).model_dump(mode="json"),
    )


def _tool_status(tool: str, status: str, summary: str, step: int) -> RAGStreamEvent:
    return RAGStreamEvent(
        event="tool_status",
        data=RAGToolStatusData(
            tool=tool,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            summary=summary[:200],
            step=min(step, 12),
        ).model_dump(mode="json"),
    )


def _error_event(category: str, message: str) -> RAGStreamEvent:
    return RAGStreamEvent(
        event="error",
        data=RAGErrorData(category=category, message=message).model_dump(mode="json"),
    )


def _complete_error(
    started: float,
    *,
    tool_calls: int,
    retrieval_latency_ms: float | None = None,
    trace: list[RAGAgentToolTraceData] | None = None,
    candidates: tuple[AgentCandidate, ...] | None = None,
) -> RAGStreamEvent:
    return RAGStreamEvent(
        event="complete",
        data=RAGCompleteData(
            status="error",
            prompt_version=AGENT_PROMPT_VERSION,
            retrieval_mode="unified",
            route="unified",
            answer_mode="agent",
            tool_calls=tool_calls,
            agent_candidates=_candidate_records(candidates or ()),
            agent_tool_trace=trace or None,
            retrieval_latency_ms=retrieval_latency_ms,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        ).model_dump(mode="json"),
    )


_CHART_FEATURE = re.compile(
    r"趋势|走势|峰值|低谷|柱状|折线|曲线|图表|变化|波动|上升|下降|最高|最低|分布"
)


def _candidate_identity(candidate: AgentCandidate) -> tuple[str, str, str, int]:
    citation = candidate.citation
    identity = str(citation.chunk_id or candidate.evidence_id or candidate.marker)
    return (
        str(citation.document_id),
        identity,
        str(citation.page_start),
        candidate.asset_index if candidate.asset_index is not None else -1,
    )


def _renumber_candidates(
    candidates: Sequence[AgentCandidate],
) -> tuple[AgentCandidate, ...]:
    renumbered: list[AgentCandidate] = []
    for index, candidate in enumerate(candidates, start=1):
        marker = f"C{index}"
        citation = candidate.citation.model_copy(update={"marker": marker})
        renumbered.append(
            AgentCandidate(
                index=index,
                marker=marker,
                citation=citation,
                document_title=candidate.document_title,
                table_columns=candidate.table_columns,
                table_rows=candidate.table_rows,
                table_issue=candidate.table_issue,
                asset_index=candidate.asset_index,
            )
        )
    return tuple(renumbered)


def _merge_candidates(
    initial: Sequence[AgentCandidate],
    extra: Sequence[AgentCandidate],
    *,
    limit: int,
) -> tuple[tuple[AgentCandidate, ...], int]:
    capacity = max(limit, 1)
    merged = list(initial[:capacity])
    seen = {_candidate_identity(candidate) for candidate in merged}
    added = 0
    for candidate in extra:
        if len(merged) >= capacity:
            break
        identity = _candidate_identity(candidate)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(candidate)
        added += 1
    return _renumber_candidates(merged), added


def _requirement_labels(
    requirement: EvidenceRequirement | None,
) -> list[str] | None:
    if requirement is None or requirement.empty:
        return None
    labels: list[str] = []
    labels.extend(requirement.objects)
    labels.extend(str(year) for year in requirement.years)
    labels.extend(requirement.metrics)
    if requirement.needs_image_feature:
        labels.append("视觉特征")
    elif requirement.needs_image:
        labels.append("图片")
    if requirement.needs_table:
        labels.append("表格")
    if requirement.comparison:
        labels.append("跨对象比较")
    return labels or None


def _evidence_answer_notes(
    before: EvidenceAssessment | None,
    after: EvidenceAssessment | None,
    *,
    supplementary_retrieval: bool,
    supplementary_error: str | None,
) -> tuple[str, ...]:
    notes: list[str] = []
    if before is not None and before.missing:
        notes.append("首次检索后存在证据缺口：" + "；".join(before.missing[:6]))
    if before is not None and before.conflicts:
        notes.extend(before.conflicts)
    if supplementary_retrieval:
        notes.append("应用层已执行一次补充检索，并合并去重候选证据。")
    if after is not None and after.missing:
        notes.append(
            "补充检索后仍缺少："
            + "；".join(after.missing[:6])
            + "。若无法补齐，请明确说明缺少什么，不要拼凑完整比较或精确计算。"
        )
    if supplementary_error:
        notes.append("补充检索失败，仅使用首次检索证据回答。")
    return tuple(notes)


def _effective_asset_index(candidate: AgentCandidate, plan: AgentPlan) -> int:
    if candidate.asset_index is not None:
        return candidate.asset_index
    return plan.asset_index


def _image_table_intent(query: str) -> bool:
    return bool(re.search(r"图片\s*表格|图像\s*表格|扫描\s*(?:表|件|图片)|扫描表", query))


def _auto_visual_plan(
    query: str,
    candidates: tuple[AgentCandidate, ...],
    risk: object,
    *,
    requirement: EvidenceRequirement | None = None,
    tool_calls: int = 0,
) -> AgentPlan | None:
    """Call the image tool when a question needs native pixels.

    The planner still owns table and text decisions. This guard only prevents a
    "final" plan from skipping the image analysis that the question requires.
    """
    image_table = _image_table_intent(query)
    comparison_after_tools = bool(
        requirement is not None
        and requirement.needs_image_feature
        and requirement.comparison
        and len(requirement.objects) >= 2
    )
    planned_after_tools = bool((image_table or comparison_after_tools) and tool_calls > 0)
    wants_feature = visual_feature_requested(query) or image_table or planned_after_tools
    if not wants_feature:
        return None
    if (
        not image_table
        and not planned_after_tools
        and (tool_calls > 0 or not bool(getattr(risk, "visual_intent", False)))
    ):
        return None
    image_candidates = [
        candidate
        for candidate in candidates
        if candidate.modality == "image" and candidate.tool_capable
    ]
    if not image_candidates:
        return None
    target = image_candidates[0]
    extracted_table = bool(
        requirement is not None
        and (
            requirement.needs_table
            or re.search(r"图片\s*表格|图像\s*表格|扫描\s*(?:表|件|图片)", query)
        )
    )
    if extracted_table:
        vision_task = "extract_table"
    elif _CHART_FEATURE.search(query):
        vision_task = "analyze_chart"
    else:
        vision_task = "describe_image"
    return AgentPlan(
        action="call_tool",
        reason="问题要求原生视觉特征分析，应用层自动调用图片分析",
        tool="analyze_image",
        evidence_index=target.index,
        asset_index=target.asset_index if target.asset_index is not None else 0,
        vision_task=vision_task,
        question=query[:500],
    )


__all__ = ["AgentChatService"]
