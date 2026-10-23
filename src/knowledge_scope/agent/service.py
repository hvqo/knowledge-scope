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
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any, Protocol

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
)
from knowledge_scope.retrieval.unified import UnifiedRetrievalService
from knowledge_scope.shared.config import Settings
from knowledge_scope.table_analysis.service import TableAnalysisService
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

        if request.knowledge_base_id is None:  # validated by the request schema
            yield _error_event("agent", "智能体问答需要指定知识库")
            yield _complete_error(started, tool_calls=0)
            return

        history_context, history_plan = await self._prepare_history(request)
        rewritten, rewrite_latency_ms = await self._rewrite_for_retrieval(request)
        retrieval_query = rewritten or request.query

        step += 1
        yield _tool_status("retrieval", "running", "正在检索知识库并整理候选来源", step)
        try:
            retrieval = await self._build_retrieval_tool(document_loader).retrieve(
                retrieval_query,
                request.knowledge_base_id,
            )
        except Exception as error:
            logger.warning("agent retrieval failed (%s)", type(error).__name__)
            trace.append(
                RAGAgentToolTraceData(
                    tool="retrieval",
                    status="failed",
                    summary="检索阶段失败，已降级为常规检索问答",
                    error_code="retrieval_failed",
                )
            )
            yield _tool_status("retrieval", "failed", "检索阶段失败，已降级为常规检索问答", step)
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

        executor = self._build_executor(document_loader)
        used_tool_keys: set[tuple[str, int]] = set()
        try:
            for _round in range(self._settings.agent_max_rounds):
                plan = await self._planner.plan(
                    request.query,
                    candidates,
                    observations,
                    round_number=tool_calls + 1,
                    history_context=history_context or None,
                )
                if plan.action == "final":
                    break
                tool = plan.tool
                if tool is None:  # validator guarantees a tool; defensive only
                    break
                key = (tool, plan.evidence_index)
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
                trace.append(
                    RAGAgentToolTraceData(
                        tool=outcome.tool,
                        status=outcome.status,
                        summary=outcome.summary,
                        evidence_id=candidate.evidence_id,
                        document_id=candidate.citation.document_id,
                        parameters=_trace_parameters(plan),
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

        async for event in self._answer_events(
            request,
            candidates=candidates,
            observations=observations,
            tool_calls=tool_calls,
            trace=trace,
            used_candidates=used_candidates,
            history_context=history_context,
            history_plan=history_plan,
            rewritten=rewritten,
            rewrite_latency_ms=rewrite_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            retrieval_degraded=retrieval_degraded,
            branch_statuses=branch_statuses,
            started=started,
        ):
            yield event

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

    async def _answer_events(
        self,
        request: RAGQueryRequest,
        *,
        candidates: tuple[AgentCandidate, ...],
        observations: list[AgentToolObservation],
        tool_calls: int,
        trace: list[RAGAgentToolTraceData],
        used_candidates: set[int],
        history_context: str,
        history_plan: HistoryPlan | None,
        rewritten: str | None,
        rewrite_latency_ms: float | None,
        retrieval_latency_ms: float,
        retrieval_degraded: bool,
        branch_statuses: dict[str, str] | None,
        started: float,
    ) -> AsyncIterator[RAGStreamEvent]:
        messages = [
            LLMMessage(role="system", content=_ANSWER_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=_answer_prompt(
                    request.query,
                    candidates,
                    observations,
                    history_context,
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
                yield RAGStreamEvent(event="complete", data=data)
            else:
                yield event


def _trace_parameters(plan: AgentPlan) -> dict[str, Any] | None:
    parameters: dict[str, Any] = {}
    if plan.operation is not None:
        parameters["operation"] = plan.operation
    if plan.vision_task is not None:
        parameters["vision_task"] = plan.vision_task
    if plan.asset_index:
        parameters["asset_index"] = plan.asset_index
    if plan.question is not None:
        parameters["question"] = plan.question
    if plan.second_evidence_index is not None:
        parameters["second_evidence_index"] = plan.second_evidence_index
    if plan.parameters is not None:
        parameters["parameters"] = plan.parameters
    return parameters or None


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
    lines.extend(
        [
            "工具结果：",
            observation_text,
            "",
            "请只引用真正支持回答的候选证据标记；如果某个工具未成功执行，"
            "请明确说明该能力不可用，不要给出自行计算的结果。",
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


__all__ = ["AgentChatService"]
