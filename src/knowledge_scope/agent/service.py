# ruff: noqa: RUF001
"""Application-controlled agent loop for multimodal RAG questions.

The service never executes model-provided code.  The planner may only choose the
bounded retrieval, image-understanding, or table-computation tools, and all
citation markers are assigned by application code rather than by the model.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Protocol

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
from knowledge_scope.rag.schemas import (
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

_ANSWER_SYSTEM_PROMPT = (
    "你是 KnowledgeScope 的多模态知识库问答助手。你只能依据用户消息中给出的"
    "候选证据和工具结果回答，不得使用模型记忆补充事实。\n"
    "引用规则：\n"
    "1. 候选证据前的 [C1]、[C2] 等标记由系统分配，回答中提到来源时必须使用这些标记。\n"
    "2. 不得编造标记，也不得编造 Evidence ID、文档名、页码或来源块编号。\n"
    "3. 如果证据不足，请明确说明当前知识库中没有找到足够依据，不要猜测。\n"
    "工具结果规则：\n"
    "4. 表格工具给出的是确定性计算结果；引用数值时保留原单位，并说明使用的公式或口径。\n"
    "5. 视觉模型结果仅供参考，须说明其来自视觉模型、可能需要人工核对；不要把它当作精确统计数据。\n"
    "6. 不要自行推算或计算数值。涉及增长率、增量、求和、平均等计算时，"
    "只能引用表格工具给出的结果；如果表格工具未成功执行，请明确说明无法计算，"
    "可以引用证据中的原始数值，但不得给出自行计算的结果。\n"
    "7. 不要输出代码块、SQL、脚本或任何可执行内容，只输出面向用户的简洁中文回答。"
)


class _RAGStreamService(Protocol):
    def stream(self, request: RAGQueryRequest) -> AsyncIterator[RAGStreamEvent]: ...


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
    ) -> None:
        self._settings = settings
        self._gateway = gateway
        self._unified_retrieval = unified_retrieval
        self._vision_service = vision_service
        self._table_service = table_service
        self._rag_service = rag_service
        self._planner = planner if planner is not None else AgentPlanner(gateway, settings)

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
        candidates: tuple[AgentCandidate, ...] = ()

        if request.knowledge_base_id is None:  # validated by the request schema
            yield _error_event("agent", "智能体问答需要指定知识库")
            yield _complete_error(started, tool_calls=0)
            return

        step += 1
        yield _tool_status("retrieval", "running", "正在检索知识库并整理候选来源", step)
        try:
            retrieval = await self._build_retrieval_tool(document_loader).retrieve(
                request.query,
                request.knowledge_base_id,
            )
        except Exception:
            async for event in self._fallback_events(
                request,
                "检索阶段失败，已降级为常规检索问答",
                started,
                step,
            ):
                yield event
            return

        candidates = retrieval.candidates
        retrieval_latency_ms = retrieval.retrieval_latency_ms
        retrieval_degraded = retrieval.degraded
        branch_statuses = retrieval.branch_statuses
        if not candidates:
            async for event in self._fallback_events(
                request,
                "未检索到可引用来源，已降级为常规检索问答",
                started,
                step,
            ):
                yield event
            return
        yield _tool_status(
            "retrieval",
            "completed",
            f"已检索到 {len(candidates)} 条候选来源",
            step,
        )

        executor = self._build_executor(document_loader)
        used_tool_keys: set[tuple[str, int]] = set()
        plan: AgentPlan | None = None
        try:
            for _round in range(self._settings.agent_max_rounds):
                plan = await self._planner.plan(
                    request.query,
                    candidates,
                    observations,
                    round_number=tool_calls + 1,
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
                    plan = None
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
                outcome = await self._run_tool(
                    executor,
                    candidate,
                    plan,
                    second_candidate,
                )
                yield _tool_status(tool, outcome.status, outcome.summary, step)
                observations.append(
                    AgentToolObservation(
                        tool=outcome.tool,
                        summary=outcome.summary,
                        payload=outcome.payload,
                    )
                )
                plan = None
        except AgentPlanningError:
            async for event in self._fallback_events(
                request,
                "工具规划失败，已降级为常规检索问答",
                started,
                step,
            ):
                yield event
            return

        async for event in self._answer_events(
            request,
            candidates=candidates,
            observations=observations,
            tool_calls=tool_calls,
            retrieval_latency_ms=retrieval_latency_ms,
            retrieval_degraded=retrieval_degraded,
            branch_statuses=branch_statuses,
            started=started,
        ):
            yield event

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
        retrieval_latency_ms: float,
        retrieval_degraded: bool,
        branch_statuses: dict[str, str] | None,
        started: float,
    ) -> AsyncIterator[RAGStreamEvent]:
        messages = [
            LLMMessage(role="system", content=_ANSWER_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=_answer_prompt(request.query, candidates, observations),
            ),
        ]
        citations: list[RAGCitation] = [candidate.citation for candidate in candidates]
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
                    emitted = True
                    yield _delta_event(event.delta)
        except LLMError:
            yield _error_event("llm", "回答生成失败，请稍后重试")
            yield _complete_error(
                started,
                tool_calls=tool_calls,
                retrieval_latency_ms=retrieval_latency_ms,
            )
            return
        llm_latency_ms = (time.perf_counter() - llm_started) * 1000.0
        if not emitted:
            yield _error_event("llm", "回答生成结果为空，请稍后重试")
            yield _complete_error(
                started,
                tool_calls=tool_calls,
                retrieval_latency_ms=retrieval_latency_ms,
            )
            return
        yield RAGStreamEvent(
            event="citations",
            data=RAGCitationsData(
                prompt_version=AGENT_PROMPT_VERSION,
                items=citations,
            ).model_dump(mode="json"),
        )
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
                rewritten_query=None,
                retrieval_cached=False,
                retrieval_escalated=False,
                rewrite_latency_ms=None,
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
        message: str,
        started: float,
        step: int,
    ) -> AsyncIterator[RAGStreamEvent]:
        yield _tool_status("retrieval", "skipped", message, step)
        if self._rag_service is None:
            yield _error_event("agent", "智能体问答暂不可用，请切换标准问答")
            yield _complete_error(started, tool_calls=0)
            return
        async for event in self._rag_service.stream(request):
            if event.event == "complete":
                data = dict(event.data)
                data["answer_mode"] = "agent"
                data["tool_calls"] = 0
                yield RAGStreamEvent(event="complete", data=data)
            else:
                yield event


def _tool_running_summary(tool: str, candidate: AgentCandidate) -> str:
    if tool == "analyze_image":
        return f"正在调用视觉模型解读 {candidate.marker} 的图片素材"
    return f"正在对 {candidate.marker} 执行确定性表格计算"


def _answer_prompt(
    question: str,
    candidates: tuple[AgentCandidate, ...],
    observations: list[AgentToolObservation],
) -> str:
    observation_text = _observations_text(observations) if observations else "（本轮未调用工具。）"
    lines: list[str] = [f"用户问题：{question}", "", "候选证据（标记由系统分配）："]
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
    lines.extend(["工具结果：", observation_text])
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
            retrieval_latency_ms=retrieval_latency_ms,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        ).model_dump(mode="json"),
    )


__all__ = ["AgentChatService"]
