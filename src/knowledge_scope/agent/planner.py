# ruff: noqa: RUF001, RUF003
"""Bounded planner that maps a question and retrieval candidates to one tool plan."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from pydantic import ValidationError

from knowledge_scope.agent.schemas import (
    AgentCandidate,
    AgentPlan,
    AgentPlanningError,
    AgentToolObservation,
    parse_agent_plan,
)
from knowledge_scope.llm.errors import LLMError
from knowledge_scope.llm.gateway import LLMGateway
from knowledge_scope.llm.schemas import (
    LLMMessage,
    LLMRequest,
    LLMResponseFormat,
)
from knowledge_scope.shared.config import Settings
from knowledge_scope.table_analysis.models import (
    TableAnalysisRequest,
    TableOperation,
)

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM_PROMPT = """你是 KnowledgeScope 的多模态检索分析助手。
你的任务不是直接回答用户，而是根据用户问题和候选来源，决定下一步调用哪个确定性工具，或者宣布信息已经足够。你只能使用下面列出的工具，不得生成代码、SQL、任意表达式或工具之外的参数。

可选工具：
1. analyze_image：读取某条图片证据的原始素材并调用视觉模型。
   - vision_task 必须三选一：describe_image（普通图片理解）、
     analyze_chart（图表趋势分析）、extract_table（从图片中提取表格）。
   - 图片型表格（表格证据仅保留截图、无法直接计算）也可用 extract_table 从截图提取。
   - 可选 question 用于说明希望关注的细节。
2. analyze_table：对某条表格证据执行确定性表格计算。
   - operation 必须六选一：select、filter、sort、aggregate、growth、compare。
   - parameters 使用 KnowledgeScope 表格分析参数契约，只允许使用候选来源中给出的真实列名/行标签。
   - growth：跨年比较用 axis=columns，并给出 row（行标签）、base_column、target_column；
     反向比较用 axis=rows，并给出 column、base_row、target_row。
   - compare：需要两条表格证据，给出 second_evidence_index、key_column、value_column。
   - compare 需要 second_evidence_index 指向另一条表格候选。
   - 如果只是想先查看表格内容，可用 select：{"columns": [], "rows": null}。

工具调用示例（输出一个 JSON 对象，不要输出解释或 Markdown）：
{"action":"call_tool","tool":"analyze_table","evidence_index":2,"operation":"aggregate","parameters":{"column":"2024年GDP","functions":["sum"]}}
{"action":"call_tool","tool":"analyze_table","evidence_index":2,"second_evidence_index":4,"operation":"compare","parameters":{"key_column":"国家","value_column":"2024年GDP","keys":["越南"]}}
{"action":"call_tool","tool":"analyze_image","evidence_index":1,"vision_task":"analyze_chart","question":"这张图说明了什么趋势？"}
{"action":"final","reason":"现有来源已经足够回答，不需要调用工具"}

规则：
- evidence_index 必须来自候选来源列表；不得编造来源或编号。
- 一条计划最多调用一个工具；已经获得的工具结果不要重复调用。
- 当问题要求对比两份不同文档/表格中的同名指标时，必须使用一次 compare 调用，并给出
  second_evidence_index、key_column、value_column；不要拆成两次 select，也不要自行计算差异。
- 如果候选文本证据已经能直接回答（例如标题、定义、原文位置、概述），优先输出 final，
  不要为了「看起来更专业」而调用视觉或表格工具。
- 但涉及表格中的具体数值查询（例如「某国某年 GDP 是多少」）、汇总、增长率或跨表对比时，
  必须调用 analyze_table，用确定性工具取数，不能凭原文摘录直接给出计算结论。
- 当问题要求分析图中趋势、峰值、变化、分布、结构或关系等原生视觉特征，且候选来源中有
  image 证据时，必须调用 analyze_image；图表类问题优先 vision_task="analyze_chart"。
  仅询问图片标题、页码、来源或纯文本事实时，不要调用 analyze_image。
- 如果工具失败或候选来源不足，宁可输出 final，让最终回答明确说明限制，也不要编造数据。
- 输出必须是单个合法 JSON 对象，且只包含上面工具约定中的字段。"""


def _infer_table_operation(parameters: dict[str, object]) -> TableOperation | None:
    """Infer the operation from an unambiguous parameter shape.

    Small planners occasionally emit correct parameters with the wrong
    ``operation`` label. The parameter contract is disjoint enough that this
    normalization is safe and avoids a needless degraded answer.
    """
    keys = set(parameters)
    if {"key_column", "value_column"} <= keys:
        return "compare"
    if "axis" in keys:
        return "growth"
    if "functions" in keys and "column" in keys:
        return "aggregate"
    if {"operator", "value"} <= keys and "column" in keys:
        return "filter"
    if "direction" in keys and "column" in keys:
        return "sort"
    if keys <= {"columns", "rows"}:
        return "select"
    return None


_OPERATION_PARAMETER_KEYS: dict[str, set[str]] = {
    "select": {"columns", "rows"},
    "filter": {"column", "operator", "value", "columns"},
    "sort": {"column", "direction", "limit", "columns"},
    "aggregate": {"column", "functions", "group_by"},
    "growth": {"axis", "column", "row", "base_column", "target_column", "base_row"},
    "compare": {"key_column", "value_column", "keys"},
}


def _sanitize_parameters(
    parameters: dict[str, object], operation: TableOperation
) -> dict[str, object]:
    """Drop keys that do not belong to the inferred operation.

    Planners sometimes mix keys from two operations (for example aggregate
    parameters plus ``columns``). After inferring the operation from the
    unambiguous keys, unrelated keys are ignored instead of failing the plan.
    """
    allowed = _OPERATION_PARAMETER_KEYS.get(operation)
    if allowed is None:
        return parameters
    sanitized = {key: value for key, value in parameters.items() if key in allowed}
    if operation == "aggregate":
        raw_functions = sanitized.get("functions")
        if isinstance(raw_functions, list):
            synonyms = {
                "average": "avg",
                "mean": "avg",
                "total": "sum",
                "minimum": "min",
                "maximum": "max",
            }
            normalized: list[str] = []
            for raw_function in raw_functions:
                if not isinstance(raw_function, str):
                    continue
                function = synonyms.get(raw_function.strip().lower(), raw_function.strip().lower())
                if function in {"sum", "avg", "min", "max"} and function not in normalized:
                    normalized.append(function)
            if normalized:
                sanitized["functions"] = normalized
    return sanitized


def _candidate_line(candidate: AgentCandidate) -> str:
    citation = candidate.citation
    modality = citation.modality or citation.candidate_kind
    page = str(citation.page_start)
    if citation.page_end != citation.page_start:
        page = f"{page}-{citation.page_end}"
    section = " / ".join(citation.section_path) or "未命名章节"
    title = candidate.document_title or citation.document_id
    lines = [
        (
            f"[{candidate.index}] marker={candidate.marker} kind={citation.candidate_kind} "
            f"modality={modality} document={title} page={page} section={section}"
        )
    ]
    if candidate.evidence_id is not None:
        lines.append(f"    evidence_id={candidate.evidence_id}")
    if candidate.table_columns:
        columns_json = json.dumps(list(candidate.table_columns), ensure_ascii=False)
        lines.append(f"    table_columns={columns_json}")
    if candidate.table_rows:
        rows_json = json.dumps(list(candidate.table_rows), ensure_ascii=False)
        lines.append(f"    table_row_labels={rows_json}")
    if candidate.table_issue:
        lines.append(f"    table_issue={candidate.table_issue}")
    snippet = (citation.snippet or "").strip()
    if snippet:
        lines.append(f"    snippet={snippet[:240]}")
    return "\n".join(lines)


def _observations_block(observations: Sequence[AgentToolObservation], max_chars: int) -> str:
    if not observations:
        return "（暂无工具结果）"
    rendered: list[str] = []
    remaining = max_chars
    for observation in observations:
        payload_text = json.dumps(observation.payload, ensure_ascii=False, default=str)
        block = f"工具 {observation.tool}: {observation.summary}\n结果: {payload_text}"
        if remaining <= 0:
            break
        block = block[:remaining]
        rendered.append(block)
        remaining -= len(block)
        if remaining <= 0:
            break
    return "\n\n".join(rendered)


class AgentPlanner:
    """Ask the configured text model for one validated, bounded plan."""

    def __init__(self, gateway: LLMGateway, settings: Settings) -> None:
        self._gateway = gateway
        self._settings = settings

    async def plan(
        self,
        query: str,
        candidates: Sequence[AgentCandidate],
        observations: Sequence[AgentToolObservation],
        *,
        round_number: int,
        history_context: str | None = None,
    ) -> AgentPlan:
        if not candidates:
            raise AgentPlanningError("没有可用于规划的检索候选来源")
        history_block = ""
        if history_context:
            history_block = f"对话历史（用于理解指代，不代表新的证据）：\n{history_context}\n\n"
        messages = [
            LLMMessage(role="system", content=_PLANNER_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    history_block + f"用户问题：{query}\n\n"
                    f"第 {round_number} 轮规划。\n\n"
                    "候选来源：\n"
                    + "\n".join(_candidate_line(candidate) for candidate in candidates)
                    + "\n\n已执行工具结果：\n"
                    + _observations_block(
                        observations,
                        self._settings.agent_tool_result_max_chars,
                    )
                    + "\n\n请只输出一个 JSON 对象。"
                ),
            ),
        ]
        last_error = "规划结果无效"
        for attempt in range(2):
            try:
                result = await self._gateway.complete(
                    LLMRequest(
                        messages=messages,
                        task_type="agent",
                        temperature=0.0,
                        max_tokens=self._settings.agent_planner_max_tokens,
                        response_format=LLMResponseFormat(type="json_object"),
                        reasoning="disabled",
                    )
                )
            except LLMError as error:
                raise AgentPlanningError("规划模型调用失败") from error
            try:
                plan = parse_agent_plan(result.text)
                self._validate_plan(plan, candidates)
                return plan
            except AgentPlanningError as error:
                last_error = error.message
                # Never log raw model output or user content: only the bounded
                # validation error and attempt metadata are safe to keep.
                logger.warning(
                    "agent planner attempt %s failed: %s",
                    attempt + 1,
                    last_error,
                )
                if attempt == 0:
                    messages = [
                        *messages,
                        LLMMessage(role="assistant", content=result.text[:2_000]),
                        LLMMessage(
                            role="user",
                            content=(
                                f"上一次输出无效：{last_error}。"
                                "请只输出一个符合工具调用约定的 JSON 对象。"
                            ),
                        ),
                    ]
        raise AgentPlanningError(last_error)

    def _validate_plan(self, plan: AgentPlan, candidates: Sequence[AgentCandidate]) -> None:
        if plan.action == "final":
            return
        by_index = {candidate.index: candidate for candidate in candidates}
        candidate = by_index.get(plan.evidence_index or 0)
        if candidate is None:
            raise AgentPlanningError("规划引用了不存在的候选来源编号")
        if not candidate.tool_capable:
            raise AgentPlanningError("该候选来源没有可读取的 Evidence")
        if plan.tool == "analyze_image":
            # 图片证据可做三种视觉任务，表格证据仅在提取图片型表格时可用视觉读取。
            allowed = candidate.modality == "image" or (
                candidate.modality == "table" and plan.vision_task == "extract_table"
            )
            if not allowed:
                raise AgentPlanningError("所选候选来源不是可读取的图片证据")
        if plan.tool == "analyze_table" and candidate.modality != "table":
            raise AgentPlanningError("所选候选来源不是表格证据")
        second = (
            by_index.get(plan.second_evidence_index)
            if plan.second_evidence_index is not None
            else None
        )
        if plan.tool == "analyze_table":
            if plan.operation == "compare":
                if second is None or second.modality != "table" or not second.tool_capable:
                    raise AgentPlanningError("跨表对比需要另一条表格证据")
                compare_table = {
                    "document_id": str(second.citation.document_id),
                    "evidence_id": second.evidence_id,
                }
            else:
                compare_table = None
            parameters = dict(plan.parameters or {})
            payload: dict[str, object] = {
                "schema_version": "1.0",
                "knowledge_base_id": candidate.citation.knowledge_base_id,
                "operation": plan.operation,
                "table": {
                    "document_id": str(candidate.citation.document_id),
                    "evidence_id": candidate.evidence_id,
                },
                "parameters": parameters,
            }
            if compare_table is not None:
                payload["compare_table"] = compare_table
            try:
                TableAnalysisRequest.model_validate(payload)
            except ValidationError as error:
                inferred = _infer_table_operation(parameters)
                if inferred is None or inferred == plan.operation:
                    first = error.errors()[0]
                    raise AgentPlanningError(
                        f"表格工具参数不合法（{first.get('loc')}）：{first.get('msg')}"
                    ) from error
                payload["operation"] = inferred
                sanitized = _sanitize_parameters(parameters, inferred)
                payload["parameters"] = sanitized
                try:
                    TableAnalysisRequest.model_validate(payload)
                except ValidationError:
                    first = error.errors()[0]
                    raise AgentPlanningError(
                        f"表格工具参数不合法（{first.get('loc')}）：{first.get('msg')}"
                    ) from error
                plan.operation = inferred
                plan.parameters = sanitized
                logger.info(
                    "agent planner normalized table operation to %s for parameters %s",
                    inferred,
                    sorted(parameters),
                )
        if plan.second_evidence_index is not None and (
            second is None or second.modality != "table" or not second.tool_capable
        ):
            raise AgentPlanningError("跨表对比需要另一条表格证据")


__all__ = ["AgentPlanner"]
