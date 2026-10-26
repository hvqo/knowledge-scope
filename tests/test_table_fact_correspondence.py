# ruff: noqa: RUF001
"""Adversarial regression tests for table value / business field correspondence."""

from __future__ import annotations

from typing import Any

import anyio

from knowledge_scope.agent.schemas import AgentToolOutcome
from knowledge_scope.rag.schemas import RAGQueryRequest, RAGStreamEvent
from test_agent import (
    DOC_ID,
    KB_ID,
    _candidate,
    _collect,
    _FakeGateway,
    _no_document,
    _retrieval_result,
    _service,
    _StubExecutor,
    _StubRetrieval,
)


def _table_outcome(
    *,
    columns: list[dict[str, str]],
    rows: list[list[str]],
    units: list[str],
    operation: str = "select",
    formula: str | None = None,
) -> AgentToolOutcome:
    payload: dict[str, Any] = {
        "status": "ok",
        "operation": operation,
        "columns": columns,
        "rows": [{"cells": row} for row in rows],
        "units": units,
        "sources": [
            {
                "document_id": str(DOC_ID),
                "evidence_id": "evidence_v1_" + "c" * 64,
                "page_start": 1,
                "page_end": 1,
                "source_block_ids": ["p1-b3"],
            }
        ],
    }
    if formula is not None:
        payload["formula"] = formula
    return AgentToolOutcome(
        tool="analyze_table",
        status="completed",
        summary="表格计算完成（fake）",
        payload=payload,
        fact_status="source_verified",
    )


def _run_high_risk_table_turn(
    *,
    query: str,
    outcome: AgentToolOutcome,
    deltas: list[str],
) -> list[RAGStreamEvent]:
    candidate = _candidate(
        1,
        marker="C1",
        modality="table",
        evidence_id="evidence_v1_" + "c" * 64,
        table_columns=("国家", "2024年GDP"),
    )
    gateway = _FakeGateway(
        plans=[
            {
                "action": "call_tool",
                "tool": "analyze_table",
                "evidence_index": 1,
                "operation": "select",
                "parameters": {"columns": [0, 1], "rows": None},
            },
            {"action": "final", "reason": "结果足够"},
        ],
        deltas=deltas,
    )
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result([candidate])),
        executor=_StubExecutor([outcome]),
    )
    request = RAGQueryRequest(
        query=query,
        knowledge_base_id=KB_ID,
        answer_mode="agent",
    )
    return anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))


def _answer_text(events: list[RAGStreamEvent]) -> str:
    return "".join(event.data["text"] for event in events if event.event == "answer_delta")


def _complete(events: list[RAGStreamEvent]) -> dict[str, Any]:
    return next(event for event in events if event.event == "complete").data


def test_number_set_alone_cannot_detect_swapped_values() -> None:
    """Old validation only compared number sets; a swap keeps the same set."""

    tool_numbers = {"4760", "5200"}
    swapped_answer_numbers = {"5200", "4760"}
    assert swapped_answer_numbers == tool_numbers


def test_swapped_country_values_are_replaced_by_deterministic_result() -> None:
    outcome = _table_outcome(
        columns=[
            {"name": "国家", "kind": "text"},
            {"name": "2024年GDP", "kind": "number", "unit": "亿美元"},
        ],
        rows=[["越南", "4760"], ["泰国", "5200"]],
        units=["亿美元"],
    )
    events = _run_high_risk_table_turn(
        query="越南和泰国 2024 年的 GDP 分别是多少？请给出具体数值。",
        outcome=outcome,
        deltas=["越南 GDP 为 5200 亿美元，泰国 GDP 为 4760 亿美元。"],
    )
    answer = _answer_text(events)
    complete = _complete(events)
    assert "越南：2024年GDP = 4760 亿美元" in answer
    assert "泰国：2024年GDP = 5200 亿美元" in answer
    assert "越南 GDP 为 5200" not in answer
    assert "泰国 GDP 为 4760" not in answer
    assert complete["fact_status"] == "source_verified"
    assert complete["safety_gate"] == "none"
    assert complete["finish_reason"] == "deterministic_table"


def test_swapped_years_are_not_presented_as_verified() -> None:
    outcome = _table_outcome(
        operation="growth",
        columns=[
            {"name": "指标", "kind": "text"},
            {"name": "基期值", "kind": "number", "unit": "亿美元"},
            {"name": "报告期值", "kind": "number", "unit": "亿美元"},
            {"name": "绝对变化", "kind": "number", "unit": "亿美元"},
            {"name": "相对变化", "kind": "percent", "unit": "%"},
        ],
        rows=[["越南", "4300", "4760", "460", "10.6977"]],
        units=["亿美元", "%"],
        formula="((4760 - 4300) / 4300) × 100% = 10.6977%",
    )
    events = _run_high_risk_table_turn(
        query="越南 2023 年和 2024 年的 GDP 分别是多少？增长了百分之几？",
        outcome=outcome,
        deltas=["越南 2023 年 GDP 为 4760 亿美元，2024 年 GDP 为 4300 亿美元，增长 10.6977%。"],
    )
    answer = _answer_text(events)
    assert "基期值 = 4300" in answer
    assert "报告期值 = 4760" in answer
    assert "绝对变化 = 460 亿美元" in answer
    assert "相对变化 = 10.6977%" in answer
    assert "2023 年 GDP 为 4760" not in answer
    assert "2024 年 GDP 为 4300" not in answer
    assert "10.6977" in answer


def test_same_number_with_wrong_unit_is_not_presented_as_verified() -> None:
    outcome = _table_outcome(
        columns=[
            {"name": "国家", "kind": "text"},
            {"name": "2024年GDP", "kind": "number", "unit": "亿美元"},
        ],
        rows=[["越南", "4760"]],
        units=["亿美元"],
    )
    events = _run_high_risk_table_turn(
        query="越南 2024 年 GDP 的具体数值是多少？",
        outcome=outcome,
        deltas=["越南 2024 年 GDP 为 4760 万元。"],
    )
    answer = _answer_text(events)
    assert "4760 亿美元" in answer
    assert "4760 万元" not in answer


def test_correct_table_calculation_and_formula_stay_verified() -> None:
    outcome = _table_outcome(
        operation="growth",
        columns=[
            {"name": "指标", "kind": "text"},
            {"name": "基期值", "kind": "number", "unit": "亿美元"},
            {"name": "报告期值", "kind": "number", "unit": "亿美元"},
            {"name": "绝对变化", "kind": "number", "unit": "亿美元"},
            {"name": "相对变化", "kind": "percent", "unit": "%"},
        ],
        rows=[["越南", "4300", "4760", "460", "10.6977"]],
        units=["亿美元", "%"],
        formula="((4760 - 4300) / 4300) × 100% = 10.6977%",
    )
    events = _run_high_risk_table_turn(
        query="越南 2024 年 GDP 相比 2023 年增长了百分之几？请给出精确结果。",
        outcome=outcome,
        deltas=["越南 2023 年 4300 亿美元，2024 年 4760 亿美元，增长 10.6977%。"],
    )
    answer = _answer_text(events)
    complete = _complete(events)
    assert "相对变化 = 10.6977%" in answer
    assert "10.6977 亿美元" not in answer
    assert "((4760 - 4300) / 4300) × 100% = 10.6977%" in answer
    assert "无法核验" not in answer
    assert complete["fact_status"] == "source_verified"
    assert complete["safety_gate"] == "none"


def test_structured_table_facts_keep_row_column_and_unit_binding() -> None:
    outcome = _table_outcome(
        columns=[
            {"name": "国家", "kind": "text"},
            {"name": "2023年GDP", "kind": "number", "unit": "亿美元"},
            {"name": "2024年GDP", "kind": "number", "unit": "亿美元"},
        ],
        rows=[["越南", "4300", "4760"]],
        units=["亿美元"],
    )
    from knowledge_scope.agent.service import _structured_table_facts

    facts = _structured_table_facts(outcome)
    assert facts == [
        "越南：2023年GDP = 4300 亿美元；2024年GDP = 4760 亿美元。",
    ]


def test_compare_result_renders_both_documents_in_order() -> None:
    from knowledge_scope.agent.service import _structured_table_facts

    outcome = AgentToolOutcome(
        tool="analyze_table",
        status="completed",
        summary="表格计算完成（compare）",
        payload={
            "status": "ok",
            "operation": "compare",
            "columns": [
                {"name": "国家", "kind": "text"},
                {"name": "差异", "kind": "number", "unit": "亿美元"},
                {"name": "相对差异", "kind": "percent", "unit": "%"},
            ],
            "rows": [{"cells": ["越南", "60", "1.2766"]}],
            "formula": "difference = A - B",
            "units": ["亿美元", "%"],
        },
        fact_status="source_verified",
    )
    facts = _structured_table_facts(outcome)
    assert facts == [
        "越南：差异 = 60 亿美元；相对差异 = 1.2766%。",
        "公式：difference = A - B",
    ]


def _real_growth_result() -> dict[str, Any]:
    from knowledge_scope.table_analysis.models import (
        TableAnalysisResult,
        TableSource,
    )
    from knowledge_scope.table_analysis.operations import GrowthParameters, run_growth
    from knowledge_scope.table_analysis.parsing import parse_markdown_table

    table = parse_markdown_table(
        "| 年份 | GDP |\n| --- | --- |\n| 2023 | 4300 |\n| 2024 | 4760 |\n",
        caption="表 1 越南 GDP 单位：亿美元",
        unit_hint="亿美元",
    )
    outcome = run_growth(
        table,
        GrowthParameters(axis="rows", column="GDP", base_row=0, target_row=1),
    )
    result = TableAnalysisResult(
        operation="growth",
        status="ok",
        columns=outcome.columns,
        rows=outcome.rows,
        formula=outcome.formula,
        inputs=outcome.inputs,
        units=outcome.units,
        warnings=outcome.warnings,
        sources=[
            TableSource(
                table_role="table_a",
                document_id=DOC_ID,
                evidence_id="evidence_v1_" + "d" * 64,
                page_start=1,
                page_end=1,
                source_block_ids=["p1-b2"],
            )
        ],
    )
    return result.model_dump(mode="json")


def test_real_table_analysis_result_has_no_metric_unit_on_percent_columns() -> None:
    from knowledge_scope.agent.service import _structured_table_facts

    payload = _real_growth_result()
    assert payload["status"] == "ok"
    columns = {column["name"]: column for column in payload["columns"]}
    assert columns["相对变化"]["kind"] == "percent"
    assert columns["相对变化"]["unit"] == "%"

    outcome = AgentToolOutcome(
        tool="analyze_table",
        status="completed",
        summary="表格计算完成（growth）",
        payload=payload,
        fact_status="source_verified",
    )
    facts = _structured_table_facts(outcome)
    text = "\n".join(facts)
    assert "基期值 = 4300 亿美元" in text
    assert "报告期值 = 4760 亿美元" in text
    assert "绝对变化 = 460 亿美元" in text
    assert "相对变化 = 10.6977%" in text
    assert "10.6977 亿美元" not in text


def test_percentage_metric_absolute_change_uses_percentage_points() -> None:
    from knowledge_scope.agent.service import _structured_table_facts
    from knowledge_scope.table_analysis.models import (
        TableAnalysisResult,
    )
    from knowledge_scope.table_analysis.operations import GrowthParameters, run_growth
    from knowledge_scope.table_analysis.parsing import parse_markdown_table

    table = parse_markdown_table(
        "| 年份 | 增长率 |\n| --- | --- |\n| 2023 | 10% |\n| 2024 | 12.5% |\n"
    )
    outcome = run_growth(
        table,
        GrowthParameters(axis="rows", column="增长率", base_row=0, target_row=1),
    )
    result = TableAnalysisResult(
        operation="growth",
        status="ok",
        columns=outcome.columns,
        rows=outcome.rows,
        formula=outcome.formula,
        inputs=outcome.inputs,
        units=outcome.units,
        warnings=outcome.warnings,
    )
    facts = _structured_table_facts(
        AgentToolOutcome(
            tool="analyze_table",
            status="completed",
            summary="",
            payload=result.model_dump(mode="json"),
            fact_status="source_verified",
        )
    )
    text = "\n".join(facts)
    assert "绝对变化 = 2.5 百分点" in text
    assert "相对变化 = 25.0000%" in text
