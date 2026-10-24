# ruff: noqa: RUF001
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anyio

from knowledge_scope.agent.schemas import AgentToolOutcome
from knowledge_scope.rag.schemas import RAGQueryRequest, RAGStreamEvent
from knowledge_scope.shared.config import Settings
from knowledge_scope.vision.safety import (
    answer_has_numeric_claims,
    assess_visual_risk,
    build_gated_visual_answer,
    extract_numeric_claims,
)
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


class _TrackingRAGService:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: RAGQueryRequest) -> AsyncIterator[RAGStreamEvent]:
        self.calls += 1
        yield RAGStreamEvent(event="answer_delta", data={"text": "常规检索回答"})
        yield RAGStreamEvent(
            event="complete",
            data={
                "status": "completed",
                "prompt_version": "rag-qa-v6",
                "retrieval_mode": "unified",
                "route": "unified",
                "latency_ms": 1.0,
            },
        )


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {"_env_file": None, "environment": "test", "agent_enabled": True}
    values.update(overrides)
    return Settings(**values)


def _vision_outcome(
    *,
    status: str = "completed",
    answer: str = "Q1 最低约 120 万元。",
    fact_status: str = "visual_estimate",
) -> AgentToolOutcome:
    payload: dict[str, Any] = {
        "status": "ok" if status == "completed" else "failed",
        "answer": answer,
        "sources": [
            {
                "knowledge_base_id": str(KB_ID),
                "document_id": str(DOC_ID),
                "evidence_id": "evidence_v1_" + "a" * 64,
                "page_start": 1,
                "page_end": 1,
                "source_block_ids": ["p1-b2"],
            }
        ],
    }
    if status != "completed":
        payload.update({"error_code": "provider_timeout", "message": "视觉模型调用超时"})
    return AgentToolOutcome(
        tool="analyze_image",
        status=status,  # type: ignore[arg-type]
        summary="图片理解完成（fake-vlm）" if status == "completed" else "图片理解失败",
        payload=payload,
        fact_status=fact_status,
        numeric_claims=[{"value": "120", "context": "Q1 最低约 120 万元。"}]
        if status == "completed"
        else [],
    )


def _table_outcome() -> AgentToolOutcome:
    return AgentToolOutcome(
        tool="analyze_table",
        status="completed",
        summary="表格计算完成（growth）",
        payload={"status": "ok", "formula": "((4760 - 4300) / 4300) × 100% = 10.6977%"},
        fact_status="source_verified",
    )


def _answers(events: list[RAGStreamEvent]) -> str:
    return "".join(event.data["text"] for event in events if event.event == "answer_delta")


def test_visual_risk_classifier_flags_precise_visual_questions() -> None:
    high_risk = assess_visual_risk("图表中最低季度的具体销售额是多少？")
    assert high_risk.high_risk is True
    assert high_risk.visual_intent is True

    text_question = assess_visual_risk("文档主要讲了什么内容？")
    assert text_question.high_risk is False


def test_extract_numeric_claims_ignores_years_and_quarters() -> None:
    claims = extract_numeric_claims("2023 年 Q4 销售额为 210 万元，2024 年增长 10.6977%。")
    values = {claim.value for claim in claims}
    assert "2023" not in values
    assert "2024" not in values
    assert "210" in values
    assert "10.6977%" in values


def test_build_gated_visual_answer_removes_unverified_numbers() -> None:
    answer, notice = build_gated_visual_answer(
        description="Q1 最低约 120 万元。",
        vision_completed=True,
    )
    assert "120" not in answer
    assert "无法核验" in answer
    assert "未经核验" in notice
    assert "数值未经核验" in answer or "精确" in answer


def test_gated_agent_turn_never_streams_unverified_numbers() -> None:
    gateway = _FakeGateway(
        plans=[
            {
                "action": "call_tool",
                "tool": "analyze_image",
                "evidence_index": 1,
                "vision_task": "analyze_chart",
            },
            {"action": "final", "reason": "证据足够"},
        ],
        deltas=["Q1 销售额为 120 万元。"],
    )
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result([_candidate(1, marker="C1")])),
        executor=_StubExecutor([_vision_outcome()]),
    )
    request = RAGQueryRequest(
        query="图表中最低季度的具体销售额是多少？",
        knowledge_base_id=KB_ID,
        answer_mode="agent",
    )
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))

    answer = _answers(events)
    assert "120" not in answer
    assert "无法核验" in answer or "未能核验" in answer
    complete = next(event for event in events if event.event == "complete")
    assert complete.data["safety_gate"] == "visual_numeric_withheld"
    assert complete.data["fact_status"] == "visual_estimate"
    assert complete.data["finish_reason"] == "safety_gate"
    assert len(complete.data["visual_sources"]) == 1
    assert complete.data["visual_sources"][0]["evidence_id"] == "evidence_v1_" + "a" * 64


def test_verified_structured_table_answer_is_not_gated() -> None:
    gateway = _FakeGateway(
        plans=[
            {
                "action": "call_tool",
                "tool": "analyze_table",
                "evidence_index": 1,
                "operation": "growth",
                "parameters": {
                    "axis": "rows",
                    "column": "GDP",
                    "base_row": 0,
                    "target_row": 1,
                },
            },
            {"action": "final", "reason": "工具结果足够"},
        ],
        deltas=["越南 2024 年 GDP 增长率为 10.6977%。[C1]"],
    )
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result([_candidate(1, marker="C1", modality="table")])),
        executor=_StubExecutor([_table_outcome()]),
    )
    request = RAGQueryRequest(
        query="表 1 中越南 2024 年 GDP 增长率是多少？",
        knowledge_base_id=KB_ID,
        answer_mode="agent",
    )
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))

    answer = _answers(events)
    assert "10.6977" in answer
    complete = next(event for event in events if event.event == "complete")
    assert complete.data["safety_gate"] == "none"
    assert complete.data["fact_status"] == "source_verified"


def test_vision_failure_on_visual_number_question_is_gated_without_rag_fallback() -> None:
    rag_service = _TrackingRAGService()
    gateway = _FakeGateway(
        plans=[
            {
                "action": "call_tool",
                "tool": "analyze_image",
                "evidence_index": 1,
                "vision_task": "analyze_chart",
            },
            {"action": "final", "reason": "工具失败"},
        ],
        deltas=["常规模型猜测的数值 120。"],
    )
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result([_candidate(1, marker="C1")])),
        executor=_StubExecutor(
            [_vision_outcome(status="failed", answer="", fact_status="not_available")]
        ),
        rag_service=rag_service,
    )
    request = RAGQueryRequest(
        query="图表中最低季度的具体销售额是多少？",
        knowledge_base_id=KB_ID,
        answer_mode="agent",
    )
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))

    answer = _answers(events)
    assert "120" not in answer
    complete = next(event for event in events if event.event == "complete")
    assert complete.data["safety_gate"] == "vision_unavailable"
    assert complete.data["fact_status"] == "not_available"
    assert rag_service.calls == 0


def test_answer_has_numeric_claims_detects_visual_numbers() -> None:
    assert answer_has_numeric_claims("最低季度销售额为 120 万元。")
    assert not answer_has_numeric_claims("图中展示了光合作用的结构。")
