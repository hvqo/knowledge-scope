# ruff: noqa: RUF001
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import anyio
import httpx
import pytest
from pydantic import ValidationError

from knowledge_scope.agent import (
    AgentCandidate,
    AgentPlan,
    AgentPlanningError,
    AgentToolOutcome,
    parse_agent_plan,
)
from knowledge_scope.agent.planner import AgentPlanner
from knowledge_scope.agent.service import AgentChatService
from knowledge_scope.agent.tools import AgentRetrievalTool, AgentToolExecutor, parse_evidence_table
from knowledge_scope.api.app import create_app
from knowledge_scope.documents.models import Document
from knowledge_scope.llm.schemas import LLMRequest, LLMResult, LLMStreamEvent
from knowledge_scope.rag.history import HistoryPlan
from knowledge_scope.rag.schemas import (
    RAGCitation,
    RAGCitationsData,
    RAGCompleteData,
    RAGHistoryTurn,
    RAGQueryRequest,
    RAGStreamEvent,
    RAGToolStatusData,
)
from knowledge_scope.shared.config import Settings
from knowledge_scope.table_analysis.models import TableAnalysisResult
from knowledge_scope.vision.schemas import VisionAnalysisResult
from test_document_assets import (
    IMAGE_ASSET_REF,
    TABLE_ASSET_REF,
    build_canonical,
    write_evidence,
    write_parsing_artifact,
)

KB_ID = uuid4()
DOC_ID = uuid4()
SECOND_DOC_ID = uuid4()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "environment": "test",
        "agent_enabled": True,
        "agent_evidence_sufficiency_enabled": False,
    }
    values.update(overrides)
    return Settings(**values)


def _citation(
    marker: str,
    *,
    document_id: UUID = DOC_ID,
    knowledge_base_id: UUID = KB_ID,
    modality: str = "image",
    evidence_id: str | None = None,
) -> RAGCitation:
    return RAGCitation(
        marker=marker,
        document_id=document_id,
        knowledge_base_id=knowledge_base_id,
        candidate_kind="evidence",
        evidence_id=evidence_id or "evidence_v1_" + "a" * 64,
        modality=modality,  # type: ignore[arg-type]
        representation_ids=["representation_v1_" + "b" * 64],
        asset_refs=["sample-textbook/auto/images/asset-0.png"],
        page_start=1,
        page_end=1,
        source_block_ids=["p1-b2"],
        section_path=["章节"],
        snippet="图 1 示例图片",
        snippet_kind="representation",
        final_rank=1,
        final_reranker_score=0.9,
        branch_provenance=[],
    )


def _candidate(
    index: int,
    *,
    marker: str,
    modality: str = "image",
    document_id: UUID = DOC_ID,
    evidence_id: str | None = None,
    table_columns: tuple[str, ...] = (),
    table_issue: str | None = None,
) -> AgentCandidate:
    return AgentCandidate(
        index=index,
        marker=marker,
        citation=_citation(
            marker,
            document_id=document_id,
            modality=modality,
            evidence_id=evidence_id,
        ),
        document_title="sample.pdf",
        table_columns=table_columns,
        table_issue=table_issue,
    )


def _document(document_id: UUID = DOC_ID, knowledge_base_id: UUID = KB_ID) -> Document:
    return Document(
        id=document_id,
        knowledge_base_id=knowledge_base_id,
        original_filename="sample.pdf",
        storage_kind="managed",
        storage_key=f"documents/{knowledge_base_id}/{document_id}/original.pdf",
        media_type="application/pdf",
        size_bytes=1024,
        sha256="a" * 64,
        status="uploaded",
    )


def _table_canonical() -> Any:
    canonical = build_canonical(DOC_ID)
    blocks = [
        block.model_copy(
            update={
                "markdown": ("| 年份 | GDP |\n| --- | --- |\n| 2023年 | 4300 |\n| 2024年 | 4760 |")
            }
        )
        if block.block_id == "p1-b3"
        else block
        for block in canonical.pages[0].blocks
    ]
    return canonical.model_copy(
        update={"pages": [canonical.pages[0].model_copy(update={"blocks": blocks})]}
    )


def _retrieval_result(candidates: list[AgentCandidate]) -> SimpleNamespace:
    return SimpleNamespace(
        candidates=tuple(candidates),
        context_text="[C1] sample.pdf 第1页 modality=image\n示例上下文",
        retrieval_latency_ms=12.5,
        degraded=False,
        branch_statuses={"multimodal": "success"},
    )


class _FakeGateway:
    """Small gateway double with scripted planner JSON and final answer deltas."""

    def __init__(
        self,
        plans: list[dict[str, Any]],
        deltas: list[str] | None = None,
    ) -> None:
        self._plans = plans
        self._deltas = deltas or ["回答"]
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResult:
        self.requests.append(request)
        plan = self._plans.pop(0)
        return LLMResult(
            text=json.dumps(plan, ensure_ascii=False),
            provider="fake",
            model="fake-planner",
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
            finish_reason="stop",
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self.requests.append(request)
        for delta in self._deltas:
            yield LLMStreamEvent(
                delta=delta,
                provider="fake",
                model="fake-answer",
                input_tokens=20,
                output_tokens=10,
                finish_reason="stop",
            )


class _StubAgentService(AgentChatService):
    def __init__(
        self,
        settings: Settings,
        *,
        gateway: Any,
        retrieval: Any,
        executor: Any,
        rag_service: Any = None,
        history_compressor: Any = None,
        query_rewrite: Any = None,
    ) -> None:
        super().__init__(
            settings,
            gateway=gateway,  # type: ignore[arg-type]
            unified_retrieval=object(),  # type: ignore[arg-type]
            vision_service=object(),  # type: ignore[arg-type]
            table_service=object(),  # type: ignore[arg-type]
            rag_service=rag_service,
            history_compressor=history_compressor,
            query_rewrite=query_rewrite,
        )
        self._retrieval = retrieval
        self._executor = executor

    def _build_retrieval_tool(self, document_loader: Any) -> Any:
        return self._retrieval

    def _build_executor(self, document_loader: Any) -> Any:
        return self._executor


async def _collect(events: AsyncIterator[RAGStreamEvent]) -> list[RAGStreamEvent]:
    return [event async for event in events]


# ---------------------------------------------------------------------------
# Planner contracts
# ---------------------------------------------------------------------------


def test_parse_agent_plan_accepts_final_plan() -> None:
    plan = parse_agent_plan('{"action": "final", "reason": "证据足够"}')
    assert plan.action == "final"
    assert plan.tool is None


def test_parse_agent_plan_accepts_table_tool_plan() -> None:
    plan = parse_agent_plan(
        json.dumps(
            {
                "action": "call_tool",
                "reason": "需要计算增长率",
                "tool": "analyze_table",
                "evidence_index": 2,
                "operation": "growth",
                "parameters": {
                    "axis": "rows",
                    "column": "GDP",
                    "base_row": "2023年",
                    "target_row": "2024年",
                },
            },
            ensure_ascii=False,
        )
    )
    assert plan.action == "call_tool"
    assert plan.tool == "analyze_table"
    assert plan.evidence_index == 2
    assert plan.operation == "growth"


def test_parse_agent_plan_rejects_invalid_json_and_contract_violations() -> None:
    with pytest.raises(AgentPlanningError, match="JSON"):
        parse_agent_plan("not json")
    with pytest.raises(AgentPlanningError):
        parse_agent_plan('{"action": "final", "tool": "analyze_image"}')
    with pytest.raises(ValidationError):
        AgentPlan(
            action="call_tool",
            tool="analyze_table",
            evidence_index=1,
            operation="compare",
        )


def test_planner_allows_vision_table_extraction_on_table_screenshot() -> None:
    planner = AgentPlanner(_FakeGateway([]), _settings())  # type: ignore[arg-type]
    table_candidate = _candidate(
        1,
        marker="C1",
        modality="table",
        evidence_id="evidence_v1_" + "d" * 64,
    )
    planner._validate_plan(
        AgentPlan(
            action="call_tool",
            tool="analyze_image",
            evidence_index=1,
            vision_task="extract_table",
        ),
        [table_candidate],
    )
    with pytest.raises(AgentPlanningError, match="图片证据"):
        planner._validate_plan(
            AgentPlan(
                action="call_tool",
                tool="analyze_image",
                evidence_index=1,
                vision_task="describe_image",
            ),
            [table_candidate],
        )


def test_planner_accepts_compare_plan_with_second_table_candidate() -> None:
    planner = AgentPlanner(_FakeGateway([]), _settings())  # type: ignore[arg-type]
    first = AgentCandidate(
        index=1,
        marker="C1",
        citation=_citation("C1", modality="table", evidence_id="evidence_v1_" + "e" * 64),
        document_title="a.pdf",
        table_columns=("国家", "2024年GDP"),
        table_rows=("越南", "泰国"),
    )
    second = AgentCandidate(
        index=2,
        marker="C2",
        citation=_citation("C2", modality="table", evidence_id="evidence_v1_" + "f" * 64),
        document_title="b.pdf",
        table_columns=("国家", "2024年GDP"),
        table_rows=("越南", "泰国"),
    )
    plan = AgentPlan(
        action="call_tool",
        tool="analyze_table",
        evidence_index=1,
        second_evidence_index=2,
        operation="compare",
        parameters={"key_column": "国家", "value_column": "2024年GDP", "keys": ["越南"]},
    )
    planner._validate_plan(plan, [first, second])


def test_planner_rejects_invalid_table_parameters_before_execution() -> None:
    planner = AgentPlanner(_FakeGateway([]), _settings())  # type: ignore[arg-type]
    table_candidate = AgentCandidate(
        index=1,
        marker="C1",
        citation=_citation(
            "C1",
            modality="table",
            evidence_id="evidence_v1_" + "d" * 64,
        ),
        document_title="sample.pdf",
        table_columns=("年份", "GDP"),
        table_rows=("2023年", "2024年"),
    )
    with pytest.raises(AgentPlanningError, match="表格工具参数不合法"):
        planner._validate_plan(
            AgentPlan(
                action="call_tool",
                tool="analyze_table",
                evidence_index=1,
                operation="growth",
                parameters={"axis": "columns"},
            ),
            [table_candidate],
        )


def test_planner_normalizes_mismatched_table_operation_label() -> None:
    planner = AgentPlanner(_FakeGateway([]), _settings())  # type: ignore[arg-type]
    table_candidate = AgentCandidate(
        index=1,
        marker="C1",
        citation=_citation("C1", modality="table", evidence_id="evidence_v1_" + "d" * 64),
        document_title="sample.pdf",
        table_columns=("国家", "2024年GDP"),
        table_rows=("越南",),
    )
    plan = AgentPlan(
        action="call_tool",
        tool="analyze_table",
        evidence_index=1,
        operation="select",
        parameters={"column": "2024年GDP", "functions": ["sum"]},
    )
    planner._validate_plan(plan, [table_candidate])
    assert plan.operation == "aggregate"


def test_planner_sanitizes_mixed_parameter_keys_on_operation_normalization() -> None:
    planner = AgentPlanner(_FakeGateway([]), _settings())  # type: ignore[arg-type]
    table_candidate = AgentCandidate(
        index=1,
        marker="C1",
        citation=_citation("C1", modality="table", evidence_id="evidence_v1_" + "d" * 64),
        document_title="sample.pdf",
        table_columns=("国家", "2023年GDP", "2024年GDP"),
        table_rows=("越南", "泰国", "文莱"),
    )
    plan = AgentPlan(
        action="call_tool",
        tool="analyze_table",
        evidence_index=1,
        operation="select",
        parameters={
            "column": "2024年GDP",
            "functions": ["sum", "avg"],
            "columns": ["国家"],
        },
    )
    planner._validate_plan(plan, [table_candidate])
    assert plan.operation == "aggregate"
    assert plan.parameters == {"column": "2024年GDP", "functions": ["sum", "avg"]}


def test_planner_normalizes_aggregate_function_synonyms() -> None:
    planner = AgentPlanner(_FakeGateway([]), _settings())  # type: ignore[arg-type]
    table_candidate = AgentCandidate(
        index=1,
        marker="C1",
        citation=_citation("C1", modality="table", evidence_id="evidence_v1_" + "d" * 64),
        document_title="sample.pdf",
        table_columns=("国家", "2024年GDP"),
        table_rows=("越南", "泰国", "文莱"),
    )
    plan = AgentPlan(
        action="call_tool",
        tool="analyze_table",
        evidence_index=1,
        operation="select",
        parameters={"column": "2024年GDP", "functions": ["average"]},
    )
    planner._validate_plan(plan, [table_candidate])
    assert plan.operation == "aggregate"
    assert plan.parameters == {"column": "2024年GDP", "functions": ["avg"]}


def test_planner_repairs_malformed_model_output_once() -> None:
    calls: list[LLMRequest] = []

    class _RepairGateway:
        def __init__(self) -> None:
            self.outputs = iter(
                [
                    "no json here",
                    json.dumps({"action": "final", "reason": "证据足够"}),
                ]
            )

        async def complete(self, request: LLMRequest) -> LLMResult:
            calls.append(request)
            return LLMResult(
                text=next(self.outputs),
                provider="fake",
                model="fake",
                latency_ms=1.0,
            )

    planner = AgentPlanner(_RepairGateway(), _settings())  # type: ignore[arg-type]

    async def run_plan() -> AgentPlan:
        return await planner.plan(
            "问题",
            [_candidate(1, marker="C1")],
            [],
            round_number=1,
        )

    plan = anyio.run(run_plan)
    assert plan.action == "final"
    assert len(calls) == 2
    assert calls[1].messages[-1].role == "user"
    assert calls[1].task_type == "agent"
    assert calls[1].response_format is not None


# ---------------------------------------------------------------------------
# Retrieval tool
# ---------------------------------------------------------------------------


def test_parse_evidence_table_reads_stored_markdown_representation(
    test_data_dir: Any,
) -> None:
    canonical = _table_canonical()
    write_parsing_artifact(test_data_dir, DOC_ID, canonical=canonical)
    write_evidence(test_data_dir, canonical, str(KB_ID))
    from knowledge_scope.documents.assets import get_evidence

    snapshot = get_evidence(
        _settings(data_dir=test_data_dir), _document(), _table_evidence_id(test_data_dir)
    )
    parsed = parse_evidence_table(snapshot.evidence)
    assert [column.header for column in parsed.columns] == ["年份", "GDP"]
    assert parsed.rows[0][1].text == "4300"


def _table_evidence_id(test_data_dir: Any) -> str:
    from knowledge_scope.evidence.lifecycle import evidence_artifact_path
    from knowledge_scope.evidence.models import MultimodalEvidenceDocument

    artifact = MultimodalEvidenceDocument.model_validate_json(
        evidence_artifact_path(test_data_dir, DOC_ID).read_bytes()
    )
    return next(item.evidence_id for item in artifact.evidence if item.modality == "table")


def _unified_candidate(
    *,
    document_id: UUID = DOC_ID,
    evidence_id: str,
    modality: str,
    source_block_ids: list[str],
    asset_refs: list[str],
) -> SimpleNamespace:
    return SimpleNamespace(
        candidate_kind="evidence",
        knowledge_base_id=KB_ID,
        document_id=document_id,
        chunk_id=None,
        evidence_id=evidence_id,
        modality=modality,
        representation_ids=["representation_v1_" + "b" * 64],
        asset_refs=asset_refs,
        page_start=1,
        page_end=1,
        source_block_ids=source_block_ids,
        section_path=["章节"],
        final_rank=1,
        final_reranker_score=0.87,
        source_text="图 1 示例图片",
        rerank_text="图 1 示例图片",
        branches=[
            SimpleNamespace(
                branch="multimodal",
                rank=1,
                score=0.87,
                graph_seed_entity_id=None,
                graph_retrieval_reason=None,
                graph_paths=[],
            )
        ],
    )


def test_retrieval_tool_enriches_table_columns_and_keeps_marker_mapping(
    test_data_dir: Any,
) -> None:
    canonical = _table_canonical()
    write_parsing_artifact(test_data_dir, DOC_ID, canonical=canonical)
    write_evidence(test_data_dir, canonical, str(KB_ID))
    table_evidence = _table_evidence_id(test_data_dir)
    image_evidence = "evidence_v1_" + "c" * 64
    result = SimpleNamespace(
        items=[
            _unified_candidate(
                evidence_id=image_evidence,
                modality="image",
                source_block_ids=["p1-b2"],
                asset_refs=[IMAGE_ASSET_REF],
            ),
            _unified_candidate(
                evidence_id=table_evidence,
                modality="table",
                source_block_ids=["p1-b3"],
                asset_refs=[TABLE_ASSET_REF],
            ),
        ],
        total_latency_ms=8.0,
        degraded=False,
        branches=[SimpleNamespace(branch="multimodal", status="success")],
    )

    class _FakeUnified:
        async def search(self, *args: object, **kwargs: object) -> Any:
            return result

    async def loader(knowledge_base_id: UUID, document_id: UUID) -> Document | None:
        return _document(document_id, knowledge_base_id)

    tool = AgentRetrievalTool(
        _settings(data_dir=test_data_dir),
        _FakeUnified(),  # type: ignore[arg-type]
        document_loader=loader,
        result_limit=4,
    )
    retrieval = anyio.run(tool.retrieve, "问题", KB_ID)
    assert [candidate.marker for candidate in retrieval.candidates] == ["C1", "C2"]
    assert retrieval.candidates[1].table_columns == ("年份", "GDP")
    assert retrieval.candidates[1].table_issue is None
    assert "[C1]" in retrieval.context_text
    assert "[C2]" in retrieval.context_text
    assert "--" not in retrieval.context_text


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


class _FakeVisionService:
    def __init__(self, result: VisionAnalysisResult | Exception) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    async def analyze(self, document: Document, **kwargs: Any) -> VisionAnalysisResult:
        self.calls.append({"document": document, **kwargs})
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeTableService:
    def __init__(self, result: TableAnalysisResult | Exception) -> None:
        self._result = result
        self.requests: list[Any] = []

    def analyze(self, request: Any, **kwargs: Any) -> TableAnalysisResult:
        self.requests.append(request)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _executor(
    *,
    vision: _FakeVisionService | None = None,
    table: _FakeTableService | None = None,
) -> AgentToolExecutor:
    async def loader(knowledge_base_id: UUID, document_id: UUID) -> Document | None:
        return _document(document_id, knowledge_base_id)

    return AgentToolExecutor(
        _settings(),
        vision_service=(vision or _FakeVisionService(_vision_ok())),  # type: ignore[arg-type]
        table_service=(table or _FakeTableService(_table_ok())),  # type: ignore[arg-type]
        document_loader=loader,
    )


def _vision_ok() -> VisionAnalysisResult:
    return VisionAnalysisResult(
        status="ok",
        task_type="describe_image",
        answer="图片展示两个空位。",
        provider="fake-vlm",
        model="fake-vlm",
        latency_ms=2.0,
    )


def _table_ok() -> TableAnalysisResult:
    return TableAnalysisResult(
        operation="growth",
        status="ok",
        rows=[],
        formula="((4760 - 4300) / 4300) × 100% = 10.6977%",
    )


def test_executor_returns_completed_vision_outcome() -> None:
    outcome = anyio.run(
        _executor().analyze_image,
        _candidate(1, marker="C1"),
        AgentPlan(
            action="call_tool",
            tool="analyze_image",
            evidence_index=1,
            vision_task="describe_image",
        ),
    )
    assert outcome.status == "completed"
    assert outcome.payload["provider"] == "fake-vlm"


def test_executor_returns_needs_review_vision_outcome() -> None:
    result = VisionAnalysisResult(
        status="needs_review",
        task_type="extract_table",
        extracted_table=None,
        warnings=["视觉模型提取，数值需核对"],
        error_code=None,
    )
    outcome = anyio.run(
        _executor(vision=_FakeVisionService(result)).analyze_image,
        _candidate(1, marker="C1", modality="image"),
        AgentPlan(
            action="call_tool",
            tool="analyze_image",
            evidence_index=1,
            vision_task="extract_table",
        ),
    )
    assert outcome.status == "completed"
    assert "人工核对" in outcome.summary


def test_executor_maps_vision_failure_to_controlled_outcome() -> None:
    from knowledge_scope.vision.errors import VisionProviderError

    outcome = anyio.run(
        _executor(
            vision=_FakeVisionService(VisionProviderError("视觉模型返回 HTTP 503"))
        ).analyze_image,
        _candidate(1, marker="C1"),
        AgentPlan(
            action="call_tool",
            tool="analyze_image",
            evidence_index=1,
            vision_task="describe_image",
        ),
    )
    assert outcome.status == "failed"
    assert outcome.error_code == "provider_error"


def test_executor_rejects_invalid_table_parameters_before_calculation() -> None:
    table = _FakeTableService(_table_ok())
    outcome = anyio.run(
        _executor(table=table).analyze_table,
        _candidate(1, marker="C1", modality="table", evidence_id="evidence_v1_" + "d" * 64),
        AgentPlan(
            action="call_tool",
            tool="analyze_table",
            evidence_index=1,
            operation="growth",
            parameters={"axis": "rows"},
        ),
    )
    assert outcome.status == "failed"
    assert outcome.error_code == "invalid_parameters"
    assert table.requests == []


def test_executor_maps_not_computable_table_result_to_failure() -> None:
    result = TableAnalysisResult(
        operation="select",
        status="not_computable",
        error_code="table_only_image",
        message="该表格证据仅保留图片素材，当前版本不支持图片表格计算",
    )
    outcome = anyio.run(
        _executor(table=_FakeTableService(result)).analyze_table,
        _candidate(1, marker="C1", modality="table", evidence_id="evidence_v1_" + "d" * 64),
        AgentPlan(
            action="call_tool",
            tool="analyze_table",
            evidence_index=1,
            operation="select",
        ),
    )
    assert outcome.status == "failed"
    assert outcome.error_code == "table_only_image"


# ---------------------------------------------------------------------------
# Agent service orchestration
# ---------------------------------------------------------------------------


class _StubRetrieval:
    def __init__(self, result: Any) -> None:
        self._result = result
        self.calls: list[tuple[str, UUID]] = []

    async def retrieve(
        self,
        query: str,
        knowledge_base_id: UUID,
        *,
        document_id: UUID | None = None,
    ) -> Any:
        self.calls.append((query, knowledge_base_id))
        return self._result


class _StubExecutor:
    def __init__(self, outcomes: list[AgentToolOutcome]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[str, int]] = []

    async def analyze_image(self, candidate: AgentCandidate, plan: AgentPlan) -> AgentToolOutcome:
        self.calls.append(("analyze_image", candidate.index))
        return self._outcomes.pop(0)

    async def analyze_table(
        self,
        candidate: AgentCandidate,
        plan: AgentPlan,
        second_candidate: AgentCandidate | None = None,
    ) -> AgentToolOutcome:
        self.calls.append(("analyze_table", candidate.index))
        return self._outcomes.pop(0)


def _completed_outcome(tool: str = "analyze_image") -> AgentToolOutcome:
    return AgentToolOutcome(
        tool=tool,  # type: ignore[arg-type]
        status="completed",
        summary="图片理解完成（fake-vlm）",
        payload={"status": "ok"},
    )


async def _no_document(knowledge_base_id: UUID, document_id: UUID) -> Document | None:
    return None


def _service(
    *,
    gateway: _FakeGateway,
    retrieval: Any,
    executor: Any,
    rag_service: Any = None,
    settings: Settings | None = None,
    history_compressor: Any = None,
    query_rewrite: Any = None,
) -> AgentChatService:
    return _StubAgentService(
        settings or _settings(),
        gateway=gateway,
        retrieval=retrieval,
        executor=executor,
        rag_service=rag_service,
        history_compressor=history_compressor,
        query_rewrite=query_rewrite,
    )


def test_agent_service_emits_visible_tool_status_and_answer_citations() -> None:
    gateway = _FakeGateway(
        plans=[
            {
                "action": "call_tool",
                "reason": "需要看图",
                "tool": "analyze_image",
                "evidence_index": 1,
                "vision_task": "describe_image",
            },
            {"action": "final", "reason": "证据足够"},
        ],
        deltas=["图片", "展示了两个空位。"],
    )
    candidates = (
        _candidate(1, marker="C1"),
        _candidate(2, marker="C2", evidence_id="evidence_v1_" + "e" * 64),
    )
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result(list(candidates))),
        executor=_StubExecutor([_completed_outcome()]),
    )
    request = RAGQueryRequest(
        query="这张图展示了什么？",
        knowledge_base_id=KB_ID,
        answer_mode="agent",
    )
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))

    names = [event.event for event in events]
    assert names[0] == "tool_status"
    assert names[1] == "tool_status"
    assert names[2] == "tool_status"
    assert names[3] == "tool_status"
    assert "answer_delta" in names
    assert names[-2] == "citations"
    assert names[-1] == "complete"
    statuses = [event.data for event in events if event.event == "tool_status"]
    assert statuses[0]["tool"] == "retrieval"
    assert statuses[0]["status"] == "running"
    assert statuses[1]["status"] == "completed"
    assert statuses[2]["tool"] == "analyze_image"
    assert statuses[2]["status"] == "running"
    assert statuses[3]["status"] == "completed"
    citation_event = next(event for event in events if event.event == "citations")
    # C1 is both cited by the answer and consumed by the tool; C2 was neither.
    assert [item["marker"] for item in citation_event.data["items"]] == ["C1"]
    complete = next(event for event in events if event.event == "complete")
    assert complete.data["status"] == "completed"
    assert complete.data["answer_mode"] == "agent"
    assert complete.data["tool_calls"] == 1
    assert complete.data["route"] == "unified"
    assert complete.data["provider"] == "fake"
    assert complete.data["agent_degraded"] is False
    assert [item["marker"] for item in complete.data["agent_candidates"]] == ["C1", "C2"]
    assert complete.data["agent_tool_trace"][0]["tool"] == "retrieval"
    assert complete.data["agent_tool_trace"][1]["tool"] == "analyze_image"


def test_agent_service_falls_back_to_rag_when_planner_fails() -> None:
    class _FailingPlanner:
        async def plan(self, *args: object, **kwargs: object) -> AgentPlan:
            raise AgentPlanningError("规划结果不是可解析的 JSON")

    gateway = _FakeGateway(plans=[], deltas=[])
    maybe_service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result([_candidate(1, marker="C1")])),
        executor=_StubExecutor([]),
        rag_service=_FakeRAGService(),
    )
    maybe_service._planner = _FailingPlanner()  # type: ignore[assignment]
    request = RAGQueryRequest(query="问题", knowledge_base_id=KB_ID, answer_mode="agent")
    events = anyio.run(
        lambda: _collect(maybe_service.stream(request, document_loader=_no_document))
    )

    names = [event.event for event in events]
    assert names[0] == "tool_status"
    assert names[1] == "tool_status"  # retrieval completed before planning
    statuses = [event.data for event in events if event.event == "tool_status"]
    assert [item["status"] for item in statuses] == ["running", "completed"]
    assert "answer_delta" in names
    first_delta = next(event for event in events if event.event == "answer_delta")
    assert "未执行视觉分析或确定性计算" in first_delta.data["text"]
    complete = next(event for event in events if event.event == "complete")
    assert complete.data["answer_mode"] == "agent"
    assert complete.data["tool_calls"] == 0
    assert complete.data["agent_degraded"] is True
    assert complete.data["agent_degraded_reason"] == "工具规划失败"
    assert complete.data["agent_tool_trace"][0]["tool"] == "retrieval"


def test_agent_service_skips_repeated_identical_tool_call() -> None:
    gateway = _FakeGateway(
        plans=[
            {
                "action": "call_tool",
                "tool": "analyze_image",
                "evidence_index": 1,
                "vision_task": "describe_image",
            },
            {
                "action": "call_tool",
                "tool": "analyze_image",
                "evidence_index": 1,
                "vision_task": "describe_image",
            },
            {"action": "final", "reason": "ok"},
        ],
        deltas=["完成"],
    )
    executor = _StubExecutor([_completed_outcome()])
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result([_candidate(1, marker="C1")])),
        executor=executor,
    )
    request = RAGQueryRequest(query="问题", knowledge_base_id=KB_ID, answer_mode="agent")
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))

    skipped = [
        event.data
        for event in events
        if event.event == "tool_status" and event.data["status"] == "skipped"
    ]
    assert skipped and skipped[-1]["tool"] == "analyze_image"
    complete = next(event for event in events if event.event == "complete")
    assert complete.data["tool_calls"] == 1
    assert executor.calls == [("analyze_image", 1)]


class _FailAfterFirstPlanner:
    def __init__(self) -> None:
        self.calls = 0

    async def plan(self, *args: object, **kwargs: object) -> AgentPlan:
        self.calls += 1
        if self.calls == 1:
            return AgentPlan(
                action="call_tool",
                tool="analyze_image",
                evidence_index=1,
                vision_task="describe_image",
            )
        raise AgentPlanningError("规划结果不符合工具调用约定")


def test_agent_service_strips_invalid_citation_markers() -> None:
    gateway = _FakeGateway(
        plans=[
            {
                "action": "call_tool",
                "tool": "analyze_image",
                "evidence_index": 1,
                "vision_task": "describe_image",
            },
            {"action": "final", "reason": "证据足够"},
        ],
        deltas=["图片[C9]展示", "了两个空位。[C1]"],
    )
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result([_candidate(1, marker="C1")])),
        executor=_StubExecutor([_completed_outcome()]),
    )
    request = RAGQueryRequest(query="图里有什么？", knowledge_base_id=KB_ID, answer_mode="agent")
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))

    answer = "".join(event.data["text"] for event in events if event.event == "answer_delta")
    assert "[C9]" not in answer
    assert "[C1]" in answer
    citation_event = next(event for event in events if event.event == "citations")
    assert [item["marker"] for item in citation_event.data["items"]] == ["C1"]


def test_agent_service_filters_unused_candidates_from_citations() -> None:
    candidates = (
        _candidate(1, marker="C1"),
        _candidate(2, marker="C2", evidence_id="evidence_v1_" + "e" * 64),
    )
    gateway = _FakeGateway(
        plans=[{"action": "final", "reason": "直接回答"}], deltas=["[C2] 是相关证据。"]
    )
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result(list(candidates))),
        executor=_StubExecutor([]),
    )
    request = RAGQueryRequest(query="问题", knowledge_base_id=KB_ID, answer_mode="agent")
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))

    citation_event = next(event for event in events if event.event == "citations")
    assert [item["marker"] for item in citation_event.data["items"]] == ["C2"]
    complete = next(event for event in events if event.event == "complete")
    assert complete.data["tool_calls"] == 0
    assert complete.data["agent_degraded"] is False


def test_agent_service_fallback_after_tool_keeps_tool_count() -> None:
    gateway = _FakeGateway(plans=[], deltas=[])
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result([_candidate(1, marker="C1")])),
        executor=_StubExecutor([_completed_outcome()]),
        rag_service=_FakeRAGService(),
    )
    service._planner = _FailAfterFirstPlanner()  # type: ignore[assignment]
    request = RAGQueryRequest(query="问题", knowledge_base_id=KB_ID, answer_mode="agent")
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))

    complete = next(event for event in events if event.event == "complete")
    assert complete.data["answer_mode"] == "agent"
    assert complete.data["tool_calls"] == 1
    assert complete.data["agent_degraded"] is True
    trace = complete.data["agent_tool_trace"]
    assert [item["tool"] for item in trace] == ["retrieval", "analyze_image"]
    assert trace[1]["status"] == "completed"


class _StubHistory:
    async def prepare(self, history: tuple[RAGHistoryTurn, ...]) -> HistoryPlan:
        return HistoryPlan(
            summary="用户问过越南GDP",
            recent_turns=(RAGHistoryTurn(role="user", content="越南 2023 年 GDP 是多少？"),),
            compressed_turns=3,
            cache_hit=True,
        )


class _StubRewrite:
    def __init__(self, rewritten: str) -> None:
        self._rewritten = rewritten

    async def rewrite(
        self,
        query: str,
        history: tuple[RAGHistoryTurn, ...],
    ) -> str | None:
        return self._rewritten


def test_agent_service_uses_rewritten_query_and_history() -> None:
    gateway = _FakeGateway(
        plans=[{"action": "final", "reason": "直接回答"}], deltas=["4300 亿美元。"]
    )
    retrieval = _StubRetrieval(_retrieval_result([_candidate(1, marker="C1", modality="text")]))
    service = _service(
        gateway=gateway,
        retrieval=retrieval,
        executor=_StubExecutor([]),
        history_compressor=_StubHistory(),
        query_rewrite=_StubRewrite("2024年越南GDP"),
    )
    request = RAGQueryRequest(
        query="那 2024 年呢？",
        knowledge_base_id=KB_ID,
        answer_mode="agent",
        history=(
            RAGHistoryTurn(role="user", content="2023 年越南的 GDP 是多少？"),
            RAGHistoryTurn(role="assistant", content="4300 亿美元。"),
        ),
    )
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))

    assert retrieval.calls[0][0] == "2024年越南GDP"
    complete = next(event for event in events if event.event == "complete")
    assert complete.data["rewritten_query"] == "2024年越南GDP"
    assert complete.data["history_compressed_turns"] == 3
    assert complete.data["history_cache_hit"] is True
    assert complete.data["history_summary_used"] is True
    planner_contents = [
        message.content for request_ in gateway.requests for message in request_.messages
    ]
    assert any("较早对话摘要：用户问过越南GDP" in content for content in planner_contents)


def test_agent_service_falls_back_when_retrieval_returns_no_candidates() -> None:
    gateway = _FakeGateway(plans=[], deltas=[])
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result([])),
        executor=_StubExecutor([]),
        rag_service=_FakeRAGService(),
    )
    request = RAGQueryRequest(query="问题", knowledge_base_id=KB_ID, answer_mode="agent")
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))

    names = [event.event for event in events]
    assert names[:2] == ["tool_status", "tool_status"]  # running + skipped fallback
    assert "answer_delta" in names
    complete = next(event for event in events if event.event == "complete")
    assert complete.data["answer_mode"] == "agent"


# ---------------------------------------------------------------------------
# API routing (agent mode)
# ---------------------------------------------------------------------------


class _FakeRAGService:
    async def stream(self, payload: object) -> AsyncIterator[RAGStreamEvent]:
        yield RAGStreamEvent(event="answer_delta", data={"text": "常规回答"})
        yield RAGStreamEvent(
            event="citations",
            data=RAGCitationsData(prompt_version="rag-qa-v6", items=[]).model_dump(mode="json"),
        )
        yield RAGStreamEvent(
            event="complete",
            data=RAGCompleteData(
                status="completed",
                prompt_version="rag-qa-v6",
                retrieval_mode="dense",
                route="dense",
                latency_ms=1.0,
            ).model_dump(mode="json"),
        )


class _FakeAgentAPIService:
    def __init__(self) -> None:
        self.payloads: list[RAGQueryRequest] = []
        self.loader_created = False

    async def stream(
        self,
        payload: RAGQueryRequest,
        *,
        document_loader: Any,
    ) -> AsyncIterator[RAGStreamEvent]:
        self.payloads.append(payload)
        self.loader_created = document_loader is not None
        yield RAGStreamEvent(
            event="tool_status",
            data=RAGToolStatusData(
                tool="retrieval",
                status="running",
                summary="正在检索知识库",
                step=1,
            ).model_dump(mode="json"),
        )
        yield RAGStreamEvent(event="answer_delta", data={"text": "智能体回答"})
        yield RAGStreamEvent(
            event="citations",
            data=RAGCitationsData(
                prompt_version="agent-rag-v1",
                items=[_citation("C1")],
            ).model_dump(mode="json"),
        )
        yield RAGStreamEvent(
            event="complete",
            data=RAGCompleteData(
                status="completed",
                prompt_version="agent-rag-v1",
                retrieval_mode="unified",
                route="unified",
                answer_mode="agent",
                tool_calls=0,
                latency_ms=2.0,
            ).model_dump(mode="json"),
        )


def _sse_frames(response: httpx.Response) -> list[str]:
    return [frame.splitlines()[0] for frame in response.text.split("\n\n") if frame]


def test_agent_api_requires_knowledge_base() -> None:
    with pytest.raises(ValidationError, match="agent"):
        RAGQueryRequest(query="问题", answer_mode="agent")


def test_agent_api_returns_503_when_service_missing() -> None:
    settings = _settings(agent_enabled=False)
    application = create_app(
        settings,
        vector_store=object(),  # type: ignore[arg-type]
        rag_service=_FakeRAGService(),  # type: ignore[arg-type]
    )

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/api/v1/rag/query",
                json={
                    "query": "问题",
                    "knowledge_base_id": str(KB_ID),
                    "answer_mode": "agent",
                },
            )

    response = anyio.run(request)
    assert response.status_code == 503
    assert response.json()["detail"] == "Agent service is not initialized"


@pytest.mark.anyio
async def test_agent_api_routes_agent_mode_to_agent_service(agent_client_factory: Any) -> None:
    service = _FakeAgentAPIService()

    async with agent_client_factory(service) as client:
        response = await client.post(
            "/api/v1/rag/query",
            json={
                "query": "问题",
                "knowledge_base_id": str(KB_ID),
                "answer_mode": "agent",
            },
        )
    assert response.status_code == 200
    assert _sse_frames(response) == [
        "event: tool_status",
        "event: answer_delta",
        "event: citations",
        "event: complete",
    ]
    assert service.payloads[0].answer_mode == "agent"
    assert service.loader_created is True


class _SlowAgentAPIService:
    def __init__(self) -> None:
        self.payloads: list[RAGQueryRequest] = []

    async def stream(
        self, payload: RAGQueryRequest, *, document_loader: Any
    ) -> AsyncIterator[RAGStreamEvent]:
        self.payloads.append(payload)
        await anyio.sleep(0.2)
        yield RAGStreamEvent(event="answer_delta", data={"text": "迟到回答"})


@pytest.mark.anyio
async def test_agent_api_times_out_stalled_turn(
    agent_client_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from knowledge_scope.api import rag as rag_api

    monkeypatch.setattr(rag_api, "_agent_timeout", lambda settings: 0.02)
    service = _SlowAgentAPIService()
    async with agent_client_factory(service) as client:
        response = await client.post(
            "/api/v1/rag/query",
            json={
                "query": "问题",
                "knowledge_base_id": str(KB_ID),
                "answer_mode": "agent",
            },
        )
    assert response.status_code == 200
    frames = _sse_frames(response)
    assert frames[0] == "event: error"
    assert "智能体执行超过" in response.text
    assert frames[-1] == "event: complete"
    assert '"answer_mode":"agent"' in response.text


def _service_table_outcome() -> AgentToolOutcome:
    return AgentToolOutcome(
        tool="analyze_table",
        status="completed",
        summary="表格计算完成（growth）",
        payload={
            "status": "ok",
            "operation": "growth",
            "columns": [
                {"name": "指标"},
                {"name": "基期值"},
                {"name": "报告期值"},
                {"name": "绝对变化"},
                {"name": "相对变化"},
            ],
            "rows": [{"cells": ["越南", "4300", "4760", "460", "10.6977"]}],
            "formula": "((4760 - 4300) / 4300) × 100% = 10.6977%",
            "units": ["亿美元", "%"],
            "sources": [
                {
                    "document_id": str(DOC_ID),
                    "evidence_id": "evidence_v1_" + "c" * 64,
                    "page_start": 1,
                    "page_end": 1,
                    "source_block_ids": ["p1-b3"],
                }
            ],
        },
        fact_status="source_verified",
    )


def _service_image_outcome() -> AgentToolOutcome:
    return AgentToolOutcome(
        tool="analyze_image",
        status="completed",
        summary="图表分析完成（fake-vlm）",
        payload={
            "status": "ok",
            "answer": "Q1 最低约 120，Q4 最高约 210",
            "fact_status": "visual_estimate",
            "sources": [
                {
                    "knowledge_base_id": str(KB_ID),
                    "document_id": str(DOC_ID),
                    "evidence_id": "evidence_v1_" + "d" * 64,
                    "page_start": 1,
                    "page_end": 1,
                    "source_block_ids": ["p1-b2"],
                }
            ],
        },
        fact_status="visual_estimate",
        numeric_claims=[{"value": "120", "fact_status": "visual_estimate"}],
    )


def test_mixed_table_and_visual_tools_gate_unverified_visual_numbers() -> None:
    table_candidate = _candidate(
        1,
        marker="C1",
        modality="table",
        evidence_id="evidence_v1_" + "c" * 64,
        table_columns=("年份", "GDP"),
    )
    image_candidate = _candidate(
        2,
        marker="C2",
        modality="image",
        evidence_id="evidence_v1_" + "d" * 64,
    )
    gateway = _FakeGateway(
        plans=[
            {
                "action": "call_tool",
                "tool": "analyze_table",
                "evidence_index": 1,
                "operation": "growth",
                "parameters": {
                    "axis": "columns",
                    "row": 0,
                    "base_column": 1,
                    "target_column": 2,
                },
            },
            {
                "action": "call_tool",
                "tool": "analyze_image",
                "evidence_index": 2,
                "vision_task": "describe_image",
            },
            {"action": "final", "reason": "证据足够"},
        ],
        deltas=["Q1 最低销售额为 120 万元。", "表格增长率为 10.6977%。"],
    )
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result([table_candidate, image_candidate])),
        executor=_StubExecutor([_service_table_outcome(), _service_image_outcome()]),
    )
    request = RAGQueryRequest(
        query="计算表格中的增长率，并给出柱状图最低季度的具体金额。",
        knowledge_base_id=KB_ID,
        answer_mode="agent",
    )
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))

    answer = "".join(event.data["text"] for event in events if event.event == "answer_delta")
    assert "120" not in answer
    assert "10.6977" in answer
    assert "未经核验" in answer
    complete = next(event for event in events if event.event == "complete")
    assert complete.data["safety_gate"] == "visual_numeric_withheld"
    assert complete.data["fact_status"] in {"visual_estimate", "unverified"}
    assert len(complete.data["visual_sources"]) == 1


def test_table_only_high_risk_answer_replaces_unverified_numbers() -> None:
    table_candidate = _candidate(
        1,
        marker="C1",
        modality="table",
        evidence_id="evidence_v1_" + "c" * 64,
        table_columns=("年份", "GDP"),
    )
    gateway = _FakeGateway(
        plans=[
            {
                "action": "call_tool",
                "tool": "analyze_table",
                "evidence_index": 1,
                "operation": "growth",
                "parameters": {
                    "axis": "columns",
                    "row": 0,
                    "base_column": 1,
                    "target_column": 2,
                },
            },
            {"action": "final", "reason": "证据足够"},
        ],
        deltas=["表格中的增长率是 99.9%。"],
    )
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result([table_candidate])),
        executor=_StubExecutor([_service_table_outcome()]),
    )
    request = RAGQueryRequest(
        query="表格中的增长率是多少？请给出精确值。",
        knowledge_base_id=KB_ID,
        answer_mode="agent",
    )
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))

    answer = "".join(event.data["text"] for event in events if event.event == "answer_delta")
    assert "99.9" not in answer
    assert "10.6977" in answer
    complete = next(event for event in events if event.event == "complete")
    assert complete.data["fact_status"] == "source_verified"
    assert complete.data["safety_gate"] == "none"
    assert complete.data["finish_reason"] == "deterministic_table"
    assert complete.data["provider"] is None


# ---------------------------------------------------------------------------
# Visual execution provenance (multi-image, asset index, failure states)
# ---------------------------------------------------------------------------


def _visual_source(
    *,
    evidence_id: str,
    document_id: UUID = DOC_ID,
    page: int = 1,
    blocks: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "knowledge_base_id": str(KB_ID),
        "document_id": str(document_id),
        "evidence_id": evidence_id,
        "page_start": page,
        "page_end": page,
        "source_block_ids": blocks or ["p1-b2"],
    }


def _analyze_image_outcome(
    *,
    status: str = "completed",
    asset_index: int = 0,
    vision_task: str = "describe_image",
    sources: list[dict[str, Any]] | None = None,
    answer: str = "柱状图展示 2023 年季度销售额，Q4 最高。",
    fact_status: str | None = "visual_estimate",
) -> AgentToolOutcome:
    payload: dict[str, Any] = {
        "status": "ok" if status == "completed" else "failed",
        "answer": answer,
        "sources": sources or [],
        "asset_index": asset_index,
        "vision_task": vision_task,
        "execution_status": status,
    }
    if status != "completed":
        payload["error_code"] = "provider_connection"
        payload["message"] = "无法连接视觉模型服务"
    return AgentToolOutcome(
        tool="analyze_image",
        status=status,  # type: ignore[arg-type]
        summary="图片理解完成（fake-vlm）" if status == "completed" else "视觉工具执行失败",
        payload=payload,
        error_code=None if status == "completed" else "provider_connection",
        error_message=None if status == "completed" else "无法连接视觉模型服务",
        fact_status=fact_status if status == "completed" else "not_available",
    )


def _visual_plan(
    evidence_index: int,
    *,
    asset_index: int = 0,
    task: str = "describe_image",
) -> dict[str, Any]:
    return {
        "action": "call_tool",
        "reason": "需要看图",
        "tool": "analyze_image",
        "evidence_index": evidence_index,
        "asset_index": asset_index,
        "vision_task": task,
    }


def _visual_sources_complete(events: list[RAGStreamEvent]) -> dict[str, Any]:
    complete = next(event for event in events if event.event == "complete")
    return complete.data


def test_agent_service_reports_every_used_image_without_asset_collapse() -> None:
    evidence_one = "evidence_v1_" + "1" * 64
    evidence_two = "evidence_v1_" + "2" * 64
    candidates = (
        _candidate(1, marker="C1", evidence_id=evidence_one),
        _candidate(2, marker="C2", evidence_id=evidence_two),
    )
    gateway = _FakeGateway(
        plans=[
            _visual_plan(1, asset_index=1, task="describe_image"),
            _visual_plan(2, asset_index=0, task="analyze_chart"),
            {"action": "final", "reason": "证据足够"},
        ],
        deltas=["图 1 的柱状图显示 Q4 最高。"],
    )
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result(list(candidates))),
        executor=_StubExecutor(
            [
                _analyze_image_outcome(
                    asset_index=1, sources=[_visual_source(evidence_id=evidence_one)]
                ),
                _analyze_image_outcome(
                    asset_index=0,
                    vision_task="analyze_chart",
                    sources=[_visual_source(evidence_id=evidence_two)],
                ),
            ]
        ),
    )
    request = RAGQueryRequest(
        query="对比图 1 和图 2，说明 2023 年季度销售额的变化趋势。",
        knowledge_base_id=KB_ID,
        answer_mode="agent",
    )
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))
    complete = _visual_sources_complete(events)
    sources = complete["visual_sources"]
    assert len(sources) == 2
    assert [item["asset_index"] for item in sources] == [1, 0]
    assert [item["evidence_id"] for item in sources] == [evidence_one, evidence_two]
    assert all(item["execution_status"] == "completed" for item in sources)
    trace = complete["agent_tool_trace"]
    analyze_entries = [item for item in trace if item["tool"] == "analyze_image"]
    assert [item["parameters"]["asset_index"] for item in analyze_entries] == [1, 0]
    assert [item["parameters"]["vision_task"] for item in analyze_entries] == [
        "describe_image",
        "analyze_chart",
    ]


def test_agent_service_keeps_two_asset_indices_of_the_same_evidence() -> None:
    evidence_id = "evidence_v1_" + "3" * 64
    candidate = _candidate(1, marker="C1", evidence_id=evidence_id)
    gateway = _FakeGateway(
        plans=[
            _visual_plan(1, asset_index=0),
            _visual_plan(1, asset_index=1),
            {"action": "final", "reason": "证据足够"},
        ],
        deltas=["两张素材都已分析。"],
    )
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result([candidate])),
        executor=_StubExecutor(
            [
                _analyze_image_outcome(
                    asset_index=0, sources=[_visual_source(evidence_id=evidence_id)]
                ),
                _analyze_image_outcome(
                    asset_index=1, sources=[_visual_source(evidence_id=evidence_id)]
                ),
            ]
        ),
    )
    request = RAGQueryRequest(
        query="图 1 中最低季度的具体销售额是多少？",
        knowledge_base_id=KB_ID,
        answer_mode="agent",
    )
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))
    complete = _visual_sources_complete(events)
    sources = complete["visual_sources"]
    assert [item["asset_index"] for item in sources] == [0, 1]
    assert all(item["evidence_id"] == evidence_id for item in sources)


def test_agent_service_marks_failed_visual_execution_separately() -> None:
    first = "evidence_v1_" + "4" * 64
    second = "evidence_v1_" + "5" * 64
    candidates = (
        _candidate(1, marker="C1", evidence_id=first),
        _candidate(2, marker="C2", evidence_id=second),
    )
    gateway = _FakeGateway(
        plans=[
            _visual_plan(1, asset_index=0),
            _visual_plan(2, asset_index=0),
            {"action": "final", "reason": "证据足够"},
        ],
        deltas=["Q1 销售额为 120 万元。"],
    )
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result(list(candidates))),
        executor=_StubExecutor(
            [
                _analyze_image_outcome(asset_index=0, sources=[_visual_source(evidence_id=first)]),
                _analyze_image_outcome(
                    status="failed", asset_index=0, sources=[_visual_source(evidence_id=second)]
                ),
            ]
        ),
    )
    request = RAGQueryRequest(
        query="图 1 中最低季度的具体销售额是多少？请给出精确数值。",
        knowledge_base_id=KB_ID,
        answer_mode="agent",
    )
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))
    answer = "".join(event.data["text"] for event in events if event.event == "answer_delta")
    complete = _visual_sources_complete(events)
    sources = complete["visual_sources"]
    assert "120" not in answer
    assert [item["execution_status"] for item in sources] == ["completed", "failed"]
    assert sources[1]["fact_status"] == "not_available"
    assert complete["safety_gate"] == "visual_numeric_withheld"


def test_agent_service_vision_failure_does_not_fall_back_to_number_guess() -> None:
    evidence_id = "evidence_v1_" + "6" * 64
    candidate = _candidate(1, marker="C1", evidence_id=evidence_id)
    gateway = _FakeGateway(
        plans=[
            _visual_plan(1),
            {"action": "final", "reason": "证据足够"},
        ],
        deltas=["Q1 销售额为 120 万元。"],
    )
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result([candidate])),
        executor=_StubExecutor(
            [
                _analyze_image_outcome(
                    status="failed", sources=[_visual_source(evidence_id=evidence_id)]
                )
            ]
        ),
    )
    request = RAGQueryRequest(
        query="图 1 中最低季度的具体销售额是多少？",
        knowledge_base_id=KB_ID,
        answer_mode="agent",
    )
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))
    answer = "".join(event.data["text"] for event in events if event.event == "answer_delta")
    complete = _visual_sources_complete(events)
    assert "120" not in answer
    assert complete["safety_gate"] == "vision_unavailable"
    assert complete["fact_status"] == "not_available"
    assert complete["visual_sources"][0]["execution_status"] == "failed"


# ---------------------------------------------------------------------------
# M9 evidence sufficiency, supplementary retrieval, and visual auto-call
# ---------------------------------------------------------------------------


def _candidate_with_snippet(
    index: int,
    snippet: str,
    *,
    modality: str = "table",
    document_id: UUID = DOC_ID,
    evidence_id: str | None = None,
    table_rows: tuple[str, ...] = (),
    table_columns: tuple[str, ...] = (),
    marker: str | None = None,
) -> AgentCandidate:
    base = _candidate(
        index,
        marker=marker or f"C{index}",
        modality=modality,
        document_id=document_id,
        evidence_id=evidence_id,
        table_columns=table_columns,
    )
    return AgentCandidate(
        index=base.index,
        marker=base.marker,
        citation=base.citation.model_copy(update={"snippet": snippet}),
        document_title=base.document_title,
        table_columns=table_columns,
        table_rows=table_rows,
        table_issue=base.table_issue,
    )


class _SequencedRetrieval:
    def __init__(self, results: list[Any], *, fail_after: int | None = None) -> None:
        self._results = list(results)
        self._fail_after = fail_after
        self.calls: list[tuple[str, UUID, UUID | None]] = []

    async def retrieve(
        self,
        query: str,
        knowledge_base_id: UUID,
        *,
        document_id: UUID | None = None,
    ) -> Any:
        self.calls.append((query, knowledge_base_id, document_id))
        if self._fail_after is not None and len(self.calls) > self._fail_after:
            raise RuntimeError("supplementary retrieval exploded")
        index = min(len(self.calls) - 1, len(self._results) - 1)
        return self._results[index]


def test_m9_supplementary_retrieval_runs_when_comparison_evidence_is_missing() -> None:
    initial = (
        _candidate_with_snippet(
            1,
            "2024年 越南 GDP 4760 亿美元",
            table_rows=("越南", "4760"),
            table_columns=("国家", "2024年GDP"),
        ),
    )
    extra = (
        _candidate_with_snippet(
            1,
            "2023年 泰国 GDP 5000 亿美元 2024年 5300 亿美元",
            document_id=SECOND_DOC_ID,
            evidence_id="evidence_v1_" + "f" * 64,
            table_rows=("泰国", "5300"),
            table_columns=("国家", "2023年GDP", "2024年GDP"),
        ),
    )
    retrieval = _SequencedRetrieval(
        [_retrieval_result(list(initial)), _retrieval_result(list(extra))]
    )
    gateway = _FakeGateway(
        plans=[{"action": "final", "reason": "基于证据回答"}],
        deltas=["越南与泰国数据见引用。"],
    )
    service = _service(
        gateway=gateway,
        retrieval=retrieval,
        executor=_StubExecutor([]),
        settings=_settings(
            agent_evidence_sufficiency_enabled=True,
            agent_max_rounds=1,
            agent_supplementary_retrieval_max=1,
        ),
    )
    request = RAGQueryRequest(
        query="比较越南和泰国 2023—2024 年 GDP。",
        knowledge_base_id=KB_ID,
        answer_mode="agent",
    )
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))

    assert len(retrieval.calls) == 2
    assert retrieval.calls[1][0] != request.query
    assert "泰国" in retrieval.calls[1][0]
    retrieval_statuses = [
        event.data["status"]
        for event in events
        if event.event == "tool_status" and event.data["tool"] == "retrieval"
    ]
    assert retrieval_statuses == ["running", "completed", "running", "completed"]
    complete = next(event for event in events if event.event == "complete")
    assert complete.data["agent_supplementary_retrieval"] is True
    assert complete.data["agent_supplementary_added"] == 1
    assert complete.data["agent_supplementary_error"] is None
    assert complete.data["agent_evidence_satisfied"] is True
    assert complete.data["agent_evidence_missing"] == []
    assert complete.data["agent_evidence_coverage_before"]["missing"] >= 1
    assert complete.data["agent_evidence_coverage_after"]["missing"] == 0
    assert "泰国" in " ".join(complete.data["agent_evidence_required"])
    markers = [item["marker"] for item in complete.data["agent_candidates"]]
    assert markers == ["C1", "C2"]


def test_m9_sufficient_simple_question_skips_supplementary_retrieval() -> None:
    candidates = (
        _candidate_with_snippet(
            1,
            "越南 2024年 GDP 4760 亿美元",
            table_rows=("越南", "4760"),
            table_columns=("国家", "2024年GDP"),
        ),
    )
    retrieval = _SequencedRetrieval([_retrieval_result(list(candidates))])
    gateway = _FakeGateway(plans=[{"action": "final"}], deltas=["4760 亿美元。"])
    service = _service(
        gateway=gateway,
        retrieval=retrieval,
        executor=_StubExecutor([]),
        settings=_settings(
            agent_evidence_sufficiency_enabled=True,
            agent_max_rounds=1,
        ),
    )
    request = RAGQueryRequest(
        query="越南 2024 年 GDP 是多少？",
        knowledge_base_id=KB_ID,
        answer_mode="agent",
    )
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))

    assert len(retrieval.calls) == 1
    complete = next(event for event in events if event.event == "complete")
    assert complete.data["agent_supplementary_retrieval"] is False
    assert complete.data["agent_evidence_satisfied"] is True


def test_m9_supplementary_retrieval_failure_keeps_initial_candidates() -> None:
    initial = (_candidate_with_snippet(1, "2024年 越南 GDP 4760 亿美元"),)
    retrieval = _SequencedRetrieval(
        [_retrieval_result(list(initial))],
        fail_after=1,
    )
    gateway = _FakeGateway(plans=[{"action": "final"}], deltas=["仅根据现有证据回答。"])
    service = _service(
        gateway=gateway,
        retrieval=retrieval,
        executor=_StubExecutor([]),
        settings=_settings(
            agent_evidence_sufficiency_enabled=True,
            agent_max_rounds=1,
        ),
    )
    request = RAGQueryRequest(
        query="比较越南和泰国 2024 年 GDP。",
        knowledge_base_id=KB_ID,
        answer_mode="agent",
    )
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))

    complete = next(event for event in events if event.event == "complete")
    assert complete.data["agent_supplementary_retrieval"] is False
    assert complete.data["agent_supplementary_error"] == "RuntimeError"
    assert len(complete.data["agent_candidates"]) == 1
    statuses = [
        event.data["status"]
        for event in events
        if event.event == "tool_status" and event.data["tool"] == "retrieval"
    ]
    assert statuses == ["running", "completed", "running", "failed"]


def test_m9_auto_calls_visual_tool_when_planner_wants_to_finish() -> None:
    candidates = (_candidate(1, marker="C1", modality="image"),)
    executor = _StubExecutor([_completed_outcome()])
    gateway = _FakeGateway(
        plans=[
            {"action": "final", "reason": "图中有柱状图"},
            {"action": "final", "reason": "工具已完成"},
        ],
        deltas=["图中显示上升趋势。"],
    )
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result(list(candidates))),
        executor=executor,
        settings=_settings(agent_visual_tool_auto=True),
    )
    request = RAGQueryRequest(
        query="分析图1中柱状图的变化趋势和结构关系。",
        knowledge_base_id=KB_ID,
        answer_mode="agent",
    )
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))

    assert executor.calls == [("analyze_image", 1)]
    complete = next(event for event in events if event.event == "complete")
    assert complete.data["agent_visual_auto_called"] is True
    analyze_trace = [
        item for item in complete.data["agent_tool_trace"] if item["tool"] == "analyze_image"
    ]
    assert analyze_trace and analyze_trace[0]["parameters"]["auto_visual"] is True


def test_m9_same_image_with_different_question_is_not_deduplicated() -> None:
    candidates = (_candidate(1, marker="C1", modality="image"),)
    executor = _StubExecutor([_completed_outcome(), _completed_outcome()])
    gateway = _FakeGateway(
        plans=[
            {
                "action": "call_tool",
                "tool": "analyze_image",
                "evidence_index": 1,
                "vision_task": "describe_image",
                "question": "图里有哪些元素？",
            },
            {
                "action": "call_tool",
                "tool": "analyze_image",
                "evidence_index": 1,
                "vision_task": "describe_image",
                "question": "图里的趋势是什么？",
            },
            {"action": "final"},
        ],
        deltas=["完成。"],
    )
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result(list(candidates))),
        executor=executor,
        settings=_settings(agent_visual_tool_auto=False),
    )
    request = RAGQueryRequest(
        query="看看这张图。",
        knowledge_base_id=KB_ID,
        answer_mode="agent",
    )
    anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))

    assert executor.calls == [("analyze_image", 1), ("analyze_image", 1)]


class _PlanCapturingExecutor:
    def __init__(self) -> None:
        self.asset_indices: list[int] = []

    async def analyze_image(self, candidate: AgentCandidate, plan: AgentPlan) -> AgentToolOutcome:
        self.asset_indices.append(plan.asset_index)
        return _analyze_image_outcome(asset_index=plan.asset_index)

    async def analyze_table(
        self,
        candidate: AgentCandidate,
        plan: AgentPlan,
        second_candidate: AgentCandidate | None = None,
    ) -> AgentToolOutcome:
        raise AssertionError("unexpected table execution")


def test_run_tool_normalizes_planner_asset_index_to_candidate_asset() -> None:
    cases = [(1, 0, 1), (0, 1, 0), (1, 1, 1)]
    for candidate_asset, plan_asset, expected in cases:
        candidate = replace(
            _candidate(1, marker="C1", evidence_id="evidence_v1_" + "a" * 64),
            asset_index=candidate_asset,
        )
        executor = _PlanCapturingExecutor()
        service = _service(
            gateway=_FakeGateway(plans=[], deltas=[]),
            retrieval=_StubRetrieval(_retrieval_result([candidate])),
            executor=executor,
        )
        plan = AgentPlan(
            action="call_tool",
            tool="analyze_image",
            evidence_index=1,
            asset_index=plan_asset,
            vision_task="describe_image",
            question="图片里有什么？",
        )
        outcome = anyio.run(service._run_tool, executor, candidate, plan, None)
        assert executor.asset_indices == [expected]
        assert outcome.payload["asset_index"] == expected


def test_run_tool_keeps_distinct_asset_indices_of_same_evidence() -> None:
    evidence_id = "evidence_v1_" + "b" * 64
    candidate = replace(_candidate(1, marker="C1", evidence_id=evidence_id), asset_index=1)
    executor = _PlanCapturingExecutor()
    service = _service(
        gateway=_FakeGateway(plans=[], deltas=[]),
        retrieval=_StubRetrieval(_retrieval_result([candidate])),
        executor=executor,
    )
    plan = AgentPlan(
        action="call_tool",
        tool="analyze_image",
        evidence_index=1,
        asset_index=0,
        vision_task="analyze_chart",
    )
    anyio.run(service._run_tool, executor, candidate, plan, None)
    assert executor.asset_indices == [1]


def test_mixed_table_and_visual_completion_is_not_source_verified() -> None:
    table_candidate = _candidate(
        1,
        marker="C1",
        modality="table",
        evidence_id="evidence_v1_" + "c" * 64,
        table_columns=("年份", "GDP"),
    )
    image_candidate = _candidate(
        2,
        marker="C2",
        modality="image",
        evidence_id="evidence_v1_" + "d" * 64,
    )
    gateway = _FakeGateway(
        plans=[
            {
                "action": "call_tool",
                "tool": "analyze_table",
                "evidence_index": 1,
                "operation": "growth",
                "parameters": {
                    "axis": "columns",
                    "row": 0,
                    "base_column": 1,
                    "target_column": 2,
                },
            },
            {
                "action": "call_tool",
                "tool": "analyze_image",
                "evidence_index": 2,
                "vision_task": "analyze_chart",
            },
            {"action": "final", "reason": "证据足够"},
        ],
        deltas=["表格增长率为 10.6977%。", "两国 2024 年 GDP 相差约 440 亿美元。"],
    )
    service = _service(
        gateway=gateway,
        retrieval=_StubRetrieval(_retrieval_result([table_candidate, image_candidate])),
        executor=_StubExecutor(
            [_service_table_outcome(), _service_image_outcome(), _service_image_outcome()]
        ),
    )
    request = RAGQueryRequest(
        query="比较越南和泰国 2024 年 GDP，并说明柱状图趋势。",
        knowledge_base_id=KB_ID,
        answer_mode="agent",
    )
    events = anyio.run(lambda: _collect(service.stream(request, document_loader=_no_document)))
    complete = next(event for event in events if event.event == "complete")
    assert complete.data["fact_status"] in {"visual_estimate", "unverified"}
    assert complete.data["fact_status"] != "source_verified"
    assert complete.data["safety_gate"] == "visual_unverified"
    assert complete.data["safety_notice"]
