# ruff: noqa: RUF001
from __future__ import annotations

import json
from collections.abc import AsyncIterator
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
from knowledge_scope.rag.schemas import (
    RAGCitation,
    RAGCitationsData,
    RAGCompleteData,
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
    ) -> None:
        super().__init__(
            settings,
            gateway=gateway,  # type: ignore[arg-type]
            unified_retrieval=object(),  # type: ignore[arg-type]
            vision_service=object(),  # type: ignore[arg-type]
            table_service=object(),  # type: ignore[arg-type]
            rag_service=rag_service,
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

    async def retrieve(self, query: str, knowledge_base_id: UUID) -> Any:
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
) -> AgentChatService:
    return _StubAgentService(
        settings or _settings(),
        gateway=gateway,
        retrieval=retrieval,
        executor=executor,
        rag_service=rag_service,
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
    assert [item["marker"] for item in citation_event.data["items"]] == ["C1", "C2"]
    complete = next(event for event in events if event.event == "complete")
    assert complete.data["status"] == "completed"
    assert complete.data["answer_mode"] == "agent"
    assert complete.data["tool_calls"] == 1
    assert complete.data["route"] == "unified"
    assert complete.data["provider"] == "fake"


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
    assert statuses[-1]["status"] == "skipped"
    assert "answer_delta" in names
    complete = next(event for event in events if event.event == "complete")
    assert complete.data["answer_mode"] == "agent"
    assert complete.data["tool_calls"] == 0


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
