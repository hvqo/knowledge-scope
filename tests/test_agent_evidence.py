# ruff: noqa: RUF001
"""Unit tests for the M9 evidence requirement and sufficiency layer."""

from __future__ import annotations

from uuid import UUID, uuid4

from knowledge_scope.agent.evidence import (
    EvidenceRequirement,
    assess_evidence,
    build_supplementary_query,
    extract_requirement,
    visual_feature_requested,
)
from knowledge_scope.agent.schemas import AgentCandidate
from knowledge_scope.agent.service import _merge_candidates
from knowledge_scope.rag.schemas import RAGCitation

KB_ID = uuid4()
DOC_A = uuid4()
DOC_B = uuid4()
EVIDENCE_A = "evidence_v1_" + "a" * 64
EVIDENCE_B = "evidence_v1_" + "b" * 64


def _citation(
    marker: str,
    *,
    document_id: UUID,
    modality: str = "text",
    evidence_id: str = EVIDENCE_A,
    snippet: str = "测试证据",
    page: int = 1,
) -> RAGCitation:
    return RAGCitation(
        marker=marker,
        document_id=document_id,
        candidate_kind="evidence" if evidence_id else "chunk",
        chunk_id=None if evidence_id else "chunk_v1_" + "c" * 64,
        knowledge_base_id=KB_ID,
        evidence_id=evidence_id or None,
        modality=modality if evidence_id else None,
        representation_ids=["representation_v1_" + "d" * 64] if evidence_id else [],
        asset_refs=[],
        page_start=page,
        page_end=page,
        source_block_ids=["p1-b1"],
        section_path=["测试"],
        section_title="测试",
        snippet=snippet,
        snippet_kind="representation" if evidence_id else "source",
        final_rank=1,
        final_reranker_score=0.9,
        branch_provenance=[],
    )


def _candidate(
    index: int,
    *,
    document_id: UUID = DOC_A,
    modality: str = "text",
    evidence_id: str = EVIDENCE_A,
    snippet: str = "测试证据",
    table_rows: tuple[str, ...] = (),
    table_columns: tuple[str, ...] = (),
) -> AgentCandidate:
    return AgentCandidate(
        index=index,
        marker=f"C{index}",
        citation=_citation(
            f"C{index}",
            document_id=document_id,
            modality=modality,
            evidence_id=evidence_id,
            snippet=snippet,
        ),
        document_title="报告.pdf",
        table_columns=table_columns,
        table_rows=table_rows,
    )


def test_extract_requirement_pulls_objects_years_metric_and_visual_intent() -> None:
    requirement = extract_requirement("比较越南和泰国 2020—2024 年 GDP，并分析各自统计图的趋势。")
    assert requirement.objects == ("越南", "泰国")
    assert requirement.years == (2020, 2024)
    assert requirement.range_years == (2020, 2024)
    assert requirement.metrics == ("GDP",)
    assert requirement.comparison is True
    assert requirement.needs_image_feature is True
    assert requirement.needs_table is True


def test_extract_requirement_simple_fact_has_no_visual_requirement() -> None:
    requirement = extract_requirement("越南 2024 年 GDP 是多少？")
    assert requirement.objects == ("越南",)
    assert requirement.metrics == ("GDP",)
    assert requirement.needs_image_feature is False
    assert requirement.comparison is False


def test_assess_evidence_reports_missing_comparison_object() -> None:
    requirement = extract_requirement("比较越南和泰国 2024 年的 GDP。")
    candidates = (
        _candidate(
            1,
            modality="table",
            snippet="越南 2024年 GDP 4760 亿美元",
            table_rows=("越南", "4760"),
            table_columns=("国家", "2024年GDP"),
        ),
    )
    assessment = assess_evidence(requirement, candidates)
    assert assessment.satisfied is False
    assert any("泰国" in item for item in assessment.missing)
    assert assessment.table_indices == (1,)


def test_assess_evidence_requires_image_for_visual_feature_question() -> None:
    requirement = extract_requirement("分析统计图中销售额的变化趋势和峰值季度。")
    text_candidates = (_candidate(1, snippet="季度销售额统计说明"),)
    assessment = assess_evidence(requirement, text_candidates)
    assert assessment.satisfied is False
    assert any("图片" in item for item in assessment.missing)
    image_candidates = (
        _candidate(1, modality="image", evidence_id=EVIDENCE_B, snippet="季度销售柱状图"),
    )
    assessment = assess_evidence(requirement, image_candidates)
    assert assessment.image_indices == (1,)
    assert not any("图片" in item for item in assessment.missing)


def test_supplementary_query_targets_missing_item_and_differs_from_question() -> None:
    requirement = extract_requirement("比较越南和泰国 2024 年的 GDP。")
    candidates = (_candidate(1, snippet="越南 2024年 GDP 4760 亿美元"),)
    assessment = assess_evidence(requirement, candidates)
    query = build_supplementary_query(requirement, assessment)
    assert "泰国" in query
    assert query != requirement.query
    assert len(query) <= 160


def test_merge_candidates_deduplicates_and_renumbers_markers() -> None:
    initial = (_candidate(1, evidence_id=EVIDENCE_A), _candidate(2, evidence_id=EVIDENCE_B))
    extra = (
        _candidate(1, evidence_id=EVIDENCE_A),
        _candidate(2, document_id=DOC_B, evidence_id=EVIDENCE_B, modality="table"),
    )
    merged, added = _merge_candidates(initial, extra, limit=6)
    assert added == 1
    assert [candidate.marker for candidate in merged] == ["C1", "C2", "C3"]
    assert [candidate.index for candidate in merged] == [1, 2, 3]
    assert merged[2].citation.document_id == DOC_B


def test_merge_candidates_keeps_initial_candidates_when_limit_reached() -> None:
    initial = tuple(_candidate(index, evidence_id=EVIDENCE_A) for index in range(1, 5))
    extra = (_candidate(1, document_id=DOC_B, evidence_id=EVIDENCE_B),)
    merged, added = _merge_candidates(initial, extra, limit=4)
    assert added == 0
    assert len(merged) == 4


def test_visual_feature_requested_ignores_title_only_question() -> None:
    assert visual_feature_requested("分析图中曲线趋势和峰值") is True
    assert visual_feature_requested("这张图的标题是什么？") is False


def test_empty_requirement_is_always_satisfied() -> None:
    requirement = EvidenceRequirement(query="你好")
    assessment = assess_evidence(requirement, ())
    assert assessment.satisfied is True
    assert assessment.missing == ()
