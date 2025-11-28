# ruff: noqa: RUF001, RUF003
"""Unit tests for the strict M5 evaluation scoring logic."""

from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "m5_agent_eval.py"
_SPEC = importlib.util.spec_from_file_location("m5_agent_eval", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
m5 = importlib.util.module_from_spec(_SPEC)
sys.modules["m5_agent_eval"] = m5
_SPEC.loader.exec_module(m5)

KB_ID = "kb-1"
DOC_A = "doc-a"
DOC_B = "doc-b"
EVIDENCE_A = "evidence_v1_" + "a" * 64
EVIDENCE_B = "evidence_v1_" + "b" * 64
LABELS = {DOC_A: "A", DOC_B: "B"}


def _citation(marker: str, document_id: str, evidence_id: str) -> dict[str, object]:
    return {
        "marker": marker,
        "knowledge_base_id": KB_ID,
        "document_id": document_id,
        "candidate_kind": "evidence",
        "evidence_id": evidence_id,
        "modality": "table",
        "representation_ids": ["representation_v1_" + "c" * 64],
        "page_start": 1,
        "page_end": 1,
        "source_block_ids": ["p1-b1"],
    }


def _raw(
    answer: str,
    *,
    citations: list[dict[str, object]] | None = None,
    trace: list[dict[str, object]] | None = None,
    status: str = "completed",
    errors: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "answer": answer,
        "citations": citations or [],
        "complete": {
            "status": status,
            "agent_tool_trace": trace or [],
            "tool_calls": 1 if trace else 0,
        },
        "errors": errors or [],
        "tool_statuses": [],
        "latency_ms": 1000.0,
    }


def _table_trace(
    *,
    operation: str = "compare",
    document_id: str = DOC_A,
    compare_document_id: str = DOC_B,
    status: str = "completed",
    error_code: str | None = None,
    summary: str = "表格计算完成（compare）",
) -> list[dict[str, object]]:
    parameters: dict[str, object] = {
        "operation": operation,
        "document_id": document_id,
        "evidence_id": EVIDENCE_A,
        "table": {"document_id": document_id, "evidence_id": EVIDENCE_A},
        "parameters": {},
    }
    if operation == "compare":
        parameters["second_document_id"] = compare_document_id
        parameters["second_evidence_id"] = EVIDENCE_B
        parameters["compare_table"] = {
            "document_id": compare_document_id,
            "evidence_id": EVIDENCE_B,
        }
    entry: dict[str, object] = {
        "tool": "analyze_table",
        "status": status,
        "summary": summary,
        "evidence_id": EVIDENCE_A,
        "document_id": document_id,
        "parameters": parameters,
    }
    if error_code:
        entry["error_code"] = error_code
    return [entry]


def _compare_question(**overrides: object) -> dict[str, object]:
    question: dict[str, object] = {
        "id": "compare-01",
        "category": "compare",
        "query": "对比 A B",
        "required_docs": ["A", "B"],
        "tools": {"required": ["analyze_table"], "allowed": ["analyze_table"]},
        "expected_operation": "compare",
        "expected_parameter_docs": ["A", "B"],
        "numbers": {"groups": [["4760"], ["4700"], ["60"], ["-100"]], "kind": "exact_calc"},
        "citation_required": True,
        "deterministic_calc": True,
    }
    question.update(overrides)
    return question


def _score(question: dict[str, object], raw: dict[str, object]):
    return m5.score_question(question, raw, knowledge_base_id=KB_ID, label_by_document=LABELS)


def test_extract_numbers_ignores_quarters_and_normalizes_percent() -> None:
    numbers = m5.extract_numbers("Q4 峰值约 210，全年增长 10.70%，差额 -1,234（单位：亿元）")
    assert Decimal("210") in numbers
    assert Decimal("10.70") in numbers
    assert Decimal("-1234") in numbers
    assert Decimal("4") not in numbers


def test_number_groups_require_all_expected_values_and_tolerance() -> None:
    assert m5.match_number_groups([Decimal("4760"), Decimal("4700")], [["4760"], ["4700"]])[0]
    matched, missing = m5.match_number_groups([Decimal("4760")], [["4760"], ["4700"]])
    assert not matched and missing == [["4700"]]
    # 10.70 must not satisfy 10.6977 with a tight tolerance.
    matched, _ = m5.match_number_groups([Decimal("10.70")], [["10.6977"]], tolerance="0.0001")
    assert not matched
    matched, _ = m5.match_number_groups([Decimal("10.6977")], [["10.6977"]], tolerance="0.0001")
    assert matched


def test_text_groups_accept_equivalent_formatting() -> None:
    normalized = m5.normalize_text("排列数为 n(n−1)，公式是 n(n-1)")
    matched, missing = m5.match_text_groups(normalized, [["n(n-1)", "n(n−1)"]])
    assert matched and not missing


def test_compare_question_passes_all_checks() -> None:
    raw = _raw(
        "越南 4760 对比 4700，绝对差异 60；泰国季度 -100。",
        citations=[
            _citation("C1", DOC_A, EVIDENCE_A),
            _citation("C2", DOC_B, EVIDENCE_B),
        ],
        trace=_table_trace(),
    )
    result = _score(_compare_question(), raw)
    assert result.checks["tool_selection"] is True
    assert result.checks["tool_execution"] is True
    assert result.checks["parameter_validity"] is True
    assert result.checks["numbers"] is True
    assert result.checks["exact_calculation"] is True
    assert result.checks["e2e"] is True


def test_forbidden_tool_fails_tool_selection_but_not_execution() -> None:
    raw = _raw(
        "越南 4760 对比 4700，绝对差异 60；泰国季度 -100。",
        citations=[
            _citation("C1", DOC_A, EVIDENCE_A),
            _citation("C2", DOC_B, EVIDENCE_B),
        ],
        trace=[
            *_table_trace(),
            {
                "tool": "analyze_image",
                "status": "completed",
                "summary": "图片理解完成",
                "document_id": DOC_A,
                "parameters": {"vision_task": "describe_image"},
            },
        ],
    )
    question = _compare_question(
        tools={
            "required": ["analyze_table"],
            "allowed": ["analyze_table"],
            "forbidden": ["analyze_image"],
        }
    )
    result = _score(question, raw)
    assert result.checks["tool_selection"] is False
    assert result.checks["tool_execution"] is True
    assert "forbidden_tool" in result.failure_codes
    assert result.checks["e2e"] is False


def test_tool_chosen_but_failed_is_separate_from_selection() -> None:
    raw = _raw(
        "无法完成对比。",
        citations=[],
        trace=_table_trace(
            status="failed",
            error_code="provider_error",
            summary="表格工具执行失败",
        ),
    )
    result = _score(_compare_question(), raw)
    assert result.checks["tool_selection"] is True
    assert result.checks["tool_execution"] is False
    assert result.checks["e2e"] is False


def test_parameter_operation_mismatch_fails_strict_parameter_check() -> None:
    raw = _raw(
        "越南 4760 对比 4700，绝对差异 60；泰国季度 -100。",
        citations=[
            _citation("C1", DOC_A, EVIDENCE_A),
            _citation("C2", DOC_B, EVIDENCE_B),
        ],
        trace=_table_trace(operation="select"),
    )
    result = _score(_compare_question(), raw)
    assert result.checks["tool_selection"] is True
    assert result.checks["parameter_validity"] is False
    assert "parameter_operation" in result.failure_codes


def test_missing_parameter_document_fails_strict_check() -> None:
    raw = _raw(
        "越南 4760 对比 4700，绝对差异 60；泰国季度 -100。",
        citations=[
            _citation("C1", DOC_A, EVIDENCE_A),
            _citation("C2", DOC_B, EVIDENCE_B),
        ],
        trace=_table_trace(compare_document_id=DOC_A),
    )
    result = _score(_compare_question(), raw)
    assert result.checks["parameter_validity"] is False
    assert "parameter_document" in result.failure_codes


def test_cross_document_coverage_requires_both_documents() -> None:
    raw = _raw(
        "越南 4760 对比 4700，绝对差异 60；泰国季度 -100。",
        citations=[_citation("C1", DOC_A, EVIDENCE_A)],
        trace=_table_trace(),
    )
    result = _score(_compare_question(), raw)
    assert result.checks["cited_document_coverage"] is False
    assert result.checks["e2e"] is False


def test_citation_structure_rejects_bad_evidence_id() -> None:
    bad = _citation("C1", DOC_A, "evidence_v1_short")
    raw = _raw(
        "越南 4760 对比 4700，绝对差异 60；泰国季度 -100。",
        citations=[bad, _citation("C2", DOC_B, EVIDENCE_B)],
        trace=_table_trace(),
    )
    result = _score(_compare_question(), raw)
    assert result.checks["citation_structure"] is False
    assert "citation_structure" in result.failure_codes


def test_controlled_division_by_zero_counts_as_tool_error_not_calculation_success() -> None:
    question = {
        "id": "growth-02",
        "category": "growth",
        "query": "文莱增长率",
        "required_docs": [],
        "tools": {"required": ["analyze_table"], "allowed": ["analyze_table"]},
        "expected_operation": "growth",
        "numbers": {"groups": [], "kind": "exact_calc"},
        "required_text_groups": [["无法", "不能"]],
        "expected_tool_error": "division_by_zero",
        "deterministic_calc": True,
    }
    raw = _raw(
        "基期值为 0，除法不可定义，无法计算增长率。",
        citations=[],
        trace=_table_trace(
            operation="growth",
            status="failed",
            error_code="division_by_zero",
            summary="表格工具执行失败：division_by_zero",
        ),
    )
    result = _score(question, raw)
    assert result.checks["tool_selection"] is True
    assert result.checks["tool_execution"] is True
    assert result.checks["exact_calculation"] is not True
    assert result.checks["e2e"] is True


def test_visual_refusal_rejects_invented_number() -> None:
    question = {
        "id": "image-03",
        "category": "image",
        "query": "最低季度具体数值？",
        "tools": {"required": ["analyze_image"], "allowed": ["analyze_image"]},
        "vision_task": "analyze_chart",
        "numbers": {"groups": [], "kind": "visual_refusal"},
        "required_text_groups": [["无法确定", "未标注"]],
        "forbid_any_number": True,
        "citation_required": True,
    }
    trace = [
        {
            "tool": "analyze_image",
            "status": "completed",
            "summary": "图表分析完成",
            "document_id": DOC_A,
            "parameters": {"vision_task": "analyze_chart"},
        }
    ]
    good = _raw(
        "图中未标注具体数值，无法确定最低季度。",
        citations=[_citation("C1", DOC_A, EVIDENCE_A)],
        trace=trace,
    )
    good_result = _score(question, good)
    assert good_result.checks["numbers"] is True
    assert good_result.checks["e2e"] is True
    bad = _raw(
        "最低季度是 Q1，约为 120。",
        citations=[_citation("C1", DOC_A, EVIDENCE_A)],
        trace=trace,
    )
    bad_result = _score(question, bad)
    assert bad_result.checks["numbers"] is False
    assert "unexpected_number" in bad_result.failure_codes


def test_visual_table_needs_review_is_not_exact_calculation() -> None:
    question = {
        "id": "image-table-01",
        "category": "image_table",
        "query": "提取图片表格",
        "tools": {"required": ["analyze_image"], "allowed": ["analyze_image"]},
        "vision_task": "extract_table",
        "numbers": {"groups": [["4300"], ["4760"]], "kind": "visual_number"},
        "needs_review_expected": True,
        "citation_required": True,
    }
    trace = [
        {
            "tool": "analyze_image",
            "status": "completed",
            "summary": "图片表格提取完成（qwen3-vl:8b），结果需人工核对",
            "document_id": DOC_A,
            "parameters": {"vision_task": "extract_table"},
        }
    ]
    raw = _raw(
        "越南 2023 年 4300，2024 年 4760（视觉模型提取，需人工核对）。",
        citations=[_citation("C1", DOC_A, EVIDENCE_A)],
        trace=trace,
    )
    result = _score(question, raw)
    assert result.checks["exact_calculation"] is None
    assert result.checks["needs_review_observed"] is True
    assert result.checks["e2e"] is True


def test_metrics_report_numerators_and_denominators() -> None:
    raw_good = _raw(
        "越南 4760 对比 4700，绝对差异 60；泰国季度 -100。",
        citations=[
            _citation("C1", DOC_A, EVIDENCE_A),
            _citation("C2", DOC_B, EVIDENCE_B),
        ],
        trace=_table_trace(),
    )
    good = _score(_compare_question(), raw_good)
    bad_raw = _raw("我不知道。", citations=[], trace=[])
    bad = _score(
        {
            **_compare_question(),
            "id": "table-01",
            "numbers": {"groups": [["999"]], "kind": "exact_calc"},
        },
        bad_raw,
    )
    metrics = m5.compute_metrics([good, bad])
    assert metrics["tool_selection_accuracy"] == {"passed": 1, "total": 2, "value": 0.5}
    # 参数有效性与工具执行率的统计口径只覆盖“确实调用了工具”的题目；
    # 未调用工具的题目已在 Tool Selection 中计为失败，不能重复计为参数无效。
    assert metrics["parameter_validity_strict"] == {"passed": 1, "total": 1, "value": 1.0}
    assert metrics["tool_execution_success_rate"] == {"passed": 1, "total": 1, "value": 1.0}
    assert metrics["exact_calculation_success_rate"] == {"passed": 1, "total": 2, "value": 0.5}
    assert metrics["citation_structure_validity"]["total"] == 2
    assert metrics["retrieval_hit_at_k"] == "UNVERIFIED"
    assert metrics["citation_semantic_support"] == "UNVERIFIED"
